import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Sequence, Tuple, Optional, Any, List, Union
import math
import functools
from scipy.signal import get_window
from modules_jax import AdaIN1d
from Modules_jax.utils import custom_stft, custom_istft, weight_norm_init
from norm_jax import weight_norm_jax

LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def custom_leaky_relu(x, negative_slope=0.01):
    """
    Custom Leaky ReLU activation for JAX.
    
    Args:
        x: Input tensor.
        negative_slope: Slope for negative values (default 0.01).
    
    Returns:
        Output tensor after applying Leaky ReLU.
    """
    return jnp.where(x >= 0, x, negative_slope * x)

def padDiff(x):
    # Equivalent to F.pad in PyTorch
    x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=0)
    return jnp.pad(x_pad - x, ((0, 0), (0, 0), (1, 0)), 'constant', constant_values=0)


class AdaINResBlock1(nn.Module):
    channels: int
    kernel_size: int = 3
    dilation: Tuple[int, int, int] = (1, 3, 5)
    style_dim: int = 64
    
    @nn.compact
    def __call__(self, x, s, train: bool = True):
        # residual = x
        
        # In PyTorch, the implementation uses ModuleList and zips components together
        # In JAX, we'll implement the same logic with explicit loops
        for i, d in enumerate(self.dilation):
            # First stage: normalization -> activation -> convolution
            xt = AdaIN1d(self.style_dim, self.channels, name=f'adain1_{i}')(x, s)
            
            # Snake1D activation (equivalent to PyTorch version)
            alpha1 = self.param(f'alpha1_{i}', nn.initializers.ones, (1, 1, self.channels))
            xt = xt + (1 / alpha1) * (jnp.sin(alpha1 * xt) ** 2)
            
            # First convolution
            # padding = get_padding(self.kernel_size, d)
            xt = weight_norm_jax(nn.Conv(
                features=self.channels,
                kernel_size=(self.kernel_size,),
                # padding=((padding, padding),),
                padding='SAME', # This maintains dimensions automatically
                kernel_dilation=(d,),
                use_bias=True,
                kernel_init=weight_norm_init(),
                bias_init=nn.initializers.zeros
            ))(xt)
            
            # Second stage: normalization -> activation -> convolution
            xt = AdaIN1d(self.style_dim, self.channels, name=f'adain2_{i}')(xt, s)
            
            # Snake1D activation again
            alpha2 = self.param(f'alpha2_{i}', nn.initializers.ones, (1, 1, self.channels))
            xt = xt + (1 / alpha2) * (jnp.sin(alpha2 * xt) ** 2)
            
            # padding = get_padding(self.kernel_size, 1)
            # Second convolution
            xt = weight_norm_jax(nn.Conv(
                features=self.channels,
                kernel_size=(self.kernel_size,),
                # padding=((padding, padding),),
                padding='SAME',
                kernel_dilation=(1,),
                use_bias=True,
                kernel_init=weight_norm_init(),
                bias_init=nn.initializers.zeros
            ))(xt)
            
            # Add to the input (residual connection)
            x = xt + x
            
        return x


class TorchSTFT(nn.Module):
    filter_length: int = 800
    hop_length: int = 200
    win_length: int = 800
    window_type: str = 'hann'

    def setup(self):
        self.window = jnp.array(get_window(self.window_type, self.win_length, fftbins=True).astype(np.float32))

    def transform(self, input_data):
        forward_transform = custom_stft(input_data, self.filter_length, self.hop_length, 
                          self.win_length, self.window, return_complex=True)
        return jnp.abs(forward_transform), jnp.angle(forward_transform)

    def inverse(self, magnitude, phase):
        complex_spec = magnitude * jnp.exp(phase * 1j)
        signal = custom_istft(complex_spec.transpose(0, 2, 1), self.filter_length, self.hop_length, 
                             self.win_length, self.window)
        return signal[:, :, None]  # Add dimension to match PyTorch implementation

    def __call__(self, input_data):
        magnitude, phase = self.transform(input_data)
        reconstruction = self.inverse(magnitude, phase)
        return reconstruction

class SineGen(nn.Module):
    samp_rate: int
    upsample_scale: int
    harmonic_num: int = 0
    sine_amp: float = 0.1
    noise_std: float = 0.003
    voiced_threshold: float = 0
    flag_for_pulse: bool = False # set to False for SineGen, don't use True (PulseGen)

    def _f02uv(self, f0):
        # Generate UV signal
        uv = jnp.array(f0 > self.voiced_threshold, dtype=jnp.float32)
        return uv

    def _f02sine(self, f0_values):
        # Convert to F0 in rad
        rad_values = (f0_values / self.samp_rate) % 1
        
        # Initial phase noise
        key = self.make_rng('sine_gen')
        rand_ini = jax.random.uniform(key, shape=(f0_values.shape[0], f0_values.shape[2]))
        rand_ini = jnp.pad(rand_ini[:, 1:], ((0, 0), (1, 0)), constant_values=0)
        rad_values = rad_values.at[:, 0, :].add(rand_ini)

        # Interpolate rad values
        rad_values_rs = jax.image.resize(
            rad_values, 
            (rad_values.shape[0], rad_values.shape[1] // self.upsample_scale, rad_values.shape[2]), 
            method='linear'
        )
        
        # Calculate phase
        phase = jnp.cumsum(rad_values_rs, axis=1) * 2 * np.pi
        
        # Interpolate back to original size
        phase = jax.image.resize(
            phase * self.upsample_scale, 
            (phase.shape[0], phase.shape[1] * self.upsample_scale, phase.shape[2]), 
            method='linear'
        )
        
        # Generate sine waves
        sines = jnp.sin(phase)
        return sines

    def __call__(self, f0):
        # Create f0_buf with harmonic frequencies
        f0_buf = jnp.zeros((f0.shape[0], f0.shape[1], self.harmonic_num + 1), dtype=jnp.float32)
        
        # Generate harmonic frequencies
        for idx in range(self.harmonic_num + 1):
            f0_buf = f0_buf.at[:, :, idx].set(f0[:, :, 0] * (idx + 1))
        
        # Generate sine waves and UV signal
        sine_waves = self._f02sine(f0_buf) * self.sine_amp
        uv = self._f02uv(f0)
        
        
        # Add noise
        key = self.make_rng('noise')
        # Calculate noise amplitude with expanded uv for proper broadcasting
        noise_amp = uv* self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * jax.random.normal(key, sine_waves.shape)
        
        # Combine sine and noise using expanded uv
        sine_waves = sine_waves * uv + noise
        
        # Return all three values needed by SourceModuleHnNSF
        return sine_waves, uv, noise

class SourceModuleHnNSF(nn.Module):
    sampling_rate: int
    upsample_scale: int
    harmonic_num: int = 0
    sine_amp: float = 0.1
    add_noise_std: float = 0.003
    voiced_threshold: float = 0

    def setup(self):
        
        # Sine wave generator
        self.l_sin_gen = SineGen(
            samp_rate=self.sampling_rate,
            upsample_scale=self.upsample_scale,
            harmonic_num=self.harmonic_num,
            sine_amp=self.sine_amp,
            noise_std=self.add_noise_std,
            voiced_threshold=self.voiced_threshold
        )
        
        # Linear layer to merge harmonics
        self.l_linear = nn.Dense(1)
        
    def __call__(self, x):
        # Generate sine source
        sine_wavs, uv, _ = self.l_sin_gen(x) # x : [B, T, 1]
        sine_merge = jnp.tanh(self.l_linear(sine_wavs))
        
        # Generate noise source
        key = self.make_rng('noise')
        noise = jax.random.normal(key, uv.shape) * self.sine_amp / 3
        
        return sine_merge, noise, uv

class UpSample1d(nn.Module):
    layer_type: str

    @nn.compact
    def __call__(self, x):
        if self.layer_type == 'none':
            return x
        else:
            # Nearest neighbor upsampling (on the spatial(time) axis)
            shape = list(x.shape)
            shape[1] = shape[1] * 2
            return jax.image.resize(x, shape, method='nearest')

class GroupedConvTranspose(nn.Module):
    features: int
    kernel_size: Union[Tuple[int, ...], int]
    strides: Union[Tuple[int, ...], int] = (1,)
    padding: str = 'SAME'
    feature_group_count: int = 1
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        kernel_size = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size,)
        strides = self.strides if isinstance(self.strides, tuple) else (self.strides,)
        
        # Manually implement grouped transposed convolution
        if self.feature_group_count == 1:
            return weight_norm_jax(nn.ConvTranspose(
                features=self.features,
                kernel_size=kernel_size,
                strides=strides,
                padding=self.padding,
                use_bias=self.use_bias
            ))(x)
        else:
            # Split input along channel dimension
            x_groups = jnp.split(x, self.feature_group_count, axis=2)
            result_groups = []
            
            # Process each group separately
            features_per_group = self.features // self.feature_group_count
            for i in range(self.feature_group_count):
                result = weight_norm_jax(nn.ConvTranspose(
                    features=features_per_group,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=self.padding,
                    use_bias=self.use_bias
                ))(x_groups[i])
                result_groups.append(result)
            
            # Concatenate results
            return jnp.concatenate(result_groups, axis=2)

class AdainResBlk1d(nn.Module):
    dim_in: int
    dim_out: int
    style_dim: int = 64
    upsample: str = 'none'
    dropout_p: float = 0.0

    def setup(self):
        self.upsample_type = self.upsample
        self.learned_sc = self.dim_in != self.dim_out
        
        # Layers
        self.norm1 = AdaIN1d(self.style_dim, self.dim_in)
        self.norm2 = AdaIN1d(self.style_dim, self.dim_out)
        
        self.conv1 = weight_norm_jax(nn.Conv(
            features=self.dim_out,
            kernel_size=(3,),
            padding=((1, 1),)
        ))
        self.conv2 = weight_norm_jax(nn.Conv(
            features=self.dim_out,
            kernel_size=(3,),
            padding=((1, 1),)
        ))
        
        self.upsample_layer = UpSample1d(self.upsample)
        
        if self.learned_sc:
            self.conv1x1 = weight_norm_jax(nn.Conv(
                features=self.dim_out,
                kernel_size=(1,),
                use_bias=False,
            ))
            
        # For upsampling in residual block
        if self.upsample == 'none':
            # Identity function
            self.pool = lambda x: x
        else:
            # Use ConvTranspose with weight normalization for depthwise transposed conv
            self.pool = GroupedConvTranspose(
                features=self.dim_in,  # Same input/output channels
                kernel_size=(3,),
                strides=(2,),
                padding='SAME',
                feature_group_count=self.dim_in  # Makes it depthwise (each channel has its own filter)
            )
    
    def _shortcut(self, x):
        x = self.upsample_layer(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x
    
    def _residual(self, x, s, train):
        x = self.norm1(x, s) # [B, T, C]
        x = custom_leaky_relu(x, LRELU_SLOPE) # [B, T, C]
        
        # Use the pool layer (either identity or ConvTranspose)
        x = self.pool(x)
        print('pool after shape:', x.shape)
        
        x = self.conv1(x if not train or self.dropout_p == 0 else 
                      nn.Dropout(rate=self.dropout_p)(x, deterministic=not train))
        x = self.norm2(x, s) # [B, T, 1024]
        x = custom_leaky_relu(x, LRELU_SLOPE)
        x = self.conv2(x if not train or self.dropout_p == 0 else 
                      nn.Dropout(rate=self.dropout_p)(x, deterministic=not train)) # [B, T, 1024]
        return x
    
    def __call__(self, x, s, train: bool = True):
        out = self._residual(x, s, train)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

class Generator(nn.Module):
    style_dim: int
    resblock_kernel_sizes: Sequence[int]
    upsample_rates: Sequence[int]
    upsample_initial_channel: int
    resblock_dilation_sizes: Sequence[Sequence[int]]
    upsample_kernel_sizes: Sequence[int]
    gen_istft_n_fft: int
    gen_istft_hop_size: int

    def setup(self):
        self.num_kernels = len(self.resblock_kernel_sizes)
        self.num_upsamples = len(self.upsample_rates)
        
        # Source module
        self.m_source = SourceModuleHnNSF(
            sampling_rate=24000,
            upsample_scale=np.prod(self.upsample_rates) * self.gen_istft_hop_size,
            harmonic_num=8,
            voiced_threshold=10
        )
        
        # Upsampling layers
        self.ups = [
            weight_norm_jax(nn.ConvTranspose(
                features=self.upsample_initial_channel // (2 ** (i + 1)),
                kernel_size=(k,),
                strides=(u,),
                # padding=(((k - u) // 2, (k - u) // 2),)
                padding='SAME',
            )) for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes))
        ]
        
        # Residual blocks
        self.resblocks = [
            AdaINResBlock1(self.upsample_initial_channel // (2 ** (i + 1)),
                           kernel_size=k,
                           dilation=tuple(d),
                           style_dim=self.style_dim)
            for i in range(len(self.ups)) for k, d in zip(self.resblock_kernel_sizes, self.resblock_dilation_sizes)
        ]

        # Noise processing
        self.noise_convs = [
            nn.Conv(
                features=self.upsample_initial_channel // (2 ** (i + 1)),
                kernel_size=(np.prod(self.upsample_rates[i + 1:]) * 2,) if i + 1 < len(self.upsample_rates) else (1,),
                strides=(np.prod(self.upsample_rates[i + 1:]),) if i + 1 < len(self.upsample_rates) else (1,),
                padding=((
                    (np.prod(self.upsample_rates[i + 1:]) + 1) // 2, 
                    (np.prod(self.upsample_rates[i + 1:]) + 1) // 2
                ),) if i + 1 < len(self.upsample_rates) else 'VALID')
            for i in range(len(self.upsample_rates))
        ]
        self.noise_res = [
            AdaINResBlock1(
                channels=self.upsample_initial_channel // (2 ** (i + 1)),
                kernel_size=7 if i + 1 < len(self.upsample_rates) else 11,
                dilation=(1, 3, 5),
                style_dim=self.style_dim
            )
            for i in range(len(self.upsample_rates))
        ]
        
        # Final layers
        self.post_n_fft = self.gen_istft_n_fft
        self.conv_post = weight_norm_jax(nn.Conv(
            features=self.post_n_fft + 2,
            kernel_size=(7,),
            padding=((3, 3),)
        ))
        
        # STFT/ISTFT
        self.stft = TorchSTFT(
            filter_length=self.gen_istft_n_fft,
            hop_length=self.gen_istft_hop_size,
            win_length=self.gen_istft_n_fft
        )

    def f0_upsamp(self, f0, scale_factor):
        # Upsampling f0
        shape = list(f0.shape)
        shape[-1] = shape[-1] * scale_factor
        return jax.image.resize(f0, shape, method='nearest')
    
    def __call__(self, x, s, f0, train: bool = True):
        '''
        x shape : (B, T, C)
        s shape : (B, C)
        f0 shape : (B, T)
        '''
        # F0 upsampling
        f0_upsamped = self.f0_upsamp(f0[:, None], np.prod(self.upsample_rates) * self.gen_istft_hop_size).transpose(0, 2, 1)
        
        # Generate harmonics and noise
        har_source, noi_source, uv = self.m_source(f0_upsamped) # (B, samples, 1)
        har_source = har_source.squeeze(-1) # (B, samples)
        har_spec, har_phase = self.stft.transform(har_source)
        har = jnp.concatenate([har_spec, har_phase], axis=1) # [B, F*2, frames]
        har = har.transpose(0, 2, 1) # [B, F, frames] -> [B, frames, F]
        
        # Process through upsampling blocks
        for i in range(self.num_upsamples):
            x = custom_leaky_relu(x, LRELU_SLOPE)
            x_source = self.noise_convs[i](har)
            print("Noise conv after shape x_source:", x_source.shape)
            x_source = self.noise_res[i](x_source, s) # [B, 600, 256]
            print("Noise res after shape x_source:", x_source.shape)
            
            print("Upsample before x shape:", x.shape) # [B, 50, 512]
            x = self.ups[i](x)
            print("Upsample after x shape:", x.shape) # [B, 1000, 256]
            if i == self.num_upsamples - 1:
                # Reflection padding
                x = jnp.pad(x, ((0, 0), (1, 0), (0, 0)), mode='reflect')
            
            x = x + x_source # [B, 1000, 256]
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, s)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels
        
        # Final processing
        x = custom_leaky_relu(x)
        x = self.conv_post(x)
        spec = jnp.exp(x[:, :, :self.post_n_fft // 2 + 1])
        phase = jnp.sin(x[:, :, self.post_n_fft // 2 + 1:])
        
        return self.stft.inverse(spec, phase)


class DecoderJax(nn.Module):
    dim_in: int = 512
    F0_channel: int = 512
    style_dim: int = 64
    dim_out: int = 80
    resblock_kernel_sizes: Sequence[int] = (3, 7, 11)
    upsample_rates: Sequence[int] = (10, 6)
    upsample_initial_channel: int = 512
    resblock_dilation_sizes: Sequence[Sequence[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    upsample_kernel_sizes: Sequence[int] = (20, 12)
    gen_istft_n_fft: int = 20
    gen_istft_hop_size: int = 5

    def setup(self):
        self.encode = AdainResBlk1d(self.dim_in + 2, 1024, self.style_dim)
        
        # Decoder blocks
        self.decode = [
            AdainResBlk1d(1024 + 2 + 64, 1024, self.style_dim),
            AdainResBlk1d(1024 + 2 + 64, 1024, self.style_dim),
            AdainResBlk1d(1024 + 2 + 64, 1024, self.style_dim),
            AdainResBlk1d(1024 + 2 + 64, 512, self.style_dim, upsample="nearest")
        ]
        
        # F0 and N convolutions
        self.F0_conv = nn.Conv(
            features=1,
            kernel_size=(3,),
            strides=(2,),
            padding=((1, 1),),
            feature_group_count=1
        )
        
        self.N_conv = nn.Conv(
            features=1,
            kernel_size=(3,),
            strides=(2,),
            padding=((1, 1),),
            feature_group_count=1
        )
        
        # ASR residual connection
        self.asr_res = nn.Conv(
            features=64,
            kernel_size=(1,)
        )
        
        # Generator
        self.generator = Generator(
            style_dim=self.style_dim,
            resblock_kernel_sizes=self.resblock_kernel_sizes,
            upsample_rates=self.upsample_rates,
            upsample_initial_channel=self.upsample_initial_channel,
            resblock_dilation_sizes=self.resblock_dilation_sizes,
            upsample_kernel_sizes=self.upsample_kernel_sizes,
            gen_istft_n_fft=self.gen_istft_n_fft,
            gen_istft_hop_size=self.gen_istft_hop_size
        )
    
    def __call__(self, asr, F0_curve, N, s, train: bool = True):
        # F0_curve shape : (B, T)
        # N shape : (B, T)
        # asr shape : (B, T, C)

        if train:
            # Random smoothing of F0 and N
            key = self.make_rng('smoothing')
            downlist_F0 = jnp.array([0, 3, 7])
            F0_down = downlist_F0[jax.random.randint(key, (), 0, 3)]
            
            downlist_N = jnp.array([0, 3, 7, 15])
            N_down = downlist_N[jax.random.randint(key, (), 0, 4)]
            
        # Apply moving average smoothing if needed
        if F0_down > 0:
            # Pad the input for valid convolution
            pad_width = F0_down // 2
            F0_padded = jnp.pad(F0_curve, ((0, 0), (pad_width, pad_width)), mode='reflect')
            
            # Use reduce_window for moving average
            F0_curve = jax.lax.reduce_window(
                F0_padded, 
                init_value=0.0,
                computation=jax.lax.add,
                window_dimensions=(1, F0_down),
                window_strides=(1, 1),
                padding='VALID'
            ) / F0_down
        
        if N_down > 0:
            # Pad the input for valid convolution
            pad_width = N_down // 2
            N_padded = jnp.pad(N, ((0, 0), (pad_width, pad_width)), mode='reflect')
            
            # Use reduce_window for moving average
            N = jax.lax.reduce_window(
                N_padded, 
                init_value=0.0,
                computation=jax.lax.add,
                window_dimensions=(1, N_down),
                window_strides=(1, 1),
                padding='VALID'
            ) / N_down

        # Process F0 and N through convolutional layers
        F0 = self.F0_conv(jnp.expand_dims(F0_curve, axis=-1))  # (B, T) -> (B, 1, T//2)
        N = self.N_conv(jnp.expand_dims(N, axis=-1))  # (B, T) -> (B, 1, T//2)
        
        # Concatenate ASR features with F0 and N
        x = jnp.concatenate([asr, F0, N], axis=-1) # (B, T, C + 2)
        x = self.encode(x, s)  # s is conditioned by AdaIN
        asr_res = self.asr_res(asr)
        
        # Apply decoder blocks
        res = True
        for block in self.decode:
            if res:
                x = jnp.concatenate([x, asr_res, F0, N], axis=-1)  # (B, 1090, T//2)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        
        # Generate final output using the generator
        x = self.generator(x, s, F0_curve, train=train)
        return x
