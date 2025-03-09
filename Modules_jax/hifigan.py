import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple, Any, Optional, List, Dict, Union, Callable
import math
import numpy as np
from functools import partial
import random
from modules_jax import AdaIN1d

LRELU_SLOPE = 0.1

# Utility functions
def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def init_weights(scale=0.01):
    def _init(key, shape, dtype):
        return scale * jax.random.normal(key, shape, dtype)
    return _init

def snake_activation(x, alpha):
    return x + (1.0 / alpha) * jnp.power(jnp.sin(alpha * x), 2)

def pad_diff(x):
    padded = jnp.pad(x, [(0, 0), (1, 0), (0, 0)], mode='constant')
    return jnp.pad(padded - x, [(0, 0), (0, 1), (0, 0)], mode='constant')

# Weight normalization for JAX
class WeightNorm(nn.Module):
    module: nn.Module
    
    def setup(self):
        self.module.param_dtype = self.param_dtype
        self.module.dtype = self.dtype
        
    def __call__(self, x, **kwargs):
        return self.module(x, **kwargs)

class Conv1d(nn.Module):
    features: int
    kernel_size: int
    stride: int = 1
    padding: str = 'SAME'
    dilation: int = 1
    use_bias: bool = True
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, inputs):
        padding = self.padding
        if isinstance(padding, int):
            padding = ((0, 0), (padding, padding), (0, 0))
        
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param(
            'kernel',
            init_weights(),
            (self.kernel_size, inputs.shape[-1], self.features),
            self.param_dtype,
        )
        kernel = jnp.asarray(kernel, self.dtype)
        
        if self.use_bias:
            bias = self.param(
                'bias',
                lambda rng, shape, dtype: jnp.zeros(shape, dtype),
                (self.features,),
                self.param_dtype,
            )
            bias = jnp.asarray(bias, self.dtype)
        else:
            bias = None
            
        result = jax.lax.conv_general_dilated(
            inputs,
            kernel,
            window_strides=[self.stride],
            padding=padding,
            lhs_dilation=[1],
            rhs_dilation=[self.dilation],
            dimension_numbers=('NCH', 'HIO', 'NCH'),
        )
        
        if bias is not None:
            result = result + bias
            
        return result

class ConvTranspose1d(nn.Module):
    features: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    output_padding: int = 0
    use_bias: bool = True
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, inputs):
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param(
            'kernel',
            init_weights(),
            (self.kernel_size, inputs.shape[-1], self.features),
            self.param_dtype,
        )
        kernel = jnp.asarray(kernel, self.dtype)
        
        if self.use_bias:
            bias = self.param(
                'bias',
                lambda rng, shape, dtype: jnp.zeros(shape, dtype),
                (self.features,),
                self.param_dtype,
            )
            bias = jnp.asarray(bias, self.dtype)
        else:
            bias = None
            
        output_shape = (inputs.shape[0], inputs.shape[1] * self.stride + self.output_padding, self.features)
            
        result = jax.lax.conv_transpose(
            inputs,
            kernel,
            strides=[self.stride],
            padding=[(self.padding, self.padding)],
            transpose_kernel=True
        )
        
        if bias is not None:
            result = result + bias[None, None, :]
            
        return result

class AdaINResBlock1(nn.Module):
    channels: int
    kernel_size: int = 3
    dilation: Tuple[int, int, int] = (1, 3, 5)
    style_dim: int = 64
    
    @nn.compact
    def __call__(self, x, s):
        # Build convolution layers
        convs1 = [
            Conv1d(self.channels, self.kernel_size, dilation=self.dilation[0], 
                   padding=get_padding(self.kernel_size, self.dilation[0])),
            Conv1d(self.channels, self.kernel_size, dilation=self.dilation[1], 
                   padding=get_padding(self.kernel_size, self.dilation[1])),
            Conv1d(self.channels, self.kernel_size, dilation=self.dilation[2], 
                   padding=get_padding(self.kernel_size, self.dilation[2]))
        ]
        
        convs2 = [
            Conv1d(self.channels, self.kernel_size, dilation=1, padding=get_padding(self.kernel_size, 1)),
            Conv1d(self.channels, self.kernel_size, dilation=1, padding=get_padding(self.kernel_size, 1)),
            Conv1d(self.channels, self.kernel_size, dilation=1, padding=get_padding(self.kernel_size, 1))
        ]
        
        adain1 = [
            AdaIN1d(self.style_dim, self.channels),
            AdaIN1d(self.style_dim, self.channels),
            AdaIN1d(self.style_dim, self.channels)
        ]
        
        adain2 = [
            AdaIN1d(self.style_dim, self.channels),
            AdaIN1d(self.style_dim, self.channels),
            AdaIN1d(self.style_dim, self.channels)
        ]
        
        # Parameter for Snake activation
        alpha1 = [self.param(f'alpha1_{i}', lambda key, shape, dtype: jnp.ones(shape), (1, self.channels, 1)) 
                 for i in range(len(convs1))]
        alpha2 = [self.param(f'alpha2_{i}', lambda key, shape, dtype: jnp.ones(shape), (1, self.channels, 1)) 
                 for i in range(len(convs2))]
        
        # Process through blocks
        orig_x = x
        for i in range(len(convs1)):
            xt = adain1[i](x, s)
            xt = snake_activation(xt, alpha1[i])
            xt = convs1[i](xt)
            xt = adain2[i](xt, s)
            xt = snake_activation(xt, alpha2[i])
            xt = convs2[i](xt)
            x = xt + x
        
        return x

class SineGen(nn.Module):
    samp_rate: int
    upsample_scale: int
    harmonic_num: int = 0
    sine_amp: float = 0.1
    noise_std: float = 0.003
    voiced_threshold: float = 0
    flag_for_pulse: bool = False
    
    def setup(self):
        self.dim = self.harmonic_num + 1
        self.sampling_rate = self.samp_rate
        
    @nn.compact
    def __call__(self, f0, train_rng=None):
        """
        f0: (batch_size, length, 1)
        """
        # Generate uv signal
        uv = jnp.float32(f0 > self.voiced_threshold)
        
        # Fundamental component
        f0_buf = jnp.zeros((*f0.shape[:-1], self.dim))
        
        # Generate harmonic overtones
        fn = f0 * jnp.arange(1, self.harmonic_num + 2, dtype=jnp.float32)[None, None, :]
        
        # Convert to F0 in rad
        rad_values = (fn / self.sampling_rate) % 1
        
        # Initial phase noise
        if train_rng is not None:
            rand_ini = jax.random.uniform(train_rng, (f0.shape[0], f0.shape[2]))
        else:
            rand_ini = jnp.random.uniform(size=(f0.shape[0], f0.shape[2]))
            
        rand_ini = rand_ini.at[:, 0].set(0)
        rad_values = rad_values.at[:, 0, :].set(rad_values[:, 0, :] + rand_ini[:, :, None])
        
        # For interpolation
        if not self.flag_for_pulse:
            # Interpolate rad_values for downsampling
            rad_values_reshape = jnp.transpose(rad_values, (0, 2, 1))
            
            # Linear interpolation for downsampling
            indices = jnp.arange(0, rad_values_reshape.shape[-1], self.upsample_scale)
            rad_values_ds = jnp.take(rad_values_reshape, indices, axis=-1)
            
            # Cumulative sum to get phase
            phase = jnp.cumsum(rad_values_ds, axis=-1) * 2 * jnp.pi
            
            # Upsample phase back
            target_len = rad_values_reshape.shape[-1]
            phase_up = jnp.zeros((*phase.shape[:-1], target_len))
            
            # Simple linear interpolation for upsampling
            for i in range(phase.shape[-1]-1):
                start_idx = i * self.upsample_scale
                end_idx = (i+1) * self.upsample_scale
                slope = (phase[..., i+1] - phase[..., i]) / self.upsample_scale
                for j in range(self.upsample_scale):
                    phase_up = phase_up.at[..., start_idx+j].set(
                        phase[..., i] + slope * j
                    )
            
            # Fill the last part
            last_start = (phase.shape[-1]-1) * self.upsample_scale
            if last_start < target_len:
                phase_up = phase_up.at[..., last_start:].set(phase[..., -1:])
                
            phase = jnp.transpose(phase_up, (0, 2, 1))
            sines = jnp.sin(phase)
            
        else:
            # Implementation for pulse train generation
            uv_1 = jnp.roll(uv, -1, axis=1)
            uv_1 = uv_1.at[:, -1, :].set(1)
            u_loc = (uv < 1) & (uv_1 > 0)
            
            # Cumulative sum for phase
            tmp_cumsum = jnp.cumsum(rad_values, axis=1)
            
            # Process voiced/unvoiced segments
            i_phase = jnp.cumsum(rad_values, axis=1)
            sines = jnp.cos(i_phase * 2 * jnp.pi)
            
        # Generate sine waves
        sine_waves = sines * self.sine_amp
        
        # Generate noise
        if train_rng is not None:
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise_rng, train_rng = jax.random.split(train_rng)
            noise = noise_amp * jax.random.normal(noise_rng, sine_waves.shape)
        else:
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * jnp.random.normal(size=sine_waves.shape)
        
        # Apply noise to sine waves
        sine_waves = sine_waves * uv + noise
        
        return sine_waves, uv, noise

class SourceModuleHnNSF(nn.Module):
    sampling_rate: int
    upsample_scale: int
    harmonic_num: int = 0
    sine_amp: float = 0.1
    add_noise_std: float = 0.003
    voiced_threshold: float = 0
    
    @nn.compact
    def __call__(self, x, train_rng=None):
        # Create sine generator
        sine_gen = SineGen(
            self.sampling_rate,
            self.upsample_scale,
            self.harmonic_num,
            self.sine_amp,
            self.add_noise_std,
            self.voiced_threshold
        )
        
        # Generate sine waves
        if train_rng is not None:
            sine_wavs, uv, _ = sine_gen(x, train_rng=train_rng)
        else:
            sine_wavs, uv, _ = sine_gen(x)
        
        # Merge sine waves
        linear = nn.Dense(1)
        sine_merge = jnp.tanh(linear(sine_wavs))
        
        # Generate noise source
        if train_rng is not None:
            noise_rng, train_rng = jax.random.split(train_rng)
            noise = jax.random.normal(noise_rng, uv.shape) * self.sine_amp / 3
        else:
            noise = jnp.random.normal(size=uv.shape) * self.sine_amp / 3
            
        return sine_merge, noise, uv

class Generator(nn.Module):
    style_dim: int
    resblock_kernel_sizes: Sequence[int]
    upsample_rates: Sequence[int]
    upsample_initial_channel: int
    resblock_dilation_sizes: Sequence[Sequence[int]]
    upsample_kernel_sizes: Sequence[int]
    
    @nn.compact
    def __call__(self, x, s, f0, train_rng=None):
        num_kernels = len(self.resblock_kernel_sizes)
        num_upsamples = len(self.upsample_rates)
        
        # Source module
        m_source = SourceModuleHnNSF(
            sampling_rate=24000,
            upsample_scale=int(np.prod(self.upsample_rates)),
            harmonic_num=8, 
            voiced_threshold=10
        )
        
        # F0 upsampling
        prod_upsample = int(np.prod(self.upsample_rates))
        f0_upsampled = jax.image.resize(
            f0[:, None, :], 
            (f0.shape[0], 1, f0.shape[1] * prod_upsample), 
            method='linear'
        )
        f0_upsampled = jnp.transpose(f0_upsampled, (0, 2, 1))
        
        # Generate source
        if train_rng is not None:
            har_source, noi_source, uv = m_source(f0_upsampled, train_rng=train_rng)
        else:
            har_source, noi_source, uv = m_source(f0_upsampled)
            
        har_source = jnp.transpose(har_source, (0, 2, 1))
        
        # Noise convolutions and residual blocks
        noise_convs = []
        noise_res_blocks = []
        
        for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            c_cur = self.upsample_initial_channel // (2 ** (i + 1))
            
            if i + 1 < len(self.upsample_rates):
                stride_f0 = int(np.prod(self.upsample_rates[i+1:]))
                noise_convs.append(Conv1d(
                    c_cur, 
                    kernel_size=stride_f0 * 2, 
                    stride=stride_f0, 
                    padding=(stride_f0+1) // 2
                ))
                noise_res_blocks.append(AdaINResBlock1(c_cur, 7, [1, 3, 5], self.style_dim))
            else:
                noise_convs.append(Conv1d(c_cur, kernel_size=1, stride=1))
                noise_res_blocks.append(AdaINResBlock1(c_cur, 11, [1, 3, 5], self.style_dim))
        
        # Upsampling layers
        ups = []
        for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            ups.append(ConvTranspose1d(
                self.upsample_initial_channel // (2**(i+1)),
                k, 
                stride=u, 
                padding=(u//2 + u%2), 
                output_padding=u%2
            ))
        
        # Residual blocks
        resblocks = []
        for i in range(len(self.upsample_rates)):
            ch = self.upsample_initial_channel // (2**(i+1))
            for j, (k, d) in enumerate(zip(self.resblock_kernel_sizes, self.resblock_dilation_sizes)):
                resblocks.append(AdaINResBlock1(ch, k, d, self.style_dim))
        
        # Snake activation alphas
        alphas = []
        alphas.append(self.param('alpha_0', lambda key, shape, dtype: jnp.ones(shape), 
                                (1, self.upsample_initial_channel, 1)))
        
        for i in range(len(ups)):
            ch = self.upsample_initial_channel // (2**(i+1))
            alphas.append(self.param(f'alpha_{i+1}', lambda key, shape, dtype: jnp.ones(shape), 
                                    (1, ch, 1)))
        
        # Final convolution
        conv_post = Conv1d(1, 7, 1, padding=3)
        
        # Forward pass
        for i in range(num_upsamples):
            x = snake_activation(x, alphas[i])
            x_source = noise_convs[i](har_source)
            x_source = noise_res_blocks[i](x_source, s)
            
            x = ups[i](x)
            x = x + x_source
            
            xs = None
            for j in range(num_kernels):
                if xs is None:
                    xs = resblocks[i*num_kernels+j](x, s)
                else:
                    xs += resblocks[i*num_kernels+j](x, s)
            
            x = xs / num_kernels
            
        x = snake_activation(x, alphas[num_upsamples])
        x = conv_post(x)
        x = jnp.tanh(x)
        
        return x

class UpSample1d(nn.Module):
    layer_type: str
    
    @nn.compact
    def __call__(self, x):
        if self.layer_type == 'none':
            return x
        else:
            # Scale factor 2 nearest neighbor upsampling
            return jax.image.resize(x, (x.shape[0], x.shape[1], x.shape[2] * 2), method='nearest')

class AdainResBlk1d(nn.Module):
    dim_in: int
    dim_out: int
    style_dim: int = 64
    upsample: str = 'none'
    dropout_p: float = 0.0
    
    @nn.compact
    def __call__(self, x, s, train=True):
        actv = lambda x: jax.nn.leaky_relu(x, 0.2)
        upsample_layer = UpSample1d(self.upsample)
        learned_sc = self.dim_in != self.dim_out
        
        # Build weights
        norm1 = AdaIN1d(self.style_dim, self.dim_in)
        norm2 = AdaIN1d(self.style_dim, self.dim_out)
        conv1 = Conv1d(self.dim_out, 3, 1, padding=1)
        conv2 = Conv1d(self.dim_out, 3, 1, padding=1)
        
        # Shortcut path
        x_skip = upsample_layer(x)
        if learned_sc:
            conv1x1 = Conv1d(self.dim_out, 1, 1, padding=0, use_bias=False)
            x_skip = conv1x1(x_skip)
            
        # Residual path
        if self.upsample == 'none':
            pool = lambda x: x  # Identity
        else:
            # Transposed conv as in PyTorch version
            kernel_size = 3
            stride = 2
            padding = 1
            output_padding = 1
            pool = ConvTranspose1d(self.dim_in, kernel_size, stride, padding, output_padding, groups=self.dim_in)
        
        # Process through residual path
        h = norm1(x, s)
        h = actv(h)
        h = pool(h)
        
        # Apply dropout
        if train and self.dropout_p > 0:
            dropout_rng = self.make_rng('dropout')
            h = jax.random.uniform(dropout_rng, h.shape) > self.dropout_p
            h = h * jnp.array(1.0 / (1 - self.dropout_p), h.dtype)
        
        h = conv1(h)
        h = norm2(h, s)
        h = actv(h)
        
        # Apply dropout again
        if train and self.dropout_p > 0:
            dropout_rng = self.make_rng('dropout')
            h = jax.random.uniform(dropout_rng, h.shape) > self.dropout_p
            h = h * jnp.array(1.0 / (1 - self.dropout_p), h.dtype)
        
        h = conv2(h)
        
        # Combine residual and shortcut
        out = (h + x_skip) / math.sqrt(2)
        return out

class Decoder(nn.Module):
    dim_in: int = 512
    F0_channel: int = 512
    style_dim: int = 64
    dim_out: int = 80
    resblock_kernel_sizes: Sequence[int] = (3, 7, 11)
    upsample_rates: Sequence[int] = (10, 5, 3, 2)
    upsample_initial_channel: int = 512
    resblock_dilation_sizes: Sequence[Sequence[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    upsample_kernel_sizes: Sequence[int] = (20, 10, 6, 4)
    
    @nn.compact
    def __call__(self, asr, F0_curve, N, s, train=True):
        # F0 and energy downsampling for data augmentation during training
        if train:
            # Choose random downsampling factors
            # Note: Using fixed values for reproducibility in JAX
            F0_down_idx = hash(str(F0_curve.sum())) % 3
            N_down_idx = hash(str(N.sum())) % 4
            
            downlist_F0 = [0, 3, 7]
            downlist_N = [0, 3, 7, 15]
            
            F0_down = downlist_F0[F0_down_idx]
            N_down = downlist_N[N_down_idx]
            
            # Downsampling with convolution
            if F0_down > 0:
                kernel_F0 = jnp.ones((1, 1, F0_down)) / F0_down
                F0_curve = jax.lax.conv_general_dilated(
                    F0_curve[:, None, :],
                    kernel_F0,
                    window_strides=(1,),
                    padding=[(F0_down//2, F0_down//2)],
                    dimension_numbers=('NCH', 'IOH', 'NCH')
                )[:, 0, :]
                
            if N_down > 0:
                kernel_N = jnp.ones((1, 1, N_down)) / N_down
                N = jax.lax.conv_general_dilated(
                    N[:, None, :],
                    kernel_N,
                    window_strides=(1,),
                    padding=[(N_down//2, N_down//2)],
                    dimension_numbers=('NCH', 'IOH', 'NCH')
                )[:, 0, :]
        
        # F0 and N convolutional layers
        F0_conv = Conv1d(1, 3, 2, padding=1)
        N_conv = Conv1d(1, 3, 2, padding=1)
        
        # Process F0 and N
        F0 = F0_conv(F0_curve[:, None, :])
        N = N_conv(N[:, None, :])
        
        # Encoder block
        x = jnp.concatenate([asr, F0, N], axis=1)
        encode = AdainResBlk1d(self.dim_in + 2, 1024, self.style_dim)
        x = encode(x, s, train=train)
        
        # ASR residual connection
        asr_res = Conv1d(64, 1)(asr)
        
        # Decoder blocks
        decode_blocks = [
            AdainResBlk1d(1024 + 2 + 64, 1024, self.style_dim),
            AdainResBlk1d(1024 + 2 + 64, 1024, self.style_dim),
            AdainResBlk1d(1024 + 2 + 64, 1024, self.style_dim),
            AdainResBlk1d(1024 + 2 + 64, 512, self.style_dim, upsample=True)
        ]
        
        # Process through decoder
        res = True
        for block in decode_blocks:
            if res:
                x = jnp.concatenate([x, asr_res, F0, N], axis=1)
            x = block(x, s, train=train)
            
            # Check if this is an upsampling block
            if block.upsample != 'none':
                res = False
                
        # Generator
        generator = Generator(
            self.style_dim,
            self.resblock_kernel_sizes,
            self.upsample_rates,
            self.upsample_initial_channel,
            self.resblock_dilation_sizes,
            self.upsample_kernel_sizes
        )
        
        # Generate output
        x = generator(x, s, F0_curve)
        return x