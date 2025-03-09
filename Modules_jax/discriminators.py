import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple, Any, Optional, List, Dict, Union
import math

from .utils import get_padding, init_weights, stft
from norm_jax import weight_norm_jax, spectral_norm_jax

LRELU_SLOPE = 0.1

class SpecDiscriminator(nn.Module):
    """Spectral discriminator."""
    fft_size: int = 1024
    shift_size: int = 120
    win_length: int = 600
    window: str = "hann_window"
    use_spectral_norm: bool = False
    
    @nn.compact
    def __call__(self, y):
        # Create layers
        norm_f = weight_norm_jax if self.use_spectral_norm == False else spectral_norm_jax
        discriminators = [
            norm_f(nn.Conv(32, (3, 9), (1, 1), ((1, 1), (4, 4)))),
            norm_f(nn.Conv(32, (3, 9), (1, 2), ((1, 1), (4, 4)))),
            norm_f(nn.Conv(32, (3, 9), (1, 2), ((1, 1), (4, 4)))),
            norm_f(nn.Conv(32, (3, 9), (1, 2), ((1, 1), (4, 4)))),
            norm_f(nn.Conv(32, (3, 3), (1, 1), ((1, 1), (1, 1)))),]
        
        out = norm_f(nn.Conv(1, (3, 3), (1, 1), ((1, 1), (1, 1))))
        
        # Process input
        fmap = []
        y = jnp.squeeze(y, axis=-1) # [B, sample, 1] -> [B, sample]
        y = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        y = jnp.expand_dims(y, axis=-1) # [B, frame, mel] -> [B, frame, mel, 1]
        
        # Apply discriminators
        for i, d in enumerate(discriminators):
            y = d(y)
            y = jax.nn.leaky_relu(y, LRELU_SLOPE)
            fmap.append(y)
        
        y = out(y)
        fmap.append(y)
        
        return jnp.reshape(y, (y.shape[0], -1)), fmap # y shape : [B, frame, mel, 1] -> [B, -1]

class MultiResSpecDiscriminator(nn.Module):
    """Multi-resolution spectral discriminator."""
    fft_sizes: Sequence[int] = (1024, 2048, 512)
    hop_sizes: Sequence[int] = (120, 240, 50)
    win_lengths: Sequence[int] = (600, 1200, 240)
    window: str = "hann_window"
    
    @nn.compact
    def __call__(self, y, y_hat):
        # Create discriminators
        discriminators = [
            SpecDiscriminator(self.fft_sizes[0], self.hop_sizes[0], self.win_lengths[0], self.window),
            SpecDiscriminator(self.fft_sizes[1], self.hop_sizes[1], self.win_lengths[1], self.window),
            SpecDiscriminator(self.fft_sizes[2], self.hop_sizes[2], self.win_lengths[2], self.window),
        ]
        
        # Process inputs
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for i, d in enumerate(discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorP(nn.Module):
    """Period discriminator."""
    period: int
    kernel_size: int = 5
    stride: int = 3
    use_spectral_norm: bool = False
    
    @nn.compact
    def __call__(self, x):
        # Create convolution layers
        norm_f = weight_norm_jax if self.use_spectral_norm == False else spectral_norm_jax
        convs = [
            norm_f(nn.Conv(32, (self.kernel_size, 1), (self.stride, 1), 
                  ((get_padding(5, 1), get_padding(5, 1)), (0, 0)))),
            norm_f(nn.Conv(128, (self.kernel_size, 1), (self.stride, 1),
                  ((get_padding(5, 1), get_padding(5, 1)), (0, 0)))),
            norm_f(nn.Conv(512, (self.kernel_size, 1), (self.stride, 1),
                  ((get_padding(5, 1), get_padding(5, 1)), (0, 0)))),
            norm_f(nn.Conv(1024, (self.kernel_size, 1), (self.stride, 1),
                  ((get_padding(5, 1), get_padding(5, 1)), (0, 0)))),
            norm_f(nn.Conv(1024, (self.kernel_size, 1), (1, 1),
                  ((2, 2), (0, 0)))),
        ]
    
        conv_post = norm_f(nn.Conv(1, (3, 1), (1, 1), ((1, 1), (0, 0))))
        
        # Process input
        fmap = []
        
        # 1d to 2d
        b, t, c = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            # Reflect padding
            x = jnp.pad(x, ((0, 0), (0, n_pad), (0, 0)), mode='reflect')
            t = t + n_pad
        x = x.reshape(b, t // self.period, self.period, c)
        
        for l in convs:
            x = l(x)
            x = jax.nn.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        
        x = conv_post(x)
        fmap.append(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        
        return x, fmap

class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator."""
    
    @nn.compact
    def __call__(self, y, y_hat):
        # Create discriminators with different periods
        discriminators = [
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ]
        
        # Process inputs
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for i, d in enumerate(discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class WavLMDiscriminator(nn.Module):
    """WavLM-based discriminator."""
    slm_hidden: int = 768
    slm_layers: int = 13
    initial_channel: int = 64
    use_spectral_norm: bool = False
    
    @nn.compact
    def __call__(self, x):
        # Create layers
        norm_f = weight_norm_jax if self.use_spectral_norm == False else spectral_norm_jax
        # Fix padding format - convert integers to proper JAX padding format
        # KEY FIX: Input channels should be slm_hidden * slm_layers
        pre = norm_f(nn.Conv(
            features=self.initial_channel,
            kernel_size=(1,),
            strides=(1,),
            padding='VALID'  # equivalent to padding=0 in PyTorch
        ))
        
        # KEY FIX: Three convs instead of four, with correct channel progression
        convs = [
            norm_f(nn.Conv(
                features=self.initial_channel * 2,
                kernel_size=(5,),
                strides=(1,),
                padding=((2, 2),)  # equivalent to padding=2 in PyTorch
            )),
            norm_f(nn.Conv(
                features=self.initial_channel * 4,
                kernel_size=(5,),
                strides=(1,),
                padding=((2, 2),)
            )),
            norm_f(nn.Conv(
                features=self.initial_channel * 4,
                kernel_size=(5,),
                strides=(1,),
                padding=((2, 2),)
            ))
        ]

        # Output layer
        conv_post = norm_f(nn.Conv(
            features=1,
            kernel_size=(3,),
            strides=(1,),
            padding=((1, 1),)  # equivalent to padding=1 in PyTorch
        ))

        # Process input
        x = pre(x)
        
        fmap = []
        for l in convs:
            x = l(x)
            x = jax.nn.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        
        x = conv_post(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        
        return x