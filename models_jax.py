#coding:utf-#coding:utf-8

import os
import os.path as osp
import copy
import math
from typing import Any, Callable, Optional, Tuple, Dict, List, Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze, unfreeze
import optax

# For PyTorch weight loading
import torch
from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet

# JAX implementations of diffusion models would be imported here
from Modules_jax.diffusion.sampler import KDiffusion, LogNormalDistribution
from Modules_jax.diffusion.modules import Transformer1d, StyleTransformer1d
from Modules_jax.diffusion.diffusion import AudioDiffusionConditional

# JAX implementations of discriminators would be imported here
from Modules_jax.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator, WavLMDiscriminator

from modules_jax import AdaIN1d

from weight_transfer_jax import transfer_jdcnet_weights, transfer_asrcnn_weights, transfer_plbert_weights

from munch import Munch
import yaml

from norm_jax import weight_norm_jax, spectral_norm_jax


class LearnedDownSample(nn.Module):
    layer_type: str
    dim_in: int
    
    def setup(self):
        if self.layer_type == 'none':
            self.conv = lambda x: x  # Identity function
        elif self.layer_type == 'timepreserve':
            self.conv = spectral_norm_jax(nn.Conv(
                features=self.dim_in,
                kernel_size=(3, 1),
                strides=(2, 1),
                padding=((1, 1), (0, 0)),
                feature_group_count=self.dim_in
            ))
        elif self.layer_type == 'half':
            self.conv = spectral_norm_jax(nn.Conv(
                features=self.dim_in,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding=((1, 1), (1, 1)),
                feature_group_count=self.dim_in
            ))
        else:
            raise RuntimeError(f'Got unexpected downsampletype {self.layer_type}, expected is [none, timepreserve, half]')
            
    def __call__(self, x, training=False):
        return self.conv(x)


class LearnedUpSample(nn.Module):
    layer_type: str
    dim_in: int
    
    def setup(self):
        if self.layer_type == 'none':
            self.conv = lambda x: x  # Identity function
        elif self.layer_type == 'timepreserve':
            self.conv = nn.ConvTranspose(
                features=self.dim_in,
                kernel_size=(3, 1),
                strides=(2, 1),
                padding=((1, 1), (0, 0)),
                feature_group_count=self.dim_in
            )
        elif self.layer_type == 'half':
            self.conv = nn.ConvTranspose(
                features=self.dim_in,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding=((1, 1), (1, 1)),
                feature_group_count=self.dim_in
            )
        else:
            raise RuntimeError(f'Got unexpected upsampletype {self.layer_type}, expected is [none, timepreserve, half]')

    def __call__(self, x, training=False):
        return self.conv(x)


class DownSample(nn.Module):
    layer_type: str
    
    def __call__(self, x, training=False):
        '''
        x : [B, 1, T, C]
        '''
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return jax.lax.reduce_window(
                x, 0.0, jax.lax.add, 
                (1, 1, 2, 1), 
                (1, 1, 2, 1), 
                'VALID'
            ) / 2.0
        elif self.layer_type == 'half':
            # Handle odd dimensions
            padded = x
            if x.shape[-2] % 2 != 0: # padding the time dimension
                padded = jnp.pad(x, ((0, 0), (0, 0), (0, 1), (0, 0)), mode='edge')
            
            return jax.lax.reduce_window(
                padded, 0.0, jax.lax.add, 
                (1, 2, 2, 1), 
                (1, 2, 2, 1), 
                'VALID'
            ) / 4.0
        else:
            raise RuntimeError(f'Got unexpected downsampletype {self.layer_type}, expected is [none, timepreserve, half]')


class UpSample(nn.Module):
    layer_type: str
    
    def __call__(self, x, training=False):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            shape = list(x.shape)
            shape[-1] *= 2
            return jax.image.resize(x, shape, method='nearest')
        elif self.layer_type == 'half':
            shape = list(x.shape)
            shape[-2] *= 2
            shape[-1] *= 2
            return jax.image.resize(x, shape, method='nearest')
        else:
            raise RuntimeError(f'Got unexpected upsampletype {self.layer_type}, expected is [none, timepreserve, half]')


class ResBlk(nn.Module):
    dim_in: int
    dim_out: int
    normalize: bool = False
    downsample: str = 'none'
    
    def setup(self):
        self.actv = lambda x: jax.nn.leaky_relu(x, 0.2)
        self.downsample_layer = DownSample(layer_type=self.downsample)
        self.downsample_res = LearnedDownSample(layer_type=self.downsample, dim_in=self.dim_in)
        self.learned_sc = self.dim_in != self.dim_out
        
        # Build weights
        self.conv1 = nn.Conv(features=self.dim_in, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        self.conv2 = nn.Conv(features=self.dim_out, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        
        if self.normalize:
            self.norm1 = nn.GroupNorm(num_groups=1, epsilon=1e-5, use_bias=True, use_scale=True)
            self.norm2 = nn.GroupNorm(num_groups=1, epsilon=1e-5, use_bias=True, use_scale=True)
            
        if self.learned_sc:
            self.conv1x1 = nn.Conv(features=self.dim_out, kernel_size=(1, 1), strides=(1, 1), padding='VALID', use_bias=False)

    def _shortcut(self, x, training):
        print(f"_shortcut input shape: {x.shape}")
        if self.learned_sc:
            x = self.conv1x1(x)
            print(f"After conv1x1 shape: {x.shape}")
        if self.downsample:
            print(f"Before downsample shape: {x.shape}")
            x = self.downsample_layer(x)
            print(f"After downsample shape: {x.shape}")
        print(f"_shortcut output shape: {x.shape}")
        return x

    def _residual(self, x, training):
        # Add debug print to monitor shapes
        print(f"_residual input shape: {x.shape}")
        
        # First stage: normalization and activation
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        
        # First convolution
        x = self.conv1(x)
        print(f"After conv1 shape: {x.shape}")
        
        # Apply the ResBlock's learned downsampling if needed
        if self.downsample != 'none':
            x = self.downsample_res(x, training)
        print(f"After downsample_res shape: {x.shape}")
        
        # Second stage: normalization, activation, and conv
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        print(f"After conv2 shape: {x.shape}")
        return x

    def __call__(self, x, training=False):
        print(f"ResBlk input shape: {x.shape}")
        x = self._shortcut(x, training) + self._residual(x, training)
        return x / math.sqrt(2)  # unit variance

class StyleEncoder(nn.Module):
    dim_in: int = 48
    style_dim: int = 48
    max_conv_dim: int = 384
    
    def setup(self):
        # Define dimensions up front
        dim_in = self.dim_in
        repeat_num = 4
        dims = [dim_in]
        for _ in range(repeat_num):
            dim_out = min(dims[-1] * 2, self.max_conv_dim)
            dims.append(dim_out)
        
        # Pre-define all blocks instead of using append
        self.conv_first = nn.Conv(
            features=self.dim_in, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            padding=((1, 1), (1, 1))
        )
        
        # Define ResBlk modules
        self.resblks = [
            ResBlk(
                dim_in=dims[i], 
                dim_out=dims[i+1], 
                downsample='half'
            )
            for i in range(repeat_num)
        ]
        
        # Final conv layer with spectral norm
        self.conv_final = spectral_norm_jax(
            nn.Conv(
                features=dims[-1], 
                kernel_size=(5, 5), 
                strides=(1, 1), 
                padding='VALID'
            )
        )
        
        # Dense layer for style output
        self.unshared = nn.Dense(features=self.style_dim)

    def __call__(self, x, training=False):
        print("input mel shape : [B, H(mels), W(frames), C] :", x.shape)
        # Initial conv
        h = self.conv_first(x)
        
        # ResBlk sequence
        for resblk in self.resblks:
            print("before resblk shape: ", h.shape)
            h = resblk(h, training)
        
        # Final processing steps
        h = jax.nn.leaky_relu(h, 0.2)
        h = self.conv_final(h)
        h = jnp.mean(h, axis=(1, 2), keepdims=True)  # AdaptiveAvgPool2d(1)
        h = jax.nn.leaky_relu(h, 0.2)
        
        # Flatten and project to style vector
        h = h.reshape(h.shape[0], -1)
        s = self.unshared(h)
        
        return s


class LinearNorm(nn.Module):
    in_dim: int
    out_dim: int
    bias: bool = True
    w_init_gain: str = 'linear'
    
    def setup(self):
        # Initialize with Xavier uniform
        if self.w_init_gain == 'linear':
            gain = 1.0
        else:
            gain = jax.nn.initializers.variance_scaling(1.0, 'fan_in', 'uniform')
            
        self.weight = self.param('weight', 
                                jax.nn.initializers.variance_scaling(gain, 'fan_in', 'uniform'),
                                (self.in_dim, self.out_dim))
        
        if self.bias:
            self.bias_param = self.param('bias', jax.nn.initializers.zeros, (self.out_dim,))
    
    def __call__(self, x, training=False):
        x = jnp.matmul(x, self.weight)
        if self.bias:
            x = x + self.bias_param
        return x

class LayerNorm(nn.Module):
    channels: int
    eps: float = 1e-5
    
    def setup(self):
        # self.gamma and self.beta are managed internally in nn.LayerNorm (JAX)
        # self.gamma = self.param('gamma', nn.initializers.ones, (self.channels,))
        # self.beta = self.param('beta', nn.initializers.zeros, (self.channels,))
        self.layernorm = nn.LayerNorm(epsilon=self.eps, use_bias=True, use_scale=True)

    
    def __call__(self, x, training=False):
        print("before ln shape: ", x.shape)
        x = jnp.transpose(x, (0, 2, 1))  # [B, emb, T] -> [B, T, emb]
        # x = nn.LayerNorm(epsilon=self.eps)(x, self.gamma, self.beta)
        x = self.layernorm(x)
        print("after ln shape: ", x.shape)
        return jnp.transpose(x, (0, 2, 1))  # [B, T, emb] -> [B, emb, T]

class TextEncoder(nn.Module):
    channels: int
    kernel_size: int
    depth: int
    n_symbols: int
    
    def setup(self):
        # Embedding layer
        self.embedding = nn.Embed(
            num_embeddings=self.n_symbols,
            features=self.channels
        )
        
        # CNN layers
        padding = (self.kernel_size - 1) // 2
        self.cnn = [
            [
                weight_norm_jax(nn.Conv(
                    features=self.channels,
                    kernel_size=(self.kernel_size,),
                    padding=((padding, padding),),
                )),
                LayerNorm(channels=self.channels),
                lambda x: jax.nn.leaky_relu(x, 0.2),
                nn.Dropout(0.2),
            ]
            for _ in range(self.depth)
        ]
        
        # LSTM layers
        # Using nn.scan with LSTMCell for bidirectional processing
        self.lstm_fw = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,  # Scan along the time dimension (axis 1 in [B, T, C])
            out_axes=1
        )(features=self.channels // 2)
        
        self.lstm_bw = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
            reverse=True  # Process the sequence in reverse for backward LSTM
        )(features=self.channels // 2)
    
    def __call__(self, x, input_lengths, m, training=False):
        # Embedding
        x = self.embedding(x)  # [B, T, emb]
        
        # Apply mask
        mask = jnp.expand_dims(m, axis=-1)  # [B, T, 1]
        x = jnp.where(mask, 0.0, x)  # [B, T, emb]
        
        # Apply CNN layers
        for layer_group in self.cnn:
            for layer in layer_group:
                if isinstance(layer, nn.Dropout):
                    x = layer(x, deterministic=not training)
                else:
                    x = layer(x)
            x = jnp.where(mask, 0.0, x)
        
        # Get dropout RNG key
        dropout_key = self.make_rng('dropout') if training else jax.random.PRNGKey(0)
        lstm_key1, lstm_key2 = jax.random.split(dropout_key)
        
        # Initialize LSTM states
        batch_size = x.shape[0]
        forward_carry = self.lstm_fw.initialize_carry(
            lstm_key1, 
            (batch_size, x.shape[-1])
        )
        backward_carry = self.lstm_bw.initialize_carry(
            lstm_key2, 
            (batch_size, x.shape[-1])
        )
        
        # Run bidirectional LSTM
        # The scan automatically processes along time dimension
        _, outputs_fw = self.lstm_fw(forward_carry, x)
        _, outputs_bw = self.lstm_bw(backward_carry, x)
        
        # Concatenate forward and backward outputs
        x = jnp.concatenate([outputs_fw, outputs_bw], axis=-1)  # [B, T, channels]
        
        # Transpose to match the expected output format
        x = jnp.transpose(x, (0, 2, 1))  # [B, channels, T]
        
        # Create padding to match mask shape if needed
        if x.shape[-1] != m.shape[-1]:
            x_pad = jnp.zeros((x.shape[0], x.shape[1], m.shape[-1]), dtype=x.dtype)
            x_pad = x_pad.at[:, :, :x.shape[-1]].set(x)
            x = x_pad
        
        # Apply final mask
        mask_expanded = jnp.expand_dims(m, axis=1)  # [B, 1, T]
        x = jnp.where(mask_expanded, 0.0, x)
        
        return x


class UpSample1d(nn.Module):
    layer_type: str
    
    def __call__(self, x, training=False):
        if self.layer_type == 'none':
            return x
        else:
            shape = list(x.shape)
            shape[2] *= 2
            return jax.image.resize(x, shape, method='nearest')


class AdainResBlk1d(nn.Module):
    dim_in: int
    dim_out: int
    style_dim: int = 64
    upsample: str = 'none'
    dropout_p: float = 0.0
    
    def setup(self):
        self.actv = lambda x: jax.nn.leaky_relu(x, 0.2)
        self.upsample_layer = UpSample1d(layer_type=self.upsample)
        self.learned_sc = self.dim_in != self.dim_out
        
        # Build weights
        self.conv1 = weight_norm_jax(nn.Conv(features=self.dim_out, kernel_size=(3,), strides=(1,), padding=((1, 1),)))
        self.conv2 = weight_norm_jax(nn.Conv(features=self.dim_out, kernel_size=(3,), strides=(1,), padding=((1, 1),)))
        self.norm1 = AdaIN1d(style_dim=self.style_dim, num_features=self.dim_in)
        self.norm2 = AdaIN1d(style_dim=self.style_dim, num_features=self.dim_out)

        # Define dropout layer in setup
        self.dropout1 = nn.Dropout(rate=self.dropout_p)
        self.dropout2 = nn.Dropout(rate=self.dropout_p)
        
        if self.learned_sc:
            self.conv1x1 = weight_norm_jax(nn.Conv(features=self.dim_out, kernel_size=(1,), strides=(1,), padding='VALID', use_bias=False))
        
        # Set up pooling
        if self.upsample == 'none':
            self.pool = lambda x: x  # Identity function
        else:
            self.pool = weight_norm_jax(nn.ConvTranspose(
                features=self.dim_in,
                kernel_size=(3,),
                strides=(2,),
                padding=((1, 1),),
                feature_group_count=self.dim_in)
            )
        
    def _shortcut(self, x, training):
        x = self.upsample_layer(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x
    
    def _residual(self, x, s, training):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout1(x, deterministic=not training))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout2(x, deterministic=not training))
        return x
    
    def __call__(self, x, s, training=False):
        out = self._residual(x, s, training)
        out = (out + self._shortcut(x, training)) / math.sqrt(2)
        return out


class AdaLayerNorm(nn.Module):
    style_dim: int
    channels: int
    eps: float = 1e-5
    
    def setup(self):
        self.fc = nn.Dense(features=self.channels * 2)
    
    def __call__(self, x, s, training=False):
        # Handle dimension shuffling for Flax
        print("before adaln shape: ", x.shape)
        
        h = self.fc(s)
        h = h.reshape(h.shape[0], 1, h.shape[1])
        gamma, beta = jnp.split(h, 2, axis=2)
        
        # Layer norm on last dimension
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        
        x = (1 + gamma) * x + beta
        
        print("after adaln shape: ", x.shape)
        return x

class DurationProsodyPredictor(nn.Module):
    style_dim: int
    d_hid: int
    nlayers: int
    max_dur: int = 50
    dropout: float = 0.1
    
    def setup(self):
        self.text_encoder = DurationEncoder(
            sty_dim=self.style_dim,
            d_model=self.d_hid,
            nlayers=self.nlayers,
            dropout=self.dropout
        )
        
        # LSTMs in Flax are different - need to use nn.scan
        self.lstm_fw = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False, "dropout": True},
            in_axes=1,
            out_axes=1,
        )(features=self.d_hid // 2)
        
        self.lstm_bw = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False, "dropout": True},
            in_axes=1,
            out_axes=1,
            reverse=True
        )(features=self.d_hid // 2)
        
        self.duration_proj = LinearNorm(
            in_dim=self.d_hid,
            out_dim=self.max_dur
        )
        
        # Replace shared_cell with bidirectional LSTM cells (like PyTorch's self.shared)
        self.shared_lstm_fw = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False, "dropout": True},
            in_axes=1,
            out_axes=1,
        )(features=self.d_hid // 2)
        
        self.shared_lstm_bw = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False, "dropout": True},
            in_axes=1,
            out_axes=1,
            reverse=True
        )(features=self.d_hid // 2)
        
        # F0 blocks
        self.F0 = [
            AdainResBlk1d(dim_in=self.d_hid, dim_out=self.d_hid, style_dim=self.style_dim, dropout_p=self.dropout),
            AdainResBlk1d(dim_in=self.d_hid, dim_out=self.d_hid // 2, style_dim=self.style_dim, upsample='linear', dropout_p=self.dropout),
            AdainResBlk1d(dim_in=self.d_hid // 2, dim_out=self.d_hid // 2, style_dim=self.style_dim, dropout_p=self.dropout)
        ]
        
        # N blocks
        self.N = [
            AdainResBlk1d(dim_in=self.d_hid, dim_out=self.d_hid, style_dim=self.style_dim, dropout_p=self.dropout),
            AdainResBlk1d(dim_in=self.d_hid, dim_out=self.d_hid // 2, style_dim=self.style_dim, upsample='linear', dropout_p=self.dropout),
            AdainResBlk1d(dim_in=self.d_hid // 2, dim_out=self.d_hid // 2, style_dim=self.style_dim, dropout_p=self.dropout)
        ]
        
        # Projection layers
        self.F0_proj = nn.Conv(features=1, kernel_size=(1,), strides=(1,), padding='VALID')
        self.N_proj = nn.Conv(features=1, kernel_size=(1,), strides=(1,), padding='VALID')

        # Define dropout layer in setup
        self.dropout_layer = nn.Dropout(rate=0.5)

    
    def __call__(self, texts, style, text_lengths, alignment, m, training=False):
        """
        texts : [B, emb1, T], style : [B, emb2], alignment : [B, T, L]
        """
        d = self.text_encoder(texts, style, text_lengths, m, training)
        
        # Transpose for LSTM processing
        d = jnp.transpose(d, (0, 2, 1))  # [B, emb, T] -> [B, T, emb]
        
        batch_size = d.shape[0]
        
        # Apply mask
        mask = jnp.expand_dims(m, axis=-1)  # [B, T, 1]
        
        # Initialize LSTM states
        rng = self.make_rng('dropout') if training else jax.random.PRNGKey(0)
        key_fw, key_bw = jax.random.split(rng)
        
        # Initialize LSTM carry states
        fw_state = self.lstm_fw.initialize_carry(key_fw, (batch_size, d.shape[-1]))
        bw_state = self.lstm_bw.initialize_carry(key_bw, (batch_size, d.shape[-1]))
        
        # Run forward and backward LSTMs
        _, fw_out = self.lstm_fw(fw_state, d)
        _, bw_out = self.lstm_bw(bw_state, d)
        
        # Concatenate outputs
        x = jnp.concatenate([fw_out, bw_out], axis=-1)  # [B, T, d_hid]
        
        # Apply mask after LSTM
        x = jnp.where(mask, 0.0, x)
        
        # Apply dropout
        x = self.dropout_layer(x, deterministic=not training)
        
        # Project to duration logits
        duration = self.duration_proj(x)
        
        # Calculate encoder output with alignment
        # For the matrix multiplication, transpose d back to [B, emb, T]
        d_transposed = jnp.transpose(d, (0, 2, 1))
        en = jnp.matmul(d_transposed, alignment)
        
        # return jnp.squeeze(duration, axis=-1), en # [B, T, max_dur], [B, emb, L]
        return duration, en # [B, T, max_dur], [B, emb, L]
    
    def F0Ntrain(self, x, s, training=False):
        """
        Generate F0 and energy (N) contours using learned style
        
        Args:
            x: Input encoder hidden features [B, C, L]
            s: Style embedding [B, style_dim]
            training: Whether in training mode
        
        Returns:
            F0: F0 contour [B, L]
            N: Energy contour [B, L]
        """
        # Process input through LSTM
        batch_size = x.shape[0]
        seq_len = x.shape[2]
        
        # Transpose for LSTM processing
        x = jnp.transpose(x, (0, 2, 1))  # [B, C, L] -> [B, L, C]
        
        # Expand style to match the sequence length
        s_expanded = jnp.tile(s[:, None, :], (1, seq_len, 1))  # [B, L, S]
        
        # Concatenate x and style as in the PyTorch version
        x_with_style = jnp.concatenate([x, s_expanded], axis=-1)  # [B, L, C+S]
        
        # Initialize LSTM states
        rng = self.make_rng('dropout') if training else jax.random.PRNGKey(0)
        key_fw, key_bw = jax.random.split(rng)
        
        fw_state = self.shared_lstm_fw.initialize_carry(key_fw, (batch_size, x_with_style.shape[-1]))
        bw_state = self.shared_lstm_bw.initialize_carry(key_bw, (batch_size, x_with_style.shape[-1]))
        
        # Run bidirectional LSTM
        _, fw_out = self.shared_lstm_fw(fw_state, x_with_style)
        _, bw_out = self.shared_lstm_bw(bw_state, x_with_style)
        
        # Concatenate forward and backward outputs
        lstm_out = jnp.concatenate([fw_out, bw_out], axis=-1)  # [B, L, d_hid]
        
        # Apply dropout
        lstm_out = self.dropout_layer(lstm_out, deterministic=not training)
        
        # Transpose back to channel-first format
        lstm_out = jnp.transpose(lstm_out, (0, 2, 1))  # [B, L, C] -> [B, C, L]
        
        # Apply F0 blocks
        F0 = lstm_out
        for block in self.F0:
            F0 = block(F0, s, training)
        F0 = self.F0_proj(F0)
        
        # Apply N blocks
        N = lstm_out
        for block in self.N:
            N = block(N, s, training)
        N = self.N_proj(N)
        
        return jnp.squeeze(F0, axis=1), jnp.squeeze(N, axis=1) 

class DurationEncoder(nn.Module):
    """JAX/Flax implementation of DurationEncoder"""
    sty_dim: int
    d_model: int
    nlayers: int
    dropout: float = 0.1
    
    @nn.compact
    def __call__(self, x, style, text_lengths, m, training=False):
        """
        Args:
            x: Text encoder output [B, D, L] (batch, features, length)
            style: Style vector [B, S]
            text_lengths: Length of each sequence in batch [B]
            m: Mask [B, L]
            training: Whether in training mode
        """
        # Transpose to JAX preferred format [B, L, D]
        x = jnp.transpose(x, (0, 2, 1))  # [B, D, L] -> [B, L, D]
        
        # Create expanded style tensor and concatenate
        batch_size, seq_len = x.shape[0], x.shape[1]
        s_expanded = jnp.tile(style[:, None, :], (1, seq_len, 1))  # [B, L, S]
        x = jnp.concatenate([x, s_expanded], axis=-1)  # [B, L, D+S]
        
        # Create mask in right format [B, L, 1]
        mask = jnp.expand_dims(m, axis=-1)
        x = jnp.where(mask, 0.0, x)

        
        # Process through alternating LSTM and AdaLayerNorm layers
        for i in range(self.nlayers):
            # Define LSTM layers inside the loop using nn.compact scope
            lstm_fw = nn.scan(
                nn.LSTMCell,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=1,
                out_axes=1,
            )(features=self.d_model // 2)
            
            lstm_bw = nn.scan(
                nn.LSTMCell,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=1,
                out_axes=1,
                reverse=True,
            )(features=self.d_model // 2)
            
            # # Use only the features part without style for LSTM
            # x_without_style = x[..., :-self.sty_dim]
            
            # Initialize LSTM states
            rng = self.make_rng('dropout') if training else jax.random.PRNGKey(0)
            key_fw, key_bw = jax.random.split(rng)
            
            fw_state = lstm_fw.initialize_carry(
                key_fw,
                (batch_size, x.shape[-1])
            )
            bw_state = lstm_bw.initialize_carry(
                key_bw,
                (batch_size, x.shape[-1])
            )
            
            # Run bidirectional LSTM
            _, fw_out = lstm_fw(fw_state, x)
            _, bw_out = lstm_bw(bw_state, x)
            
            # Concatenate forward and backward outputs
            lstm_out = jnp.concatenate([fw_out, bw_out], axis=-1)  # [B, L, D]
            

            # Apply dropout
            dropout_layer = nn.Dropout(rate=self.dropout)
            lstm_out = dropout_layer(lstm_out, deterministic=not training)
            
            # # Replace features part in x, keeping style part
            # x = jnp.concatenate([lstm_out, s_expanded], axis=-1)
            x = lstm_out
            
            # Apply mask
            x = jnp.where(mask, 0.0, x)
            
            # Define AdaLayerNorm inside the loop
            ada_norm = AdaLayerNorm(
                channels=self.d_model, 
                style_dim=self.sty_dim,
                name=f'norm_{i}'
            )
            
            # Process through AdaLayerNorm (features only)
            features = ada_norm(x, style)
            
            # # Recombine with style
            # x = jnp.concatenate([features, s_expanded], axis=-1)
            
            # Apply mask
            x = jnp.where(mask, 0.0, x)
        
        # Return in format expected by rest of model [B, D, L]
        return jnp.transpose(x, (0, 2, 1))

# Load functions for transferring PyTorch weights to JAX models

def load_F0_models_jax(path, rng):
    """Load F0 model weights from PyTorch checkpoint to JAX"""
    # First load the PyTorch model
    F0_model_pt = JDCNet(num_class=1, seq_len=192)
    params = torch.load(path, map_location='cpu')['net']
    F0_model_pt.load_state_dict(params)
    
    # Initialize JAX model
    from Utils_jax.JDC.model import JDCNetJax
    F0_model_jax = JDCNetJax(num_class=1, seq_len=192)
    
    # Initialize with dummy input
    dummy_input = jnp.ones((1, 1, 192, 80))
    rng, lstm_rng = jax.random.split(rng)
    variables = F0_model_jax.init({'params': rng, 'lstm': lstm_rng}, dummy_input)
    
    # Transfer weights
    jax_params = transfer_jdcnet_weights(F0_model_pt, variables)
    
    return F0_model_jax, jax_params


def load_ASR_models_jax(ASR_MODEL_PATH, ASR_MODEL_CONFIG, rng):
    """Load ASR model weights from PyTorch checkpoint to JAX"""
    # Load config
    def _load_config(path):
        with open(path) as f:
            config = yaml.safe_load(f)
        model_config = config['model_params']
        return model_config
    
    # Load PyTorch model
    asr_model_config = _load_config(ASR_MODEL_CONFIG)
    asr_model_pt = ASRCNN(**asr_model_config)
    params = torch.load(ASR_MODEL_PATH, map_location='cpu')['model']
    asr_model_pt.load_state_dict(params)
    
    # Initialize JAX model
    from Utils_jax.ASR.models import ASRCNNJax
    asr_model_jax = ASRCNNJax(**asr_model_config)
    
    # Initialize with dummy input
    dummy_mels = jnp.ones((1, 80, 100))
    dummy_mask = jnp.ones((1, 100 // 2))
    dummy_text = jnp.ones((1, 20), dtype=jnp.int32)
    variables = asr_model_jax.init(rng, dummy_mels, dummy_mask, dummy_text, rng)
    
    # Transfer weights
    jax_params = transfer_asrcnn_weights(asr_model_pt, variables)
    
    return asr_model_jax, jax_params


def load_plbert_jax(BERT_PATH, rng):
    """Load PLBERT model into JAX"""
    # We'll import a pretrained JAX BERT model
    try:
        from transformers import FlaxBertModel, BertConfig
        
        # Initialize from pretrained or config
        if os.path.exists(BERT_PATH):
            bert_model = FlaxBertModel.from_pretrained(BERT_PATH)
            bert_params = bert_model.params
        else:
            config = BertConfig.from_pretrained("bert-base-uncased")
            bert_model = FlaxBertModel(config)
            variables = bert_model.init(rng, jnp.ones((1, 10), dtype=jnp.int32))
            bert_params = variables["params"]
            
    except ImportError:
        # Simplified fallback if transformers not available
        print("Warning: transformers library not found. Using placeholder BERT model")
        from modules_jax.bert import SimplifiedBertJax
        config = {"hidden_size": 768, "num_attention_heads": 12, "intermediate_size": 3072}
        bert_model = SimplifiedBertJax(**config)
        variables = bert_model.init(rng, jnp.ones((1, 10), dtype=jnp.int32))
        bert_params = variables["params"]
    
    return bert_model, bert_params

def build_model_jax(args, text_aligner, text_aligner_params, pitch_extractor, pitch_extractor_params, bert, bert_params, rng):
    """Build JAX model components including diffusion models"""
    # Verify decoder type
    assert args.decoder.type in ['istftnet', 'hifigan'], 'Decoder type unknown'
    
    # Split RNG for multiple initializations
    rng_keys = jax.random.split(rng, 9)
    key_text, key_dur, key_acoustic, key_prosodic, key_decoder, key_bert, key_diff, key_mpd, key_msd = rng_keys
    
    # Create submodels with appropriate configurations
    if args.decoder.type == "istftnet":
        from Modules_jax.istftnet import DecoderJax
        decoder = DecoderJax(
            dim_in=args.hidden_dim,
            style_dim=args.style_dim,
            dim_out=args.n_mels,
            resblock_kernel_sizes=args.decoder.resblock_kernel_sizes,
            upsample_rates=args.decoder.upsample_rates,
            upsample_initial_channel=args.decoder.upsample_initial_channel,
            resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
            upsample_kernel_sizes=args.decoder.upsample_kernel_sizes,
            gen_istft_n_fft=args.decoder.gen_istft_n_fft,
            gen_istft_hop_size=args.decoder.gen_istft_hop_size
        )
    else:
        from Modules_jax.hifigan import DecoderJax
        decoder = DecoderJax(
            dim_in=args.hidden_dim,
            style_dim=args.style_dim,
            dim_out=args.n_mels,
            resblock_kernel_sizes=args.decoder.resblock_kernel_sizes,
            upsample_rates=args.decoder.upsample_rates,
            upsample_initial_channel=args.decoder.upsample_initial_channel,
            resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
            upsample_kernel_sizes=args.decoder.upsample_kernel_sizes
        )
    
    # Initialize text encoder
    text_encoder = TextEncoder(
        channels=args.hidden_dim,
        kernel_size=5,
        depth=args.n_layer,
        n_symbols=args.n_token
    )
    
    # Initialize duration predictor
    duration_prosody_predictor = DurationProsodyPredictor(
        style_dim=args.style_dim,
        d_hid=args.hidden_dim,
        nlayers=args.n_layer,
        max_dur=args.max_dur,
        dropout=args.dropout
    )
    
    # Initialize style encoders
    acoustic_style_encoder = StyleEncoder(
        dim_in=args.dim_in, 
        style_dim=args.style_dim, 
        max_conv_dim=args.hidden_dim
    )
    
    prosodic_style_encoder = StyleEncoder(
        dim_in=args.dim_in, 
        style_dim=args.style_dim, 
        max_conv_dim=args.hidden_dim
    )
    
    # Initialize BERT encoder
    bert_encoder = nn.Dense(features=args.hidden_dim)
    
    # Initialize diffusion models
    if args.multispeaker:
        # For multispeaker
        transformer = StyleTransformer1d(
            channels=args.style_dim * 2,
            context_embedding_features=bert.config.hidden_size,
            context_features=args.style_dim * 2,
            **args.diffusion.transformer
        )
    else:
        # For single speaker
        transformer = Transformer1d(
            channels=args.style_dim * 2,
            context_embedding_features=bert.config.hidden_size,
            **args.diffusion.transformer
        )
    
    # Initialize diffusion
    diffusion = AudioDiffusionConditional(
        in_channels=1,
        embedding_max_length=bert.config.max_position_embeddings,
        embedding_features=bert.config.hidden_size,
        embedding_mask_proba=args.diffusion.embedding_mask_proba,
        channels=args.style_dim * 2,
        context_features=args.style_dim * 2,
    )
    
    # Set up diffusion sampling
    diffusion_sampler = KDiffusion(
        net=transformer,
        sigma_distribution=LogNormalDistribution(
            mean=args.diffusion.dist.mean,
            std=args.diffusion.dist.std
        ),
        sigma_data=args.diffusion.dist.sigma_data,
        dynamic_threshold=0.0
    )
    
    # Hook up the diffusion components
    diffusion.diffusion = diffusion_sampler
    diffusion.unet = transformer
    
    # Initialize discriminators
    mpd = MultiPeriodDiscriminator()
    msd = MultiResSpecDiscriminator()
    wd = WavLMDiscriminator(
        slm_hidden=args.slm.hidden,
        slm_layers=args.slm.nlayers,
        initial_channel=args.slm.initial_channel
    )
    
    # Combine all models
    models = Munch(
        bert=bert,
        bert_encoder=bert_encoder,
        
        duration_prosody_predictor=duration_prosody_predictor,
        decoder=decoder,
        text_encoder=text_encoder,
        
        prosodic_style_encoder=prosodic_style_encoder,
        acoustic_style_encoder=acoustic_style_encoder,
        
        text_aligner=text_aligner,
        pitch_extractor=pitch_extractor,
        
        diffusion=diffusion,
        transformer=transformer,
        
        mpd=mpd,
        msd=msd,
        wd=wd
    )
    
    # Create dummy inputs for initializing parameters
    dummy_text = jnp.ones((1, 50), dtype=jnp.int32)
    dummy_lengths = jnp.ones((1,), dtype=jnp.int32) * 50
    dummy_mask = jnp.ones((1, 50), dtype=jnp.bool_)
    dummy_mel = jnp.ones((1, 80, 100)) # [B, mel_bin, frames]
    dummy_style = jnp.ones((1, args.style_dim))
    dummy_wav = jnp.ones((1, 24000, 1))
    dummy_wav_hat = jnp.ones((1, 24000, 1))
    
    # Initialize parameters for each component
    bert_encoder_params = bert_encoder.init(key_bert, jnp.ones((1, bert.config.hidden_size)))['params']
    
    text_encoder_params = text_encoder.init(
        key_text, dummy_text, dummy_lengths, dummy_mask, training=False
    )['params']
    
    duration_prosody_predictor_params = duration_prosody_predictor.init(
        {'params': key_dur, 'dropout': jax.random.PRNGKey(0)},
        jnp.ones((1, args.hidden_dim, 50)), dummy_style, dummy_lengths,
        jnp.ones((1, 50, 100)), dummy_mask, training=False
    )['params']
    
    # unsqueeze dummy_mel
    acoustic_style_encoder_params = acoustic_style_encoder.init(
        key_acoustic, jnp.expand_dims(dummy_mel, axis=-1), training=False
    )['params']
    
    prosodic_style_encoder_params = prosodic_style_encoder.init(
        key_prosodic, jnp.expand_dims(dummy_mel, axis=-1), training=False
    )['params']
    
    decoder_params = decoder.init(
        key_decoder,
        jnp.ones((1, 100 // 2, args.hidden_dim)), # hidden (asr)
        jnp.ones((1, 100)),  # f0
        jnp.ones((1, 100)),  # norm
        dummy_style                # style
    )['params']
    
    # transformer_params = transformer.init(
    #     key_diff,
    #     jnp.ones((1, args.style_dim * 2, 100)),
    #     jnp.ones((1, 100, bert.config.hidden_size)),
    #     jnp.ones((1, args.style_dim * 2))
    # )['params']
    
    # diffusion_params = diffusion.init(
    #     key_diff,
    #     jnp.ones((1, 1, 100)),
    #     jnp.ones((1, 100, bert.config.hidden_size)),
    #     jnp.ones((1, args.style_dim * 2))
    # )['params']
    print("initialise mpd") 
    mpd_params = mpd.init(key_mpd, dummy_wav, dummy_wav_hat)['params']
    print("initialise msd")
    msd_params = msd.init(key_msd, dummy_wav, dummy_wav_hat)['params']
    print("initialise wd")
    wd_params = wd.init(key_msd, dummy_wav)['params']
    
    # Combine all parameters
    variables = {
        'params': {
            'bert': bert_params,
            'bert_encoder': bert_encoder_params,
            'text_encoder': text_encoder_params,
            'duration_prosody_predictor': duration_prosody_predictor_params,
            'acoustic_style_encoder': acoustic_style_encoder_params,
            'prosodic_style_encoder': prosodic_style_encoder_params,
            'decoder': decoder_params,
            'text_aligner': text_aligner_params,
            'pitch_extractor': pitch_extractor_params,
            # 'transformer': transformer_params,
            # 'diffusion': diffusion_params,
            'mpd': mpd_params,
            'msd': msd_params,
            'wd': wd_params
        }
    }
    
    return models, variables

# Add after imports section

class ModelWrapper:
    """Wrapper class to provide apply method for the model components"""
    def __init__(self, models):
        self.models = models
    
    def apply(self, params, *args, method=None, **kwargs):
        """Dispatch to the appropriate model component's apply method"""
        if method is None:
            raise ValueError("Method must be provided for ModelWrapper.apply")
        
        # Extract the module and method from the lambda function
        # The lambda has the form: lambda module, *args, **kwargs: module.method_name(*args, **kwargs)
        module_name = method.__code__.co_consts[0].split('.')[0]
        method_name = method.__code__.co_consts[0].split('.')[1]
        
        # Apply the method to the appropriate model component
        component = getattr(self.models, module_name)
        return getattr(component, method_name)(*args, **kwargs)


def load_checkpoint_jax(model, optimizer, path, load_only_params=True, ignore_modules=[]):
    """Load checkpoint from PyTorch to JAX models"""
    # Load PyTorch checkpoint
    state = torch.load(path, map_location='cpu')
    pt_params = state['net']
    
    # Create updated model dictionary
    model_updated = {}
    
    # Transfer weights for each component
    for key in model:
        if key in pt_params and key not in ignore_modules:
            print(f'{key} loaded')
            
            # Special handling for pre-trained models with separate parameters
            if key == 'text_aligner':
                model_updated[key] = model[key]
                model_updated[f'{key}_params'] = transfer_asrcnn_weights(pt_params[key], model[f'{key}_params'])
            
            elif key == 'pitch_extractor':
                model_updated[key] = model[key]
                model_updated[f'{key}_params'] = transfer_jdcnet_weights(pt_params[key], model[f'{key}_params'])
            
            elif key == 'bert':
                model_updated[key] = model[key]
                model_updated[f'{key}_params'] = transfer_plbert_weights(pt_params[key], model[f'{key}_params'])
            
            # Handle other components with specific transfer functions as needed
            else:
                # For now, we just keep the original model component
                # In a complete implementation, each component would have its own transfer function
                model_updated[key] = model[key]
        else:
            # Keep components that don't need transfer
            model_updated[key] = model[key]
            if f'{key}_params' in model:
                model_updated[f'{key}_params'] = model[f'{key}_params']
    
    # Retrieve training state from checkpoint if needed
    if not load_only_params:
        epoch = state.get("epoch", 0)
        iters = state.get("iters", 0)
        
        # TODO: Transfer optimizer state if needed
        # This is complex and depends on the specific optimizer being used
        
    else:
        epoch = 0
        iters = 0
    
    return model_updated, optimizer, epoch, iters