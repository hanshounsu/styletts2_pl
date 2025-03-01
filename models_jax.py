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
# from modules_jax.diffusion.sampler import KDiffusion, LogNormalDistribution
# from modules_jax.diffusion.modules import Transformer1d, StyleTransformer1d
# from modules_jax.diffusion.diffusion import AudioDiffusionConditional

# JAX implementations of discriminators would be imported here
# from modules_jax.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator, WavLMDiscriminator

from munch import Munch
import yaml

# JAX utility functions for spectral_norm and weight_norm
def weight_norm_jax(x, axis=0):
    """JAX equivalent of PyTorch's weight_norm"""
    g = jnp.sqrt(jnp.sum(x**2, axis=axis, keepdims=True))
    return x * g

def spectral_norm_jax(w, n_iterations=1):
    """JAX equivalent of PyTorch's spectral_norm (simplified)"""
    w_shape = w.shape
    w = w.reshape(-1, w_shape[-1])
    
    u = jnp.ones((1, w_shape[-1]))
    
    # Power iteration
    for _ in range(n_iterations):
        v = jnp.matmul(u, w.T)
        v = v / jnp.linalg.norm(v, ord=2)
        u = jnp.matmul(v, w) 
        u = u / jnp.linalg.norm(u, ord=2)
        
    sigma = jnp.matmul(jnp.matmul(v, w), u.T)
    return w.reshape(w_shape) / sigma


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
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return jax.lax.reduce_window(
                x, 0.0, jax.lax.add, 
                (1, 2, 1, 1), 
                (1, 2, 1, 1), 
                'VALID'
            ) / 2.0
        elif self.layer_type == 'half':
            # Handle odd dimensions
            padded = x
            if x.shape[-1] % 2 != 0:
                padded = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 1)), mode='constant')
            
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
            shape[1] *= 2
            return jax.image.resize(x, shape, method='nearest')
        elif self.layer_type == 'half':
            shape = list(x.shape)
            shape[1] *= 2
            shape[2] *= 2
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
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample_layer(x)
        return x

    def _residual(self, x, training):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def __call__(self, x, training=False):
        x = self._shortcut(x, training) + self._residual(x, training)
        return x / math.sqrt(2)  # unit variance


class StyleEncoder(nn.Module):
    dim_in: int = 48
    style_dim: int = 48
    max_conv_dim: int = 384
    
    def setup(self):
        blocks = []
        blocks.append(nn.Conv(features=self.dim_in, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1))))
        
        dim_in = self.dim_in
        repeat_num = 4
        
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, self.max_conv_dim)
            blocks.append(ResBlk(dim_in=dim_in, dim_out=dim_out, downsample='half'))
            dim_in = dim_out
        
        blocks.append(lambda x: jax.nn.leaky_relu(x, 0.2))
        blocks.append(spectral_norm_jax(nn.Conv(features=dim_out, kernel_size=(5, 5), strides=(1, 1), padding='VALID')))
        blocks.append(lambda x: jnp.mean(x, axis=(1, 2), keepdims=True))  # AdaptiveAvgPool2d(1)
        blocks.append(lambda x: jax.nn.leaky_relu(x, 0.2))
        
        self.shared = blocks
        self.unshared = nn.Dense(features=self.style_dim)

    def __call__(self, x, training=False):
        h = x
        for layer in self.shared:
            h = layer(h)
        h = h.reshape(h.shape[0], -1)  # Flatten
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
            self.bias = self.param('bias', jax.nn.initializers.zeros, (self.out_dim,))
    
    def __call__(self, x, training=False):
        x = jnp.matmul(x, self.weight)
        if self.bias:
            x = x + self.bias
        return x

class LayerNorm(nn.Module):
    channels: int
    eps: float = 1e-5
    
    def setup(self):
        self.gamma = self.param('gamma', nn.initializers.ones, (self.channels,))
        self.beta = self.param('beta', nn.initializers.zeros, (self.channels,))
    
    def __call__(self, x, training=False):
        x = jnp.transpose(x, (0, 2, 1))  # [B, emb, T] -> [B, T, emb]
        x = nn.LayerNorm(epsilon=self.eps)(x, self.gamma, self.beta)
        return jnp.transpose(x, (0, 2, 1))  # [B, T, emb] -> [B, emb, T]

class TextEncoder(nn.Module):
    channels: int
    kernel_size: int
    depth: int
    n_symbols: int
    
    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.n_symbols,
            features=self.channels
        )
        
        padding = (self.kernel_size - 1) // 2
        self.cnn = []
        
        for _ in range(self.depth):
            self.cnn.append([
                nn.Conv(
                    features=self.channels,
                    kernel_size=(self.kernel_size,),
                    padding=((padding, padding),),
                ),
                LayerNorm(channels=self.channels),
                lambda x: jax.nn.leaky_relu(x, 0.2),
                nn.Dropout(0.2),
            ])
        
        self.lstm = nn.scan(nn.LSTMCell, # TODO : what is this?
                          variable_broadcast="params",
                          split_rngs={"params": False})
        self.lstm_fw = self.lstm(self.channels // 2)
        self.lstm_bw = self.lstm(self.channels // 2)
    
    def __call__(self, x, input_lengths, m, training=False):
        x = self.embedding(x)  # [B, T, emb]
        x = jnp.transpose(x, (0, 2, 1))  # [B, emb, T]
        
        # Apply mask (similar to masked_fill in PyTorch)
        mask = jnp.expand_dims(m, axis=1)
        x = jnp.where(mask, 0.0, x)
        
        # Apply CNN layers
        for layer_group in self.cnn:
            for layer in layer_group:
                if isinstance(layer, nn.Dropout):
                    x = layer(x, deterministic=not training)
                else:
                    x = layer(x)
            x = jnp.where(mask, 0.0, x)
        
        # Prepare for LSTM
        x = jnp.transpose(x, (0, 2, 1))  # [B, T, channels]
        
        # Create a padded sequence (JAX doesn't have direct equivalent to pad_sequence)
        # Here we'll use the mask to handle variable lengths
        
        # Bidirectional LSTM (simplification - a proper implementation would handle sequences)
        # Forward pass
        def scan_lstm_fw(carry, x):
            return self.lstm_fw(carry, x)
        
        # Backward pass
        def scan_lstm_bw(carry, x):
            return self.lstm_bw(carry, x)
        
        # Initialize states
        batch_size = x.shape[0]
        init_state_fw = self.lstm_fw.initialize_carry(jnp.zeros((batch_size,)), self.channels // 2)
        init_state_bw = self.lstm_bw.initialize_carry(jnp.zeros((batch_size,)), self.channels // 2)
        
        # Run LSTM (simplified - doesn't handle proper masking)
        _, outputs_fw = jax.lax.scan(scan_lstm_fw, init_state_fw, x)
        _, outputs_bw = jax.lax.scan(scan_lstm_bw, init_state_bw, jnp.flip(x, axis=1))
        outputs_bw = jnp.flip(outputs_bw, axis=1)
        
        # Concatenate directions
        x = jnp.concatenate([outputs_fw, outputs_bw], axis=-1)
        
        # Transpose back and create padded output
        x = jnp.transpose(x, (0, 2, 1))  # [B, channels, T]
        
        # Create padding to match mask shape
        x_pad = jnp.zeros((x.shape[0], x.shape[1], m.shape[-1]), dtype=x.dtype)
        
        # Place x into the padded tensor (simplified - would need dynamic slicing)
        x_pad = x_pad.at[:, :, :x.shape[-1]].set(x)
        
        # Apply mask
        x = jnp.where(mask, 0.0, x_pad)
        
        return x

class AdaIN1d(nn.Module):
    style_dim: int
    num_features: int
    
    def setup(self):
        self.norm = nn.GroupNorm(num_groups=1, epsilon=1e-5, use_bias=False, use_scale=False)
        self.fc = nn.Dense(features=self.num_features * 2)
    
    def __call__(self, x, s, training=False):
        h = self.fc(s)
        h = h.reshape(h.shape[0], h.shape[1], 1)
        gamma, beta = jnp.split(h, 2, axis=1)
        return (1 + gamma) * self.norm(x) + beta


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
        self.conv1 = nn.Conv(features=self.dim_out, kernel_size=(3,), strides=(1,), padding=((1, 1),))
        self.conv2 = nn.Conv(features=self.dim_out, kernel_size=(3,), strides=(1,), padding=((1, 1),))
        self.norm1 = AdaIN1d(style_dim=self.style_dim, num_features=self.dim_in)
        self.norm2 = AdaIN1d(style_dim=self.style_dim, num_features=self.dim_out)
        
        if self.learned_sc:
            self.conv1x1 = nn.Conv(features=self.dim_out, kernel_size=(1,), strides=(1,), padding='VALID', use_bias=False)
        
        # Set up pooling
        if self.upsample == 'none':
            self.pool = lambda x: x  # Identity function
        else:
            self.pool = nn.ConvTranspose(
                features=self.dim_in,
                kernel_size=(3,),
                strides=(2,),
                padding=((1, 1),),
                feature_group_count=self.dim_in
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
        x = self.conv1(nn.Dropout(rate=self.dropout_p, deterministic=not training)(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(nn.Dropout(rate=self.dropout_p, deterministic=not training)(x))
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
        x = jnp.transpose(x, (0, 2, 1))  # [B, D, L] -> [B, L, D]
        
        h = self.fc(s)
        h = h.reshape(h.shape[0], 1, h.shape[1])
        gamma, beta = jnp.split(h, 2, axis=2)
        
        # Layer norm on last dimension
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        
        x = (1 + gamma) * x + beta
        
        # Restore original dimensions
        x = jnp.transpose(x, (0, 2, 1))  # [B, L, D] -> [B, D, L]
        return x


# Load functions for transferring PyTorch weights to JAX models

def load_F0_models_jax(path, rng):
    """Load F0 model weights from PyTorch checkpoint to JAX"""
    # First load the PyTorch model
    F0_model_pt = JDCNet(num_class=1, seq_len=192)
    params = torch.load(path, map_location='cpu')['net']
    F0_model_pt.load_state_dict(params)
    
    # Initialize JAX model
    from modules_jax.jdcnet import JDCNetJax
    F0_model_jax = JDCNetJax(num_class=1, seq_len=192)
    
    # Initialize with dummy input
    dummy_input = jnp.ones((1, 1, 192, 513))
    variables = F0_model_jax.init(rng, dummy_input)
    
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
    from modules_jax.asrcnn import ASRCNNJax
    asr_model_jax = ASRCNNJax(**asr_model_config)
    
    # Initialize with dummy input
    dummy_mels = jnp.ones((1, 80, 100))
    dummy_mask = jnp.ones((1, 100))
    dummy_text = jnp.ones((1, 20), dtype=jnp.int32)
    variables = asr_model_jax.init(rng, dummy_mels, dummy_mask, dummy_text)
    
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


def build_model_jax(args, text_aligner_model, text_aligner_params,
                  pitch_extractor_model, pitch_extractor_params,
                  bert_model, bert_params, rng):
    """Build JAX model components"""
    # Implement models in JAX/Flax
    
    # Create model dictionary
    models = Munch(
        bert=bert_model,
        bert_params=bert_params,
        bert_encoder=nn.Dense(features=args.hidden_dim),
        
        text_aligner=text_aligner_model,
        text_aligner_params=text_aligner_params,
        
        pitch_extractor=pitch_extractor_model,
        pitch_extractor_params=pitch_extractor_params,
        
        # Additional models would be initialized here
        # ...
    )
    
    # Initialize remaining models with dummy inputs
    # ...
    
    return models


# Weight transfer helper functions
def transfer_jdcnet_weights(pytorch_model, jax_variables):
    """Transfer weights from PyTorch JDCNet to JAX JDCNet"""
    # This would contain detailed weight mapping logic
    params = unfreeze(jax_variables["params"])
    
    # Map weights from PyTorch to JAX format
    # For each layer, extract PyTorch weights and convert to JAX format
    
    # Example:
    # for conv layers: transpose weight dimensions
    # params["conv1"]["kernel"] = np.transpose(pytorch_model.conv1.weight.detach().numpy(), (2, 3, 1, 0))
    
    return freeze({"params": params})


def transfer_asrcnn_weights(pytorch_model, jax_variables):
    """Transfer weights from PyTorch ASRCNN to JAX ASRCNN"""
    params = unfreeze(jax_variables["params"])
    
    # Map weights from PyTorch to JAX format
    
    return freeze({"params": params})


def load_checkpoint_jax(state, optimizer, path):
    """Load checkpoint for JAX models"""
    # Load PyTorch checkpoint
    pt_state = torch.load(path, map_location='cpu')
    pt_params = pt_state['net']
    
    # Convert PyTorch parameters to JAX parameters
    # This would involve detailed mapping logic
    
    # Return updated state
    return state, optimizer, pt_state.get("epoch", 0), pt_state.get("iters", 0)