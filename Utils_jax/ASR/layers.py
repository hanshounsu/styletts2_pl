import math
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Any, Tuple, Optional, Dict, Callable, Sequence

def _get_activation_fn(activ):
    """Get activation function by name"""
    if activ == 'relu':
        return nn.relu
    elif activ == 'lrelu':
        return lambda x: nn.leaky_relu(x, negative_slope=0.2)
    elif activ == 'swish':
        return lambda x: x * jax.nn.sigmoid(x)
    else:
        raise RuntimeError(f'Unexpected activ type {activ}, expected [relu, lrelu, swish]')


class LinearNorm(nn.Module):
    """Linear layer with Xavier initialization"""
    in_features: int
    out_features: int
    bias: bool = True
    w_init_gain: str = 'linear'
    
    def setup(self):
        # Calculate gain based on activation following PyTorch's init.calculate_gain logic
        if self.w_init_gain == 'linear':
            gain = 1.0
        elif self.w_init_gain == 'tanh':
            gain = 5.0/3.0
        elif self.w_init_gain == 'relu':
            gain = math.sqrt(2.0)
        else:
            gain = 1.0  # Default
        
        # Xavier uniform initialization (corresponds to torch.nn.init.xavier_uniform_)
        limit = math.sqrt(6.0 / (self.in_features + self.out_features)) * gain
        
        def xavier_uniform(key, shape):
            return jax.random.uniform(key, shape, minval=-limit, maxval=limit)
        
        self.weight = self.param('weight', xavier_uniform, (self.in_features, self.out_features))
        
        if self.bias:
            self.bias_param = self.param('bias', jax.nn.initializers.zeros, (self.out_features,))
    
    def __call__(self, x):
        x = jnp.matmul(x, self.weight)
        if self.bias:
            x = x + self.bias_param
        return x


class ConvNorm(nn.Module):
    """1D Convolutional layer with Xavier initialization"""
    in_channels: int  # Required parameter
    out_channels: int
    kernel_size: int = 1
    stride: int = 1
    padding: Optional[int] = None
    dilation: int = 1
    bias: bool = True
    w_init_gain: str = 'linear'
    
    def setup(self):
        # Calculate padding if not provided
        padding = self.padding
        if padding is None:
            assert(self.kernel_size % 2 == 1)
            padding = int(self.dilation * (self.kernel_size - 1) / 2)
        self.padding_val = padding
        
        # Calculate gain based on activation
        if self.w_init_gain == 'linear':
            gain = 1.0
        elif self.w_init_gain == 'tanh':
            gain = 5.0/3.0
        elif self.w_init_gain == 'relu':
            gain = math.sqrt(2.0)
        else:
            gain = 1.0
        
        # Xavier uniform initialization
        fan_in = self.in_channels * self.kernel_size
        fan_out = self.out_channels * self.kernel_size
        limit = math.sqrt(6.0 / (fan_in + fan_out)) * gain
        
        def xavier_uniform(key, shape):
            return jax.random.uniform(key, shape, minval=-limit, maxval=limit)
        
        self.weight = self.param('weight', xavier_uniform, 
                               (self.out_channels, self.in_channels, self.kernel_size))
        
        if self.bias:
            self.bias_param = self.param('bias', jax.nn.initializers.zeros, (self.out_channels,))
    
    def __call__(self, signal):
        # Apply 1D convolution manually (this gives us more control over dimensions)
        x = jax.lax.conv_general_dilated(
            signal,
            self.weight,
            window_strides=(self.stride,),
            padding=((self.padding_val, self.padding_val),),
            lhs_dilation=(1,),
            rhs_dilation=(self.dilation,),
            dimension_numbers=('NCH', 'OIH', 'NCH')
        )
        
        # Add bias if needed
        if self.bias:
            x = x + self.bias_param[:, None]
        
        return x


class ConvBlock(nn.Module):
    """Block of convolutional layers with residual connections"""
    hidden_dim: int
    n_conv: int = 3
    dropout_p: float = 0.2
    activ: str = 'relu'
    
    def setup(self):
        self.act_fn = _get_activation_fn(self.activ)
        
        # Create convolutional blocks with residual connections
        self.conv_layers = [{
            'conv1': ConvNorm(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=3,
                padding=3**i,
                dilation=3**i
            ),
            'norm1': nn.GroupNorm(num_groups=8, epsilon=1e-5),
            'drop1': nn.Dropout(rate=self.dropout_p),
            
            # Second set of layers
            'conv2': ConvNorm(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=3,
                padding=1,
                dilation=1
            ),
            'drop2': nn.Dropout(rate=self.dropout_p)
        } for i in range(self.n_conv)]
    
    def __call__(self, x, training=True):
        # Apply convolutional blocks
        for i in range(self.n_conv):
            res = x
            
            # First conv layer
            x = self.conv_layers[i]['conv1'](x)
            x = self.act_fn(x)
            x = jnp.transpose(x, (0, 2, 1))
            x = self.conv_layers[i]['norm1'](x)
            x = jnp.transpose(x, (0, 2, 1))
            x = self.conv_layers[i]['drop1'](x, deterministic=not training)
            
            # Second conv layer
            x = self.conv_layers[i]['conv2'](x)
            x = self.act_fn(x)
            x = self.conv_layers[i]['drop2'](x, deterministic=not training)
            
            # Add residual connection
            x = x + res
            
        return x


class LocationLayer(nn.Module):
    """Location-based attention layer"""
    attention_n_filters: int
    attention_kernel_size: int
    attention_dim: int
    
    def setup(self):
        padding = int((self.attention_kernel_size - 1) / 2)
        
        # Location convolution
        self.conv = ConvNorm(
            in_channels=2,
            out_channels=self.attention_n_filters,
            kernel_size=self.attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1
        )
        
        # Dense projection
        self.linear = LinearNorm(
            in_features=self.attention_n_filters,
            out_features=self.attention_dim,
            bias=False,
            w_init_gain='tanh'
        )
    
    def __call__(self, attention_weights_cat):
        # Process with convolution
        processed_attention = self.conv(attention_weights_cat)
        
        # Transpose from [B, C, T] to [B, T, C]
        processed_attention = jnp.transpose(processed_attention, (0, 2, 1))
        
        # Apply linear projection
        processed_attention = self.linear(processed_attention)
        
        return processed_attention


class Attention(nn.Module):
    """Attention mechanism for sequence-to-sequence models"""
    attention_rnn_dim: int
    embedding_dim: int
    attention_dim: int
    attention_location_n_filters: int
    attention_location_kernel_size: int
    
    def setup(self):
        self.query_layer = LinearNorm(
            in_features=self.attention_rnn_dim,
            out_features=self.attention_dim,
            bias=False,
            w_init_gain='tanh'
        )
        
        self.memory_layer = LinearNorm(
            in_features=self.embedding_dim,
            out_features=self.attention_dim,
            bias=False,
            w_init_gain='tanh'
        )
        
        self.v = LinearNorm(
            in_features=self.attention_dim,
            out_features=1,
            bias=False
        )
        
        self.location_layer = LocationLayer(
            attention_n_filters=self.attention_location_n_filters,
            attention_kernel_size=self.attention_location_kernel_size,
            attention_dim=self.attention_dim
        )
        
        self.score_mask_value = -float("inf")
    
    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        """
        Calculate alignment energies
        
        Args:
            query: decoder output (batch, n_mel_channels * n_frames_per_step)
            processed_memory: processed encoder outputs (B, T_in, attention_dim)
            attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
            
        Returns:
            alignment (batch, max_time)
        """
        # Process query to [B, 1, attention_dim]
        processed_query = self.query_layer(jnp.expand_dims(query, 1))
        
        # Process attention weights to [B, max_time, attention_dim]
        processed_attention_weights = self.location_layer(attention_weights_cat)
        
        # Calculate energies and squeeze
        energies = self.v(nn.tanh(
            processed_query + processed_attention_weights + processed_memory))
        
        energies = jnp.squeeze(energies, -1)
        return energies

    def __call__(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        Attention mechanism forward pass
        
        Args:
            attention_hidden_state: attention rnn last output (B, attention_rnn_dim)
            memory: encoder outputs (B, max_time, attention_dim)
            processed_memory: processed encoder outputs (B, max_time, attention_dim)
            attention_weights_cat: previous and cummulative attention weights (B, 2, max_time)
            mask: binary mask for padded data (B, max_time)
            
        Returns:
            attention_context: context vector (B, attention_dim)
            attention_weights: attention weights (B, max_time)
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)
        
        # Apply mask if provided - ensure proper broadcasting
        if mask is not None:
            # Ensure mask has same shape as alignment
            if mask.ndim < alignment.ndim:
                mask = jnp.expand_dims(mask, axis=-1)
                
            alignment = jnp.where(mask, alignment, jnp.full_like(alignment, self.score_mask_value))
        
        # Apply softmax to get attention weights
        attention_weights = jax.nn.softmax(alignment, axis=1)
        
        # Calculate context vector
        attention_context = jnp.matmul(
            jnp.expand_dims(attention_weights, 1), 
            memory
        )
        attention_context = jnp.squeeze(attention_context, 1)
        
        return attention_context, attention_weights


class MFCC(nn.Module):
    """MFCC transformation layer"""
    n_mfcc: int = 40
    n_mels: int = 80
    
    def setup(self):
        # Create DCT matrix (following torchaudio implementation)
        self.norm = 'ortho'
        n = self.n_mfcc
        k = jnp.arange(n).reshape((n, 1))
        p = jnp.arange(self.n_mels).reshape((1, self.n_mels))
        
        # Create DCT matrix similar to torchaudio's create_dct
        dct_mat = jnp.cos((jnp.pi * k * (2 * p + 1)) / (2 * self.n_mels))
        
        if self.norm == 'ortho':
            dct_mat = dct_mat * jnp.sqrt(2 / self.n_mels)
            dct_mat = dct_mat.at[0, :].set(dct_mat[0, :] / jnp.sqrt(2))
        
        self.dct_mat = dct_mat
    
    def __call__(self, mel_specgram):
        # Handle different input dimensions
        orig_shape = mel_specgram.shape
        unsqueezed = False
        
        # Handle 2D input - add batch dimension if missing
        if len(orig_shape) == 2:
            mel_specgram = jnp.expand_dims(mel_specgram, 0)
            unsqueezed = True
        
        # Apply DCT transform (carefully handling dimensions)
        # DCT formula: DCT(x) = D * x where D is the DCT matrix
        # We need to apply this to each channel independently
        
        # First transpose to get time dimension last: [B, C, T] -> [B, T, C]
        mel_transposed = jnp.transpose(mel_specgram, (0, 2, 1))
        
        # Apply matrix multiplication: [B, T, n_mels] @ [n_mels, n_mfcc] -> [B, T, n_mfcc]
        mfcc = jnp.matmul(mel_transposed, self.dct_mat.T)
        
        # Transpose back: [B, T, n_mfcc] -> [B, n_mfcc, T]
        mfcc = jnp.transpose(mfcc, (0, 2, 1))
        
        # Restore original dimensions if needed
        if unsqueezed:
            mfcc = jnp.squeeze(mfcc, 0)
            
        return mfcc