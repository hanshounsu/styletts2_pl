
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Union

# JAX utility functions for spectral_norm and weight_norm
def weight_norm_jax(x, axis=0):
    """JAX equivalent of PyTorch's weight_norm"""
    g = jnp.sqrt(jnp.sum(x**2, axis=axis, keepdims=True))
    return x * g

class SpectralNormConv(nn.Module):
    """Convolutional layer with spectral normalization for both 1D and 2D"""
    features: int
    kernel_size: Union[Tuple[int, ...], int]
    strides: Union[Tuple[int, ...], int] = None  # Default handled in __call__
    padding: Union[str, Tuple[Tuple[int, int], ...]] = 'VALID'
    n_power_iterations: int = 1
    eps: float = 1e-12
    use_bias: bool = True
    
    @nn.compact
    def __call__(self, x, training=False):
        input_dim = x.shape[-1]
        
        # Handle kernel_size and determine if 1D or 2D convolution
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size,)
            is_1d = True
        else:
            kernel_size = self.kernel_size
            is_1d = len(kernel_size) == 1
        
        # Handle strides with appropriate defaults
        if self.strides is None:
            strides = (1,) if is_1d else (1, 1)
        elif isinstance(self.strides, int):
            strides = (self.strides,)
        else:
            strides = self.strides
        
        # Create kernel shape based on dimensionality
        kernel_shape = (*kernel_size, input_dim, self.features)
        
        # Create kernel parameter
        kernel = self.param('kernel', 
                          nn.initializers.normal(0.02),
                          kernel_shape)
        
        # Reshape kernel for spectral norm calculation
        kernel_flat = kernel.reshape(-1, self.features)
        
        # Initialize or get u and v vectors (preserved between calls)
        u = self.variable('spectral_norm', 'u', 
                         lambda: jax.random.normal(self.make_rng('params'), (self.features,)))
        v = self.variable('spectral_norm', 'v',
                         lambda: jax.random.normal(self.make_rng('params'), 
                                                (kernel_flat.shape[0],)))
        
        # Normalize initial vectors if needed
        if jnp.all(jnp.equal(u.value, 0)):
            u_init = jax.random.normal(self.make_rng('params'), (self.features,))
            v_init = jax.random.normal(self.make_rng('params'), (kernel_flat.shape[0],))
            u.value = u_init / (jnp.linalg.norm(u_init) + self.eps)
            v.value = v_init / (jnp.linalg.norm(v_init) + self.eps)
        
        # Power iteration when training
        u_hat = u.value
        v_hat = v.value
        
        if training:
            for _ in range(self.n_power_iterations):
                v_hat = jnp.matmul(kernel_flat, u_hat)
                v_hat = v_hat / (jnp.linalg.norm(v_hat) + self.eps)
                
                u_hat = jnp.matmul(kernel_flat.T, v_hat)
                u_hat = u_hat / (jnp.linalg.norm(u_hat) + self.eps)
            
            # Update persistent vectors if training
            u.value = u_hat
            v.value = v_hat
        
        # Compute spectral norm and normalize kernel
        sigma = jnp.sum(u_hat * jnp.matmul(kernel_flat.T, v_hat))
        normalized_kernel = kernel / sigma
        
        # Create bias if needed
        bias = None
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
        
        # Apply convolution with normalized kernel, using dimension format based on dimensionality
        if is_1d:
            # 1D convolution
            out = jax.lax.conv_general_dilated(
                x, normalized_kernel, strides, self.padding,
                dimension_numbers=('NHC', 'HIO', 'NHC')
            )
        else:
            # 2D convolution
            out = jax.lax.conv_general_dilated(
                x, normalized_kernel, strides, self.padding,
                dimension_numbers=('NHWC', 'HWIO', 'NHWC')
            )
        
        # Add bias if present
        if bias is not None:
            out = out + bias
            
        return out

# Function to use as a drop-in replacement for nn.Conv with spectral norm
# Updated function to work with the merged class
def spectral_norm_jax(conv_layer, n_power_iterations=1):
    """Creates a spectrally normalized version of a Conv layer (both conv1d and conv2d supported)"""
    features = conv_layer.features
    kernel_size = conv_layer.kernel_size
    strides = getattr(conv_layer, 'strides', None)
    padding = getattr(conv_layer, 'padding', 'VALID')
    use_bias = getattr(conv_layer, 'use_bias', True)
    
    return SpectralNormConv(
        features=features,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        n_power_iterations=n_power_iterations,
        use_bias=use_bias
    )


class WeightNormConv(nn.Module):
    """Convolutional layer with weight normalization"""
    features: int
    kernel_size: Tuple[int, ...] 
    strides: Tuple[int, ...] = (1, 1)
    padding: str = 'VALID'
    use_bias: bool = True
    feature_group_count: int = 1
    
    @nn.compact
    def __call__(self, x, training=False):
        input_dim = x.shape[-1]
        kernel_shape = (*self.kernel_size, input_dim // self.feature_group_count, self.features)
        
        # Create kernel parameters - direction (v) and magnitude (g)
        v = self.param('kernel_v', 
                     nn.initializers.lecun_normal(), 
                     kernel_shape)
        g = self.param('kernel_g', 
                     lambda rng, shape: jnp.ones(shape), 
                     (self.features,))
        
        # Compute the normalized weight
        norm = jnp.sqrt(jnp.sum(jnp.square(v), axis=(0, 1, 2)) + 1e-10)
        w = g[None, None, None, :] * v / norm[None, None, None, :]
        
        # Optional bias
        bias = None
        if self.use_bias:
            bias = self.param('bias', 
                            nn.initializers.zeros, 
                            (self.features,))
            
        # Apply convolution with normalized weights
        out = jax.lax.conv_general_dilated(
            x, w, self.strides, self.padding,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=self.feature_group_count
        )
        
        if bias is not None:
            out = out + bias
            
        return out

class WeightNormConv1d(nn.Module):
    """1D Convolutional layer with weight normalization"""
    features: int
    kernel_size: Union[Tuple[int, ...], int]
    strides: Union[Tuple[int, ...], int] = (1,)
    padding: str = 'VALID'
    use_bias: bool = True
    feature_group_count: int = 1
    
    @nn.compact
    def __call__(self, x, training=False):
        input_dim = x.shape[-1]
        
        # Ensure kernel_size and strides are tuples
        kernel_size = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size,)
        strides = self.strides if isinstance(self.strides, tuple) else (self.strides,)
        
        kernel_shape = (*kernel_size, input_dim // self.feature_group_count, self.features)
        
        # Create kernel parameters - direction (v) and magnitude (g)
        v = self.param('kernel_v', 
                     nn.initializers.lecun_normal(), 
                     kernel_shape)
        g = self.param('kernel_g', 
                     lambda rng, shape: jnp.ones(shape), 
                     (self.features,))
        
        # Compute the normalized weight
        norm = jnp.sqrt(jnp.sum(jnp.square(v), 
                                axis=tuple(range(len(kernel_size))) + (len(kernel_size),)) + 1e-10)
        w = g[None, None, :] * v / norm[None, None, :]
        
        # Optional bias
        bias = None
        if self.use_bias:
            bias = self.param('bias', 
                            nn.initializers.zeros, 
                            (self.features,))
            
        # Apply convolution with normalized weights
        out = jax.lax.conv_general_dilated(
            x, w, strides, self.padding,
            dimension_numbers=('NHC', 'HIO', 'NHC'), # Input array layout (N=batch, H=spatial, C=channels) : NHC, Kernel layout(H=spatial,I=Input Channels,O=Output Channels) : HIO, Output array layout : NHC
            feature_group_count=self.feature_group_count
        )
        
        if bias is not None:
            out = out + bias
            
        return out

class WeightNormConvTranspose1d(nn.Module):
    """1D Transposed Convolutional layer with weight normalization"""
    features: int
    kernel_size: Union[Tuple[int, ...], int]
    strides: Union[Tuple[int, ...], int] = (1,)
    padding: str = 'VALID'
    use_bias: bool = True
    
    @nn.compact
    def __call__(self, x, training=False):
        input_dim = x.shape[-1]
        
        # Ensure kernel_size and strides are tuples
        kernel_size = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size,)
        strides = self.strides if isinstance(self.strides, tuple) else (self.strides,)
        
        # CHANGED: Swapped dimensions for JAX's expected transposed conv kernel layout
        kernel_shape = (*kernel_size, input_dim, self.features)
        
        # Create kernel parameters - direction (v) and magnitude (g)
        v = self.param('kernel_v', 
                     nn.initializers.lecun_normal(), 
                     kernel_shape)
        g = self.param('kernel_g', 
                     lambda rng, shape: jnp.ones(shape), 
                     (self.features,))
        
        # CHANGED: Updated norm calculation axes for the new shape
        norm = jnp.sqrt(jnp.sum(jnp.square(v), axis=(0, 1), keepdims=True) + 1e-10)
        g = g.reshape(1, 1, -1)  # Shape becomes (1, 1, features)
        w = v * (g / norm)
        
        # Rest of the method remains the same...
        
        # Optional bias
        bias = None
        if self.use_bias:
            bias = self.param('bias', 
                            nn.initializers.zeros, 
                            (self.features,))
            
        # Apply transposed convolution with normalized weights
        out = jax.lax.conv_transpose(
            x, w, strides, self.padding,
            dimension_numbers=('NHC', 'HIO', 'NHC')
        )
        
        if bias is not None:
            out = out + bias[None, None, :]
            
        return out
def weight_norm_jax(layer):
    """Creates a weight normalized version of a Conv or ConvTranspose layer"""
    features = layer.features
    kernel_size = layer.kernel_size
    
    # Ensure kernel_size is a tuple
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)
        
    strides = getattr(layer, 'strides', (1, 1))
    # Ensure strides is a tuple
    if isinstance(strides, int):
        strides = (strides,)
        
    padding = getattr(layer, 'padding', 'VALID')
    use_bias = getattr(layer, 'use_bias', True)
    
    # Check if this is a transposed convolution
    is_transpose = layer.__class__.__name__ == 'ConvTranspose'
    
    if is_transpose:
        # It's a transposed convolution
        if len(kernel_size) == 1:
            return WeightNormConvTranspose1d(
                features=features,
                kernel_size=kernel_size[0],
                strides=strides[0] if len(strides) == 1 else strides,
                padding=padding,
                use_bias=use_bias
            )
        # Add WeightNormConvTranspose2d if needed
    else:
        # Regular convolution
        feature_group_count = getattr(layer, 'feature_group_count', 1)
        if len(kernel_size) > 1:
            return WeightNormConv(
                features=features,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                feature_group_count=feature_group_count
            )
        else:
            return WeightNormConv1d(
                features=features,
                kernel_size=kernel_size[0],
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                feature_group_count=feature_group_count
            )