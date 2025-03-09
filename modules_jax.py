import jax.numpy as jnp
import jax
import flax.linen as nn

class AdaIN1d(nn.Module):
    style_dim: int
    num_features: int

    @nn.compact
    def __call__(self, x, s):
        # Instance normalization without affine parameters / x shape: [B, C, L]
        mean = jnp.mean(x, axis=2, keepdims=True)
        var = jnp.var(x, axis=2, keepdims=True) + 1e-8
        norm_x = (x - mean) / jnp.sqrt(var)
        
        # Style mapping to scale and bias
        h = nn.Dense(self.num_features * 2)(s)
        h = h.reshape(h.shape[0], 1, h.shape[1])
        gamma, beta = jnp.split(h, 2, axis=2)
        
        return (1 + gamma) * norm_x + beta

# class AdaIN1d(nn.Module):
#     style_dim: int
#     num_features: int
    
#     def setup(self):
#         self.norm = nn.GroupNorm(num_groups=1, epsilon=1e-5, use_bias=False, use_scale=False)
#         self.fc = nn.Dense(features=self.num_features * 2)
    
#     def __call__(self, x, s, training=False):
#         h = self.fc(s)
#         h = h.reshape(h.shape[0], h.shape[1], 1)
#         gamma, beta = jnp.split(h, 2, axis=1)
#         return (1 + gamma) * self.norm(x) + beta