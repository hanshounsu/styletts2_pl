import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
import numpy as np
import matplotlib.pyplot as plt
from munch import Munch
from typing import Optional, List, Dict, Tuple, Any


# Note: monotonic_align may need a JAX-compatible implementation
# This is a placeholder for the JAX version of maximum_path
def maximum_path_jax(neg_cent, mask):
    """ JAX version of monotonic alignment function.
    neg_cent: [b, t_t, t_s] - JAX array
    mask: [b, t_t, t_s] - JAX array
    
    Returns: alignment path as a JAX array
    """
    # For now, we'll convert to numpy, use the existing C function, then convert back
    # In a production setting, you would reimplement or find a JAX-native version
    from monotonic_align.core import maximum_path_c
    
    # Convert JAX arrays to numpy
    neg_cent_np = np.ascontiguousarray(np.array(neg_cent).astype(np.float32))
    path = np.ascontiguousarray(np.zeros(neg_cent_np.shape, dtype=np.int32))
    
    # Calculate sums for lengths
    t_t_max = np.ascontiguousarray(np.sum(np.array(mask), axis=1)[:, 0].astype(np.int32))
    t_s_max = np.ascontiguousarray(np.sum(np.array(mask), axis=2)[:, 0].astype(np.int32))
    
    # Call the C function
    maximum_path_c(path, neg_cent_np, t_t_max, t_s_max)
    
    # Convert back to JAX array
    return jnp.array(path)


def get_data_path_list(train_path: Optional[str] = None, 
                       val_path: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """Get paths for training and validation data."""
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list


def length_to_mask(lengths: jnp.ndarray) -> jnp.ndarray:
    """Convert lengths to boolean mask.
    
    Args:
        lengths: JAX array of shape (batch_size,) containing lengths
        
    Returns:
        Boolean mask of shape (batch_size, max_length)
    """
    max_len = jnp.max(lengths)
    mask = jnp.arange(max_len) < jnp.expand_dims(lengths, axis=1)
    return ~mask  # Invert to match PyTorch gt behavior


def log_norm(x: jnp.ndarray, mean: float = -4, std: float = 4, axis: int = 2) -> jnp.ndarray:
    """Calculate log of norm.
    
    Args:
        x: Input tensor
        mean: Mean for denormalization
        std: Standard deviation for denormalization
        axis: Axis along which to compute norm
        
    Returns:
        Log of the norm of denormalized input
    """
    # Denormalize log mel -> mel
    x_denorm = jnp.exp(x * std + mean)
    # Compute norm along specified axis
    x_norm = jnp.linalg.norm(x_denorm, axis=axis)
    # Take log
    return jnp.log(x_norm)


def get_image(arrs: jnp.ndarray) -> plt.Figure:
    """Create a matplotlib figure from an array.
    
    Args:
        arrs: Array to visualize
        
    Returns:
        Matplotlib figure
    """
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(np.array(arrs))  # Convert JAX array to numpy for matplotlib
    return fig


def recursive_munch(d: Any) -> Any:
    """Recursively convert dictionaries to Munch objects.
    
    Args:
        d: Input data structure (dict, list, or base type)
        
    Returns:
        Converted structure with dicts as Munch objects
    """
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d


def log_print(message: str, logger) -> None:
    """Log a message and print it to stdout.
    
    Args:
        message: Message to log
        logger: Logger object
    """
    logger.info(message)
    print(message)


# Additional JAX-specific utilities that might be useful
def batch_to_device(batch: Dict[str, jnp.ndarray], device=None) -> Dict[str, jnp.ndarray]:
    """Move a batch of data to the specified device (for API compatibility).
    
    In JAX, we don't explicitly move data to devices like in PyTorch.
    This function is provided for compatibility with PyTorch code.
    
    Args:
        batch: Dictionary of arrays
        device: Ignored in JAX
        
    Returns:
        The same batch dictionary
    """
    # JAX handles device placement automatically
    return batch


def param_count(params) -> int:
    """Count parameters in a JAX model.
    
    Args:
        params: JAX model parameters
        
    Returns:
        Number of parameters
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

# Add this function to your code - place it before the main function
def shard_batch(batch):
    """
    Shards a batch for data-parallel training across multiple devices.
    
    Args:
        batch: A tuple or list of arrays/tensors
        
    Returns:
        A tuple of sharded arrays
    """
    # Get the number of local devices
    n_devices = jax.local_device_count()
    
    # Process each element in the batch
    def _shard_array(x):
        # Skip non-array elements (like strings or None)
        if not isinstance(x, (np.ndarray, jnp.ndarray)) or x is None:
            return x
        
        # Calculate the batch size
        batch_size = x.shape[0]
        
        # Make sure batch size is divisible by device count
        assert batch_size % n_devices == 0, f"Batch size {batch_size} must be divisible by device count {n_devices}"
        
        # Reshape to (n_devices, batch_size_per_device, ...)
        per_device = batch_size // n_devices
        return x.reshape((n_devices, per_device) + x.shape[1:])
    
    # Apply sharding to each element in the batch
    if isinstance(batch, (list, tuple)):
        return type(batch)(_shard_array(x) for x in batch)
    else:
        # If batch is a single array
        return _shard_array(batch)

def slice_audio_for_training(asr, mels, wav, mel_input_length, max_len, rng):
    """Extract random fixed-length segments for training.
    
    Args:
        asr: Hidden representations from text encoder with attention applied (B, C, T_mel)
        mels: Mel spectrograms (B, C, T_mel*2)
        wav: Raw audio waveforms (B, T_audio)
        mel_input_length: Valid length for each mel spectrogram
        max_len: Maximum segment length (in encoder time steps)
        rng: JAX random key
        
    Returns:
        Tuple containing sliced encoder features, mel spectrograms, and raw audio
    """
    # Get batch size
    batch_size = asr.shape[0]
    
    # Calculate valid segment length (ensure it's not too long)
    mel_len = jnp.minimum(jnp.min(mel_input_length // 2 - 1), max_len // 2)
    
    # Create random starting positions (constrained by valid lengths)
    rng, subkey = random.split(rng)
    max_start_positions = jnp.maximum(0, mel_input_length // 2 - mel_len - 1)
    random_starts = random.randint(
        subkey, 
        shape=(batch_size,), 
        minval=0, 
        maxval=jnp.maximum(1, max_start_positions)
    )
    
    # Batch indices
    indices = jnp.arange(batch_size)
    
    # Extract segments using vectorized operations
    # Encoder features (at encoder temporal resolution)
    en = jax.vmap(lambda i, start: asr[i, :, start:start+mel_len])(indices, random_starts)
    
    # Mel spectrograms (at spectrogram temporal resolution, 2x encoder)
    gt = jax.vmap(lambda i, start: mels[i, :, (start * 2):((start+mel_len) * 2)])(indices, random_starts)
    
    # Raw audio (at waveform temporal resolution, 300x mel for 24kHz audio)
    hop_size = 300  # Assuming 24kHz with 300 hop size
    wav_clips = jax.vmap(lambda i, start: wav[i][(start * 2) * hop_size:((start+mel_len) * 2) * hop_size])(
        indices, random_starts
    )
    
    return en, gt, wav_clips

def create_attention_mask(mel_mask, text_mask):
    """Create attention mask from mel and text masks for JAX.
    
    Args:
        mel_mask: (batch_size, mel_len) boolean mask
        text_mask: (batch_size, text_len) boolean mask
        
    Returns:
        attention_mask: (batch_size, text_len, mel_len) boolean mask
    """
    with jax.lax.stop_gradient():
        # First part - expand mask
        attn_mask = ~mel_mask  # Invert the mask
        attn_mask = jnp.expand_dims(attn_mask, axis=-1)  # Add dimension at end
        # Broadcast to desired shape 
        attn_mask = jnp.broadcast_to(
            attn_mask, 
            (mel_mask.shape[0], mel_mask.shape[1], text_mask.shape[-1])
        )
        attn_mask = jnp.transpose(attn_mask, (0, 2, 1))  # Transpose last two dimensions
        
        # Second part - expand text_mask
        text_mask_expanded = ~text_mask  # Invert text mask
        text_mask_expanded = jnp.expand_dims(text_mask_expanded, axis=-1)
        text_mask_expanded = jnp.broadcast_to(
            text_mask_expanded, 
            (text_mask.shape[0], text_mask.shape[1], mel_mask.shape[-1])
        )
        
        # Combine masks
        attn_mask = attn_mask.astype(jnp.float32) * text_mask_expanded.astype(jnp.float32)
        attn_mask = attn_mask < 1  # Convert back to boolean (True where padding is applied)
    
    return attn_mask