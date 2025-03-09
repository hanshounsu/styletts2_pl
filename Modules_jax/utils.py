import jax.numpy as jnp
import jax
import flax.linen as nn

# def init_weights(m, mean=0.0, std=0.01):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def init_weights(scale=0.01):
    def _init(key, shape, dtype):
        return scale * jax.random.normal(key, shape, dtype)
    return _init


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def custom_stft(input_signal, n_fft=2048, hop_length=None, win_length=None,
               window=None, center=True, pad_mode='reflect', normalized=False,
               onesided=True, return_complex=None):
    """
    JAX implementation of the Short-Time Fourier Transform (STFT).
    
    Args:
        input_signal: Input signal tensor of shape (..., signal_length)
        n_fft: Size of Fourier transform. Defaults to 2048.
        hop_length: Hop length between frames. Defaults to n_fft // 4.
        win_length: Window length. Defaults to n_fft.
        window: 1-D tensor window. Defaults to jnp.hanning(win_length).
        center: Whether to pad the signal. Defaults to True.
        pad_mode: Padding mode ('reflect', 'constant', 'edge'). Defaults to 'reflect'.
        normalized: Whether to normalize the STFT. Defaults to False.
        onesided: Whether to return only the positive frequencies. Defaults to True.
        return_complex: Whether to return a complex tensor.
        
    Returns:
        STFT of the input signal
    """
    print("abc stft input shape and type", input_signal.shape, input_signal.dtype)
    
    # Set default values
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = jnp.hanning(win_length)
    elif isinstance(window, str):
        # Convert string window type to actual window array
        if window.lower() == 'hann' or window.lower() == 'hanning' or 'hann_window':
            window = jnp.hanning(win_length)
        elif window.lower() == 'hamming':
            window = jnp.hamming(win_length)
        elif window.lower() == 'blackman':
            window = jnp.blackman(win_length)
        elif window.lower() == 'bartlett':
            window = jnp.bartlett(win_length)
        else:
            raise ValueError(f"Unsupported window type: {window}")
            
    if return_complex is None:
        return_complex = jnp.iscomplexobj(input_signal)
    
    # Adjust window size if win_length != n_fft
    if win_length != n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = (n_fft - win_length + 1) // 2
        window = jnp.pad(window, ((pad_left, pad_right),))
    
    # Get input dimensions
    input_shape = input_signal.shape
    signal_length = input_shape[-1]
    
    # Reshape batch dimensions for uniform handling
    input_signal = input_signal.reshape(-1, signal_length)
    batch_size = input_signal.shape[0]
    
    # Pad the signal if center is True
    if center:
        padding = n_fft // 2
        
        # JAX padding modes differ from PyTorch
        if pad_mode == 'reflect':
            # Check if signal is long enough for reflect padding
            if signal_length < padding:
                pad_mode = 'edge'  # Use edge instead of replicate
                
            input_signal = jnp.pad(input_signal, ((0, 0), (padding, padding)), mode=pad_mode)
        else:
            input_signal = jnp.pad(input_signal, ((0, 0), (padding, padding)), mode=pad_mode)
    
    # Calculate number of frames
    padded_length = input_signal.shape[-1]
    num_frames = max(1, (padded_length - n_fft) // hop_length + 1)
    
    # Create frames using a vectorized approach
    def get_frames(signal):
        indices = jnp.arange(n_fft)[None, :] + jnp.arange(num_frames)[:, None] * hop_length
        return signal[indices]
    
    # Apply to each batch element
    frames = jax.vmap(get_frames)(input_signal)
    
    # Apply window
    window_adjusted = window.reshape(1, 1, -1)
    frames = frames * window_adjusted
    
    # Apply normalization if requested
    if normalized:
        frames = frames / jnp.sqrt(jnp.sum(window**2))
    
    # Apply FFT
    stft_matrix = jnp.fft.fft(frames, axis=-1)
    
    # Keep only positive frequencies if onesided is True
    if onesided:
        stft_matrix = stft_matrix[..., :(n_fft // 2) + 1]
    
    # Transpose from (batch, frames, freq) to (batch, freq, frames)
    stft_matrix = jnp.transpose(stft_matrix, (0, 2, 1))
    
    # Reshape back to original batch dimensions
    output_shape = list(input_shape[:-1])
    if onesided:
        output_shape.append(n_fft // 2 + 1)
    else:
        output_shape.append(n_fft)
    output_shape.append(num_frames)
    
    stft_matrix = stft_matrix.reshape(output_shape)
    
    # Handle return format
    if not return_complex:
        # Stack real and imaginary parts
        stft_real = jnp.real(stft_matrix)
        stft_imag = jnp.imag(stft_matrix)
        
        # Expand dims to prepare for stacking
        expand_dims = len(stft_matrix.shape) - 2
        stft_real = jnp.expand_dims(stft_real, axis=expand_dims)
        stft_imag = jnp.expand_dims(stft_imag, axis=expand_dims)
        
        # Stack along the new dimension
        stft_matrix = jnp.concatenate([stft_real, stft_imag], axis=expand_dims)
    
    print("abc stft output shape and type", stft_matrix.shape, stft_matrix.dtype)
    return stft_matrix

# def custom_istft(stft_matrix, n_fft=2048, hop_length=None, win_length=None, 
#                 window=None, center=True, normalized=False, onesided=True, 
#                 length=None, return_complex=False):
#     """
#     JAX implementation of the Inverse Short-Time Fourier Transform (ISTFT).
    
#     Args:
#         stft_matrix: Complex tensor for the STFT result
#         n_fft: Size of Fourier transform. Defaults to 2048.
#         hop_length: Hop length between frames. Defaults to n_fft // 4.
#         win_length: Window length. Defaults to n_fft.
#         window: 1-D tensor window. Defaults to jnp.hanning(win_length).
#         center: Whether the signal was padded. Defaults to True.
#         normalized: Whether the STFT was normalized. Defaults to False.
#         onesided: Whether the STFT was performed on one side only. Defaults to True.
#         length: Length of the original signal. Defaults to None.
#         return_complex: Whether to return a complex tensor. Defaults to False.
        
#     Returns:
#         Time domain signal.
#     """
#     print("abc istft input shape and type", stft_matrix.shape, stft_matrix.dtype)
    
#     # Set default values
#     if hop_length is None:
#         hop_length = n_fft // 4
#     if win_length is None:
#         win_length = n_fft
#     if window is None:
#         window = jnp.hanning(win_length)
    
#     # Adjust window size if win_length != n_fft
#     if win_length != n_fft:
#         pad_left = (n_fft - win_length) // 2
#         pad_right = (n_fft - win_length + 1) // 2
#         window = jnp.pad(window, (pad_left, pad_right))
    
#     # Handle one-sided FFT
#     if onesided:
#         # Make sure the input matches expected dimensions
#         expected_size = n_fft // 2 + 1
#         if stft_matrix.shape[-2] != expected_size:
#             raise ValueError(f"Expected stft_matrix with {expected_size} rows, got {stft_matrix.shape[-2]}")
        
#         # Create the full FFT by using conjugate symmetry
#         if not return_complex and jnp.iscomplexobj(stft_matrix):
#             # Take only the real part for DC and Nyquist
#             stft_matrix = stft_matrix.at[..., 0, :].set(jnp.real(stft_matrix[..., 0, :]))
#             if n_fft % 2 == 0:
#                 stft_matrix = stft_matrix.at[..., -1, :].set(jnp.real(stft_matrix[..., -1, :]))
        
#         # Only expand if needed (i.e., if n_fft > 2)
#         if n_fft > 2:
#             # Create indices for the symmetric part
#             indices = jnp.arange(stft_matrix.shape[-2]-2, 0, -1)
            
#             # Create conjugate symmetric part
#             symmetric_part = jnp.conj(stft_matrix[..., indices, :])
            
#             # Concatenate to get full FFT
#             stft_matrix = jnp.concatenate([stft_matrix, symmetric_part], axis=-2)
    
#     # Apply IFFT along the frequency axis
#     ifft = jnp.fft.ifft(stft_matrix, axis=-2)
#     print("ifft shape and type", ifft.shape, ifft.dtype)
    
#     # If not returning complex, check that the output is real (or nearly real)
#     if not return_complex:
#         ifft = jnp.real(ifft)
    
#     # Get dimensions
#     n_frames = ifft.shape[-1]
#     expected_signal_len = n_fft + hop_length * (n_frames - 1)
    
#     # Initialize output tensor and normalization buffer
#     batch_shape = ifft.shape[:-2]
#     output = jnp.zeros(batch_shape + (expected_signal_len,))
#     norm_buffer = jnp.zeros(batch_shape + (expected_signal_len,))
    
#     # Scale by window if normalized
#     if normalized:
#         ifft = ifft * n_fft
    
#     # Apply window and overlap-add
#     def process_frame(i, carry):
#         output_acc, norm_buffer_acc = carry
#         start = i * hop_length
        
#         # Apply window to current frame
#         windowed_frame = ifft[..., i] * window # [B, n_fft]
        
#         # Use dynamic_update_slice for output
#         output_update = jax.lax.dynamic_update_slice(
#             output_acc,
#             jnp.expand_dims(windowed_frame, axis=0),  # Ensure shape compatibility
#             (0, start)  # Start indices for each dimension
#         )
        
#         # Similar update for norm buffer
#         norm_update = jax.lax.dynamic_update_slice(
#             norm_buffer_acc,
#             jnp.expand_dims(window ** 2, axis=0),  # Ensure shape compatibility
#             (0, start)  # Start indices for each dimension
#         )
    
#         return output_update, norm_update
    
#     # Use fori_loop for the frames processing
#     output, norm_buffer = jax.lax.fori_loop(
#         0, n_frames, 
#         lambda i, val: process_frame(i, val),
#         (output, norm_buffer)
#     )
    
#     # Normalize by the sum of squared windows
#     # Add small epsilon to avoid division by zero
#     epsilon = 1e-10
#     output = output / (norm_buffer + epsilon)
    
#     # Trim the output if original length is provided
#     if length is not None:
#         if center:
#             # If centered, need to trim padding
#             start = n_fft // 2
#             output = output[..., start:start + length]
#         else:
#             # If not centered, just trim to length
#             output = output[..., :length]
#     else:
#         # If no length is given and centered, remove padding
#         if center:
#             start = n_fft // 2
#             end = -(n_fft // 2)
#             output = output[..., start:end]

#     print("abc istft output shape and type", output.shape, output.dtype)
#     return output

def custom_istft(stft_matrix, n_fft=2048, hop_length=None, win_length=None, 
                window=None, center=True, normalized=False, onesided=True, 
                length=None, return_complex=False):
    """
    JAX implementation of the Inverse Short-Time Fourier Transform (ISTFT).
    
    Args:
        stft_matrix: Complex tensor for the STFT result
        n_fft: Size of Fourier transform. Defaults to 2048.
        hop_length: Hop length between frames. Defaults to n_fft // 4.
        win_length: Window length. Defaults to n_fft.
        window: 1-D tensor window. Defaults to jnp.hanning(win_length).
        center: Whether the signal was padded. Defaults to True.
        normalized: Whether the STFT was normalized. Defaults to False.
        onesided: Whether the STFT was performed on one side only. Defaults to True.
        length: Length of the original signal. Defaults to None.
        return_complex: Whether to return a complex tensor. Defaults to False.
        
    Returns:
        Time domain signal.
    """
    # Set default values
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = jnp.hanning(win_length)
    
    # Adjust window size if win_length != n_fft
    if win_length != n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = (n_fft - win_length + 1) // 2
        window = jnp.pad(window, (pad_left, pad_right))
    
    # Handle complex input
    if not jnp.iscomplexobj(stft_matrix):
        # Assume real/imag parts are stacked along dimension -3
        stft_real = stft_matrix[..., 0, :, :]
        stft_imag = stft_matrix[..., 1, :, :]
        stft_matrix = stft_real + 1j * stft_imag
    
    # Handle one-sided FFT by reconstructing the symmetric part
    if onesided:
        # Only expand if needed (more than 2 frequency bins)
        if n_fft > 2:
            # Create indices for the symmetric part (exclude DC and Nyquist)
            indices = jnp.arange(n_fft // 2 - 1, 0, -1)
            # Create conjugate symmetric part
            symmetric_part = jnp.conj(stft_matrix[..., indices, :])
            # Concatenate to get full FFT
            stft_matrix = jnp.concatenate([stft_matrix, symmetric_part], axis=-2)
    
    # Apply IFFT along the frequency axis to get time frames
    ifft_frames = jnp.fft.ifft(stft_matrix, axis=-2)
    
    # Convert to real if not returning complex
    if not return_complex:
        ifft_frames = jnp.real(ifft_frames)
    
    # Apply normalization if needed
    if normalized:
        ifft_frames = ifft_frames * n_fft
    
    # Apply window to each frame
    windowed_frames = ifft_frames * window.reshape(*([1] * (ifft_frames.ndim - 2)), -1, 1)
    
    # Calculate expected output length
    n_frames = windowed_frames.shape[-1]
    expected_signal_len = hop_length * (n_frames - 1) + n_fft
    
    # Vectorized overlap-add approach
    # Create an empty output buffer
    batch_shape = windowed_frames.shape[:-2]
    output = jnp.zeros(batch_shape + (expected_signal_len,))
    norm_buffer = jnp.zeros(batch_shape + (expected_signal_len,))
    
    # Use vmap to create shifted versions of each frame
    def add_frame_at_position(frame_idx, frame):
        start_idx = frame_idx * hop_length
        indices = jnp.arange(n_fft) + start_idx
        
        # Create a one-hot encoding of where this frame contributes
        frame_contribution = jnp.zeros((expected_signal_len,))
        mask = (indices >= 0) & (indices < expected_signal_len)
        valid_indices = indices[mask]
        frame_contribution = frame_contribution.at[valid_indices].set(frame[mask])
        
        # Create the normalization mask
        norm_contribution = jnp.zeros((expected_signal_len,))
        norm_contribution = norm_contribution.at[valid_indices].set((window ** 2)[mask])
        
        return frame_contribution, norm_contribution
    
    # Process all frames and sum contributions
    frame_indices = jnp.arange(n_frames)
    
    # Reshape frames for vmap processing
    flat_batch_size = int(jnp.prod(jnp.array(batch_shape))) if batch_shape else 1
    reshaped_frames = windowed_frames.reshape(flat_batch_size, n_fft, n_frames)
    
    # Process each batch element separately
    def process_batch(batch_frames):
        # For each frame in the batch
        all_contributions = []
        all_norms = []
        
        for i in range(n_frames):
            # Get frame and add it at the right position
            frame = batch_frames[:, i]
            start_idx = i * hop_length
            end_idx = start_idx + n_fft
            
            # Create frame contribution
            frame_contribution = jnp.zeros((expected_signal_len,))
            valid_end = min(end_idx, expected_signal_len)
            valid_length = valid_end - start_idx
            frame_contribution = frame_contribution.at[start_idx:valid_end].set(frame[:valid_length])
            
            # Create normalization contribution
            norm_contribution = jnp.zeros((expected_signal_len,))
            norm_contribution = norm_contribution.at[start_idx:valid_end].set((window ** 2)[:valid_length])
            
            all_contributions.append(frame_contribution)
            all_norms.append(norm_contribution)
            
        # Sum all contributions
        output_sum = jnp.sum(jnp.stack(all_contributions), axis=0)
        norm_sum = jnp.sum(jnp.stack(all_norms), axis=0)
        
        return output_sum, norm_sum
    
    # Process each batch element
    batch_results = jax.vmap(process_batch)(jnp.transpose(reshaped_frames, (0, 2, 1)))
    output = batch_results[0].reshape(batch_shape + (expected_signal_len,))
    norm_buffer = batch_results[1].reshape(batch_shape + (expected_signal_len,))
    
    # Avoid division by zero
    epsilon = 1e-10
    output = output / (norm_buffer + epsilon)
    
    # Trim the output if needed
    if length is not None:
        if center:
            # Trim the center padding
            start = n_fft // 2
            output = output[..., start:start + length]
        else:
            # Just trim to length
            output = output[..., :length]
    elif center:
        # Remove padding if centered
        start = n_fft // 2
        end = -n_fft // 2 if n_fft % 2 == 0 else -(n_fft // 2 + 1)
        output = output[..., start:] if end == 0 else output[..., start:end]
    
    return output

def stft(x, fft_size, hop_size, win_length, window_type):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Array): Input signal array (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window_type (str): Window function type.
    Returns:
        Array: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = custom_stft(x, fft_size, hop_size, win_length, window_type, return_complex=True)
    return jnp.abs(x_stft).transpose(0, 2, 1).real

def custom_leaky_relu(x, negative_slope=0.01):
    """
    Custom Leaky ReLU activation that is fully TPU-friendly.
    
    Args:
        x (torch.Tensor): Input tensor (expected to be real-valued).
        negative_slope (float): Slope for negative values (default 0.01).
    
    Returns:
        torch.Tensor: Output tensor after applying Leaky ReLU.
    """
    # Use torch.where to select between x and negative_slope * x.
    return torch.where(x >= 0, x, negative_slope * x)


# Helper function to create weight normalized initializer (similar to PyTorch weight_norm)
def weight_norm_init():
    def init(key, shape, dtype=jnp.float32):
        # Initialize weights like in PyTorch
        std = 1.0 / jnp.sqrt(shape[0])
        return jax.random.normal(key, shape, dtype) * std
    return init

