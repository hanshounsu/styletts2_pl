import torch
import torch.nn.functional as F

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

import torch
import numpy as np

def custom_istft(stft_matrix, n_fft=2048, hop_length=None, win_length=None, 
                window=None, center=True, normalized=False, onesided=True, 
                length=None, return_complex=False):
    """
    Custom implementation of the Inverse Short-Time Fourier Transform (ISTFT).
    
    Args:
        stft_matrix (Tensor): Complex tensor of shape (..., n_fft // 2 + 1, n_frames) 
                            or (..., n_fft, n_frames) if onesided=False.
        n_fft (int, optional): Size of Fourier transform. Defaults to 2048.
        hop_length (int, optional): Hop length between frames. Defaults to n_fft // 4.
        win_length (int, optional): Window length. Defaults to n_fft.
        window (Tensor, optional): 1-D tensor window. Defaults to torch.hann_window(win_length).
        center (bool, optional): Whether the signal was padded. Defaults to True.
        normalized (bool, optional): Whether the STFT was normalized. Defaults to False.
        onesided (bool, optional): Whether the STFT was performed on one side only. Defaults to True.
        length (int, optional): Length of the original signal. Defaults to None.
        return_complex (bool, optional): Whether to return a complex tensor. Defaults to False.
        
    Returns:
        Tensor: Time domain signal of shape (..., signal_length).
    """
    print("abc istft input shape and type", stft_matrix.shape, stft_matrix.dtype)
    # Set default values
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(win_length, device=stft_matrix.device)
    
    # Adjust window size if win_length != n_fft
    if win_length != n_fft:
        window = torch.nn.functional.pad(window, [(n_fft - win_length) // 2, (n_fft - win_length + 1) // 2])
    
    # Ensure window is on the same device as stft_matrix
    window = window.to(stft_matrix.device)
    
    # Handle one-sided FFT
    if onesided:
        # Make sure the input matches expected dimensions
        expected_size = n_fft // 2 + 1
        if stft_matrix.shape[-2] != expected_size:
            raise ValueError(f"Expected stft_matrix with {expected_size} rows, got {stft_matrix.shape[-2]}")
        
        # Create the full FFT by using conjugate symmetry
        if not return_complex and stft_matrix.is_complex():
            # Take only the real part for DC and Nyquist
            stft_matrix[..., 0, :] = stft_matrix[..., 0, :].real
            if n_fft % 2 == 0:
                stft_matrix[..., -1, :] = stft_matrix[..., -1, :].real
        
        # Only expand if needed (i.e., if n_fft > 2)
        if n_fft > 2:
            # Get dimensions for easier handling
            shape = list(stft_matrix.shape)
            shape[-2] = n_fft - shape[-2]
            
            # Create conjugate symmetric part
            indices = torch.arange(shape[-2], 0, -1, device=stft_matrix.device)
            symmetric_part = torch.conj(stft_matrix[..., indices, :])
            
            # Concatenate to get full FFT
            stft_matrix = torch.cat([stft_matrix, symmetric_part], dim=-2)
    
    # Apply IFFT along the frequency axis
    ifft = torch.fft.ifft(stft_matrix, dim=-2)
    print("ifft shape and type", ifft.shape, ifft.dtype)
    print("ifft : ", ifft)
    
    # If not returning complex, check that the output is real (or nearly real)
    if not return_complex:
        # if ifft.is_complex():
        #     if torch.max(torch.abs(ifft.imag)) > 1e-5:
        #         print("Warning: ISTFT result has a significant imaginary component which will be discarded")
        ifft = ifft.real
    
    # Get dimensions
    n_frames = ifft.shape[-1]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    
    # Initialize output tensor
    output = torch.zeros(*ifft.shape[:-2], expected_signal_len, device=ifft.device)
    
    # Initialize normalization buffer
    norm_buffer = torch.zeros(*ifft.shape[:-2], expected_signal_len, device=ifft.device)
    
    # Scale by window if normalized
    if normalized:
        ifft = ifft * n_fft
    
    # Apply window and overlap-add
    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        
        # Apply window
        windowed_frame = ifft[..., i] * window
        
        # Overlap-add
        output[..., start:end] += windowed_frame
        norm_buffer[..., start:end] += window ** 2
    
    # Normalize by the sum of squared windows
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    output = output / (norm_buffer + epsilon)
    
    # Trim the output if original length is provided
    if length is not None:
        if center:
            # If centered, need to trim padding
            start = n_fft // 2
            output = output[..., start:start + length]
        else:
            # If not centered, just trim to length
            output = output[..., :length]
    else:
        # If no length is given and centered, remove padding
        if center:
            start = n_fft // 2
            end = -(n_fft // 2)
            output = output[..., start:end]

    print("abc istft output shape and type", output.shape, output.dtype)
    
    return output

def custom_stft(input_signal, n_fft=2048, hop_length=None, win_length=None,
               window=None, center=True, pad_mode='reflect', normalized=False,
               onesided=True, return_complex=None):
    """
    Custom implementation of the Short-Time Fourier Transform (STFT).
    
    Args:
        input_signal (Tensor): Input signal tensor of shape (..., signal_length)
        n_fft (int, optional): Size of Fourier transform. Defaults to 2048.
        hop_length (int, optional): Hop length between frames. Defaults to n_fft // 4.
        win_length (int, optional): Window length. Defaults to n_fft.
        window (Tensor, optional): 1-D tensor window. Defaults to torch.hann_window(win_length).
        center (bool, optional): Whether to pad the signal. Defaults to True.
        pad_mode (str, optional): Padding mode. Options: 'reflect', 'constant', 'replicate'. Defaults to 'reflect'.
        normalized (bool, optional): Whether to normalize the STFT. Defaults to False.
        onesided (bool, optional): Whether to return only the positive frequencies. Defaults to True.
        return_complex (bool, optional): Whether to return a complex tensor. Defaults to None
                                        (which becomes True if input is complex, False otherwise).
        
    Returns:
        Tensor: STFT of the input signal, of shape:
               (..., n_fft // 2 + 1, num_frames) if onesided=True
               (..., n_fft, num_frames) if onesided=False
    """
    print("abc stft input shape and type", input_signal.shape, input_signal.dtype)
    # Set default values
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(win_length, device=input_signal.device)
    if return_complex is None:
        return_complex = input_signal.is_complex()
    
    # Ensure window is on the same device as input_signal
    window = window.to(input_signal.device)
    
    # Adjust window size if win_length != n_fft
    if win_length != n_fft:
        window = torch.nn.functional.pad(window, [(n_fft - win_length) // 2, (n_fft - win_length + 1) // 2])
    
    # Get input dimensions
    input_dim = input_signal.dim()
    batch_dims = input_dim - 1  # number of batch dimensions
    
    # Reshape batch dimensions to deal with them uniformly
    input_shape = input_signal.shape
    signal_length = input_shape[-1]
    input_signal = input_signal.reshape(-1, signal_length)
    batch_size = input_signal.shape[0]
    
    # Pad the signal if center is True
    if center:
        padding = n_fft // 2
        if pad_mode == 'reflect':
            # Ensure the input is long enough for reflect padding
            if signal_length < padding:
                # If signal too short, use replicate instead
                pad_mode = 'replicate'
                
        # Apply padding
        input_signal = torch.nn.functional.pad(input_signal, (padding, padding), mode=pad_mode)
    
    # Calculate number of frames
    padded_length = input_signal.shape[-1]
    num_frames = max(1, (padded_length - n_fft) // hop_length + 1)
    
    # Create frame indices
    frame_indices = torch.arange(0, n_fft, device=input_signal.device).unsqueeze(0)
    hop_indices = torch.arange(0, num_frames * hop_length, hop_length, device=input_signal.device).unsqueeze(1)
    indices = frame_indices + hop_indices
    
    # Extract frames
    frames = input_signal.unsqueeze(1)[:, :, indices]
    frames = frames.reshape(batch_size, num_frames, n_fft)
    
    # Apply window
    frames = frames * window.unsqueeze(0).unsqueeze(0)
    
    # Apply normalization if requested
    if normalized:
        frames = frames / torch.sqrt(torch.sum(window.pow(2)))
    
    # Apply FFT
    stft_matrix = torch.fft.fft(frames, dim=-1)
    
    # Keep only positive frequencies if onesided is True
    if onesided:
        stft_matrix = stft_matrix[..., :(n_fft // 2) + 1]
    
    # Transpose from (batch, frames, freq) to (batch, freq, frames)
    stft_matrix = stft_matrix.transpose(1, 2)
    
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
        # Stack real and imaginary parts as separate channels
        stft_real = stft_matrix.real
        stft_imag = stft_matrix.imag
        
        # Create a new dimension for real/imag parts at position before the last 2 dimensions
        stacked_shape = list(stft_matrix.shape)
        stacked_shape.insert(-2, 2)
        
        # Reshape real and imag to prepare for stacking
        stft_real = stft_real.unsqueeze(-3)
        stft_imag = stft_imag.unsqueeze(-3)
        
        # Stack along the new dimension
        stft_matrix = torch.cat([stft_real, stft_imag], dim=-3)
    
    print("abc stft output shape and type", stft_matrix.shape, stft_matrix.dtype)
        
    return stft_matrix

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