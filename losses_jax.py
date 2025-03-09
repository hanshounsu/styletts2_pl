import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Any, Dict, Optional
from flax.core import freeze, unfreeze

# We'll need a JAX-compatible mel spectrogram implementation
class MelSpectrogramTransform:
    """JAX implementation of MelSpectrogram transform"""
    
    def __init__(self, sample_rate=24000, n_fft=1024, win_length=600, hop_length=120):
        """Initialize mel spectrogram transform."""
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        
        # Precompute mel filterbank
        import librosa
        self.mel_basis = jnp.array(
            librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=80)
        )
        
        # Hann window
        self.window = jnp.hanning(win_length)
    
    def __call__(self, audio):
        """Convert audio to mel spectrogram.
        
        Args:
            audio: (batch_size, samples) audio signal
            
        Returns:
            (batch_size, n_mels, time) mel spectrogram
        """
        # Pad audio if needed
        pad_len = self.n_fft - self.win_length
        audio = jnp.pad(audio, ((0, 0), (pad_len//2, pad_len//2)))
        
        # Compute STFT
        stft = jax.vmap(self._stft)(audio)
        
        # Convert to mel scale
        mel = jnp.einsum('bft,mf->bmt', jnp.abs(stft)**2, self.mel_basis)
        
        return mel
    
    def _stft(self, audio):
        """Compute STFT for a single audio signal."""
        # Frame the signal
        frames = librosa.util.frame(audio, frame_length=self.n_fft, hop_length=self.hop_length)
        
        # Apply window
        windowed = frames * self.window
        
        # Compute FFT
        stft = jnp.fft.rfft(windowed, axis=0)
        
        return stft


class SpectralConvergenceLoss:
    """Spectral convergence loss function."""
    
    def __call__(self, x_mag, y_mag):
        """Calculate spectral convergence loss.
        
        Args:
            x_mag: Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag: Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
            
        Returns:
            Spectral convergence loss value.
        """
        return jnp.linalg.norm(y_mag - x_mag, ord=1) / jnp.linalg.norm(y_mag, ord=1)


class STFTLoss:
    """STFT loss function."""
    
    def __init__(self, fft_size=1024, shift_size=120, win_length=600):
        """Initialize STFT loss."""
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.to_mel = MelSpectrogramTransform(
            sample_rate=24000, 
            n_fft=fft_size, 
            win_length=win_length, 
            hop_length=shift_size
        )
        self.spectral_convergence_loss = SpectralConvergenceLoss()
    
    def __call__(self, x, y):
        """Calculate STFT loss.
        
        Args:
            x: Predicted signal (B, T).
            y: Groundtruth signal (B, T).
            
        Returns:
            Spectral convergence loss value.
        """
        x_mag = self.to_mel(x)
        mean, std = -4, 4
        x_mag = (jnp.log(1e-5 + x_mag) - mean) / std
        
        y_mag = self.to_mel(y)
        y_mag = (jnp.log(1e-5 + y_mag) - mean) / std
        
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        return sc_loss


class MultiResolutionSTFTLoss:
    """Multi-resolution STFT loss function."""
    
    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
    ):
        """Initialize multi-resolution STFT loss."""
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = [
            STFTLoss(fs, ss, wl)
            for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths)
        ]
    
    def __call__(self, x, y):
        """Calculate multi-resolution STFT loss.
        
        Args:
            x: Predicted signal (B, T).
            y: Groundtruth signal (B, T).
            
        Returns:
            Multi-resolution spectral convergence loss value.
        """
        sc_loss = 0.0
        for f in self.stft_losses:
            sc_loss += f(x, y)
        sc_loss /= len(self.stft_losses)
        
        return sc_loss


def feature_loss(fmap_r, fmap_g):
    """Calculate feature loss between real and generated feature maps."""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += jnp.mean(jnp.abs(rl - gl))
    
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """Calculate discriminator loss."""
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = jnp.mean((1 - dr) ** 2)
        g_loss = jnp.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(float(r_loss))
        g_losses.append(float(g_loss))
    
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """Calculate generator loss."""
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = jnp.mean((1 - dg) ** 2)
        gen_losses.append(float(l))
        loss += l
    
    return loss, gen_losses


def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    """Calculate discriminator Two-Phase Relative Least Squares (TPRLS) loss."""
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = jnp.median(dr - dg)
        # Need to handle the conditional selection differently in JAX
        # Create mask for dr < dg + m_DG
        mask = dr < dg + m_DG
        diff = (dr - dg) - m_DG
        # Use where to conditionally compute the loss
        squared_diff = jnp.where(mask, diff**2, 0.0)
        # Use sum and count non-zeros for mean to avoid div by zero
        count = jnp.sum(mask)
        L_rel = jnp.sum(squared_diff) / jnp.maximum(count, 1)
        
        loss += tau - jax.nn.relu(tau - L_rel)
    
    return loss


def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    """Calculate generator Two-Phase Relative Least Squares (TPRLS) loss."""
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = jnp.median(dr - dg)
        # Need to handle the conditional selection differently in JAX
        # Create mask for dr < dg + m_DG
        mask = dr < dg + m_DG
        diff = (dr - dg) - m_DG
        # Use where to conditionally compute the loss
        squared_diff = jnp.where(mask, diff**2, 0.0)
        # Use sum and count non-zeros for mean to avoid div by zero
        count = jnp.sum(mask)
        L_rel = jnp.sum(squared_diff) / jnp.maximum(count, 1)
        
        loss += tau - jax.nn.relu(tau - L_rel)
    
    return loss


class GeneratorLoss:
    """Generator loss function."""
    
    def __init__(self, mpd, msd):
        """Initialize generator loss."""
        self.mpd = mpd
        self.msd = msd
    
    def __call__(self, y, y_hat, mpd_params, msd_params):
        """Calculate generator loss.
        
        Args:
            y: Real audio
            y_hat: Generated audio
            mpd_params: Parameters for MPD
            msd_params: Parameters for MSD
            
        Returns:
            Total generator loss
        """
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd.apply(
            {"params": mpd_params}, y, y_hat, method=self.mpd.forward_features
        )
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd.apply(
            {"params": msd_params}, y, y_hat, method=self.msd.forward_features
        )
        
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, _ = generator_loss(y_df_hat_g)
        loss_gen_s, _ = generator_loss(y_ds_hat_g)
        
        loss_rel = generator_TPRLS_loss(y_df_hat_r, y_df_hat_g) + generator_TPRLS_loss(y_ds_hat_r, y_ds_hat_g)
        
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_rel
        
        return loss_gen_all


class DiscriminatorLoss:
    """Discriminator loss function."""
    
    def __init__(self, mpd, msd):
        """Initialize discriminator loss."""
        self.mpd = mpd
        self.msd = msd
    
    def __call__(self, y, y_hat, mpd_params, msd_params):
        """Calculate discriminator loss.
        
        Args:
            y: Real audio
            y_hat: Generated audio
            mpd_params: Parameters for MPD
            msd_params: Parameters for MSD
            
        Returns:
            Total discriminator loss
        """
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd.apply(
            {"params": mpd_params}, y, y_hat, method=self.mpd.forward_features
        )
        loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
        
        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd.apply(
            {"params": msd_params}, y, y_hat, method=self.msd.forward_features
        )
        loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        
        loss_rel = discriminator_TPRLS_loss(y_df_hat_r, y_df_hat_g) + discriminator_TPRLS_loss(y_ds_hat_r, y_ds_hat_g)
        
        d_loss = loss_disc_s + loss_disc_f + loss_rel
        
        return d_loss


class WavLMLoss:
    """WavLM-based loss function."""
    
    def __init__(self, wavlm_model, wd, model_sr=24000, slm_sr=16000):
        """Initialize WavLM loss."""
        self.wavlm = wavlm_model  # This should be a JAX version of WavLM
        self.wd = wd  # Discriminator
        self.model_sr = model_sr
        self.slm_sr = slm_sr
        
        # We would need a JAX-compatible resampler
        try:
            import librosa
            self.resample = lambda audio: librosa.resample(
                audio.astype(float), 
                orig_sr=model_sr,
                target_sr=slm_sr
            )
        except ImportError:
            raise ImportError("librosa is required for WavLMLoss")
    
    def __call__(self, wav, y_rec, wavlm_params, wd_params):
        """Calculate WavLM feature loss."""
        # Resample to WavLM input rate
        wav_16 = jax.vmap(self.resample)(wav)
        
        # Get WavLM embeddings for original audio (without gradient)
        wav_embeddings = self.wavlm.apply(
            {"params": wavlm_params}, 
            input_values=wav_16, 
            output_hidden_states=True,
            method=self.wavlm.get_hidden_states
        )
        
        # Resample generated audio
        y_rec_16 = jax.vmap(self.resample)(jnp.squeeze(y_rec, axis=1))
        
        # Get WavLM embeddings for generated audio (with gradient)
        y_rec_embeddings = self.wavlm.apply(
            {"params": wavlm_params}, 
            input_values=y_rec_16, 
            output_hidden_states=True,
            method=self.wavlm.get_hidden_states
        )
        
        # Calculate feature loss across all layers
        floss = 0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += jnp.mean(jnp.abs(er - eg))
        
        return floss
    
    def generator(self, y_rec, wavlm_params, wd_params):
        """Calculate generator loss using WavLM features."""
        y_rec_16 = jax.vmap(self.resample)(y_rec)
        
        # Get WavLM embeddings
        y_rec_embeddings = self.wavlm.apply(
            {"params": wavlm_params}, 
            input_values=y_rec_16, 
            output_hidden_states=True,
            method=self.wavlm.get_hidden_states
        )
        
        # Stack and reshape embeddings
        y_rec_embeddings = jnp.stack(y_rec_embeddings, axis=1)
        y_rec_embeddings = jnp.transpose(y_rec_embeddings, (0, 2, 1, 3))
        y_rec_embeddings = jnp.reshape(y_rec_embeddings, 
                                      (y_rec_embeddings.shape[0], -1, y_rec_embeddings.shape[-1]))
        
        # Apply discriminator
        y_df_hat_g = self.wd.apply({"params": wd_params}, y_rec_embeddings)
        
        # Calculate generator loss
        loss_gen = jnp.mean((1 - y_df_hat_g) ** 2)
        
        return loss_gen
    
    def discriminator(self, wav, y_rec, wavlm_params, wd_params):
        """Calculate discriminator loss using WavLM features."""
        # Process real audio (without gradient)
        wav_16 = jax.vmap(self.resample)(wav)
        wav_embeddings = self.wavlm.apply(
            {"params": wavlm_params}, 
            input_values=wav_16, 
            output_hidden_states=True,
            method=self.wavlm.get_hidden_states
        )
        
        # Process generated audio (without gradient)
        y_rec_16 = jax.vmap(self.resample)(y_rec)
        y_rec_embeddings = self.wavlm.apply(
            {"params": wavlm_params}, 
            input_values=y_rec_16, 
            output_hidden_states=True,
            method=self.wavlm.get_hidden_states
        )
        
        # Stack and reshape embeddings
        y_embeddings = jnp.stack(wav_embeddings, axis=1)
        y_embeddings = jnp.transpose(y_embeddings, (0, 2, 1, 3))
        y_embeddings = jnp.reshape(y_embeddings, 
                                 (y_embeddings.shape[0], -1, y_embeddings.shape[-1]))
        
        y_rec_embeddings = jnp.stack(y_rec_embeddings, axis=1)
        y_rec_embeddings = jnp.transpose(y_rec_embeddings, (0, 2, 1, 3))
        y_rec_embeddings = jnp.reshape(y_rec_embeddings, 
                                     (y_rec_embeddings.shape[0], -1, y_rec_embeddings.shape[-1]))
        
        # Apply discriminator
        y_d_rs = self.wd.apply({"params": wd_params}, y_embeddings)
        y_d_gs = self.wd.apply({"params": wd_params}, y_rec_embeddings)
        
        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs
        
        # Calculate discriminator loss
        r_loss = jnp.mean((1 - y_df_hat_r) ** 2)
        g_loss = jnp.mean(y_df_hat_g ** 2)
        
        loss_disc_f = r_loss + g_loss
        
        return loss_disc_f
    
    def discriminator_forward(self, wav, wavlm_params, wd_params):
        """Forward pass through discriminator using WavLM features."""
        wav_16 = jax.vmap(self.resample)(wav)
        wav_embeddings = self.wavlm.apply(
            {"params": wavlm_params}, 
            input_values=wav_16, 
            output_hidden_states=True,
            method=self.wavlm.get_hidden_states
        )
        
        y_embeddings = jnp.stack(wav_embeddings, axis=1)
        y_embeddings = jnp.transpose(y_embeddings, (0, 2, 1, 3))
        y_embeddings = jnp.reshape(y_embeddings, 
                                 (y_embeddings.shape[0], -1, y_embeddings.shape[-1]))
        
        y_d_rs = self.wd.apply({"params": wd_params}, y_embeddings)
        
        return y_d_rs