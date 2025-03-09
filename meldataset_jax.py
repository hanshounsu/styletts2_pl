#coding: utf-8
import os
import os.path as osp
import time
import random
import numpy as np
import soundfile as sf
import librosa
import jax
import jax.numpy as jnp
from flax import linen as nn
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Character dictionaries
_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»"" '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len(symbols)):
    dicts[symbols[i]] = i

class TextCleaner:
    """Converts text to sequence of indices"""
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
        
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(f"Character not found in dictionary: {text}")
        return indexes

# Set random seeds for reproducibility
np.random.seed(1)
random.seed(1)

# Spectrogram parameters
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
}

# Constants for normalization
MEAN, STD = -4, 4

def create_mel_filter_bank(sr, n_fft, n_mels):
    """Create a JAX-based mel filter bank"""
    # Mel points
    lower_edge_hertz = 0.0
    upper_edge_hertz = sr / 2.0
    
    # Generate mel points
    mel_points = jnp.linspace(
        librosa.hz_to_mel(lower_edge_hertz),
        librosa.hz_to_mel(upper_edge_hertz),
        n_mels + 2
    )
    hz_points = librosa.mel_to_hz(mel_points)
    
    # Convert Hz to FFT bins
    bin_indices = jnp.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    # Create filter bank
    filter_bank = jnp.zeros((n_mels, int(n_fft // 2 + 1)))
    
    # Fill in the filter bank matrix
    def create_filter(i, fb):
        fbank = fb.at[i].set(
            jnp.linspace(0, 1, bin_indices[i+1] - bin_indices[i])
        )
        fbank = fb.at[i, bin_indices[i]:bin_indices[i+1]].set(
            jnp.linspace(0, 1, bin_indices[i+1] - bin_indices[i])
        )
        fbank = fb.at[i, bin_indices[i+1]:bin_indices[i+2]].set(
            jnp.linspace(1, 0, bin_indices[i+2] - bin_indices[i+1])
        )
        return fbank
    
    for i in range(n_mels):
        filter_bank = create_filter(i, filter_bank)
        
    return filter_bank

def mel_spectrogram(audio, n_fft=2048, hop_length=300, win_length=1200, n_mels=80):
    """Convert waveform to mel spectrogram using JAX"""
    # Create window
    window = jnp.hanning(win_length)
    
    # STFT
    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann'
    )
    
    # Convert to magnitude
    magnitudes = jnp.abs(stft)
    
    # Create mel filter bank
    mel_filter = create_mel_filter_bank(sr=24000, n_fft=n_fft, n_mels=n_mels)
    
    # Apply mel filter
    mel_spec = jnp.matmul(mel_filter, magnitudes)
    
    return mel_spec

def preprocess(wave):
    """Preprocess audio waveform to mel spectrogram"""
    wave_array = jnp.array(wave, dtype=jnp.float32)
    mel_tensor = mel_spectrogram(
        wave_array,
        n_fft=SPECT_PARAMS["n_fft"], 
        win_length=SPECT_PARAMS["win_length"], 
        hop_length=SPECT_PARAMS["hop_length"], 
        n_mels=MEL_PARAMS["n_mels"]
    )
    mel_tensor = (jnp.log(1e-5 + mel_tensor) - MEAN) / STD
    return mel_tensor

class FilePathDataset:
    """Dataset for loading and processing audio files with text"""
    
    def __init__(self,
                 data_list,
                 root_path,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 OOD_data="Data/OOD_texts.txt",
                 min_length=50):
        
        _data_list = [l.strip().split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, '0') for data in _data_list]
        if not validation:
            # For debugging or development, use just a few examples
            self.data_list = self.data_list[-10:]
            
        self.text_cleaner = TextCleaner()
        self.sr = sr
        self.df = pd.DataFrame(self.data_list)
        
        self.mean, self.std = MEAN, STD
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        
        self.min_length = min_length
        with open(OOD_data, 'r', encoding='utf-8') as f:
            tl = f.readlines()
        idx = 1 if '.wav' in tl[0].split('|')[0] else 0
        self.ptexts = [t.split('|')[idx] for t in tl]
        
        self.root_path = root_path
        self.validation = validation

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        path = data[0]
        
        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave)
        
        acoustic_feature = mel_tensor
        length_feature = acoustic_feature.shape[1]
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]
        
        # Get reference sample for the same speaker
        ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
        ref_mel_tensor, ref_label = self._load_data(ref_data[:3])
        
        # Get OOD text for training
        ps = ""
        while len(ps) < self.min_length:
            rand_idx = np.random.randint(0, len(self.ptexts) - 1)
            ps = self.ptexts[rand_idx]
            
            text = self.text_cleaner(ps)
            text.insert(0, 0)
            text.append(0)
            ref_text = jnp.array(text, dtype=jnp.int32)
        
        # Convert PyTorch tensors to JAX arrays
        text_array = jnp.array(text_tensor, dtype=jnp.int32)
        
        return speaker_id, acoustic_feature, text_array, ref_text, ref_mel_tensor, ref_label, path, wave

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        try:
            wave, sr = sf.read(osp.join(self.root_path, wave_path))
        except:
            raise ValueError(f"Failed to load {osp.join(self.root_path, wave_path)}")
            
        if len(wave.shape) > 1 and wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
            
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
            print(f"Resampled {wave_path} from {sr} to 24000Hz")
            
        # Add padding
        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
        
        # Process text
        text = self.text_cleaner(text)
        text.insert(0, 0)
        text.append(0)
        
        text = jnp.array(text, dtype=jnp.int32)
        
        return wave, text, speaker_id

    def _load_data(self, data):
        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave)
        
        mel_length = mel_tensor.shape[1]
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]
            
        return mel_tensor, speaker_id

class Collater:
    """Collate function for batching data"""
    
    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave
        
    def __call__(self, batch):
        batch_size = len(batch)
        
        # Sort by mel length for more efficient batch processing
        batch = sorted(batch, key=lambda x: x[1].shape[1], reverse=True)
        
        nmels = batch[0][1].shape[0]
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rtext_length = max([b[3].shape[0] for b in batch])
        
        # Create arrays for the batch
        labels = jnp.zeros((batch_size,), dtype=jnp.int32)
        mels = jnp.zeros((batch_size, nmels, max_mel_length), dtype=jnp.float32)
        texts = jnp.zeros((batch_size, max_text_length), dtype=jnp.int32)
        ref_texts = jnp.zeros((batch_size, max_rtext_length), dtype=jnp.int32)
        
        input_lengths = jnp.zeros(batch_size, dtype=jnp.int32)
        ref_lengths = jnp.zeros(batch_size, dtype=jnp.int32)
        output_lengths = jnp.zeros(batch_size, dtype=jnp.int32)
        ref_mels = jnp.zeros((batch_size, nmels, self.max_mel_length), dtype=jnp.float32)
        ref_labels = jnp.zeros((batch_size,), dtype=jnp.int32)
        
        paths = ['' for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]
        
        # Fill in the batch arrays
        for bid, (label, mel, text, ref_text, ref_mel, ref_label, path, wave) in enumerate(batch):
            mel_size = mel.shape[1]
            text_size = text.shape[0]
            rtext_size = ref_text.shape[0]
            
            # Update arrays with data from this sample
            labels = labels.at[bid].set(label)
            mels = mels.at[bid, :, :mel_size].set(mel)
            texts = texts.at[bid, :text_size].set(text)
            ref_texts = ref_texts.at[bid, :rtext_size].set(ref_text)
            
            input_lengths = input_lengths.at[bid].set(text_size)
            ref_lengths = ref_lengths.at[bid].set(rtext_size)
            output_lengths = output_lengths.at[bid].set(mel_size)
            
            paths[bid] = path
            
            ref_mel_size = ref_mel.shape[1]
            ref_mels = ref_mels.at[bid, :, :ref_mel_size].set(ref_mel)
            ref_labels = ref_labels.at[bid].set(ref_label)
            waves[bid] = wave
            
        return waves, texts, input_lengths, ref_texts, ref_lengths, mels, output_lengths, ref_mels

def process_batch(batch_data, device=None):
    """Process a batch for JAX/Flax training"""
    waves, texts, input_lengths, ref_texts, ref_lengths, mels, output_lengths, ref_mels = batch_data
    
    # Convert any remaining numpy arrays to JAX arrays
    texts = jnp.array(texts)
    input_lengths = jnp.array(input_lengths)
    ref_texts = jnp.array(ref_texts)
    ref_lengths = jnp.array(ref_lengths)
    mels = jnp.array(mels)
    output_lengths = jnp.array(output_lengths)
    ref_mels = jnp.array(ref_mels)
    
    return {
        'waves': waves,
        'texts': texts,
        'input_lengths': input_lengths,
        'ref_texts': ref_texts,
        'ref_lengths': ref_lengths,
        'mels': mels,
        'output_lengths': output_lengths,
        'ref_mels': ref_mels
    }

def jax_dataloader(dataset, batch_size, shuffle=True, drop_last=True):
    """JAX compatible data loader function"""
    collater = Collater()
    dataset_size = len(dataset)
    
    if drop_last:
        # Drop last incomplete batch if needed
        valid_idx = range(0, dataset_size - (dataset_size % batch_size), batch_size)
    else:
        # Include all batches
        valid_idx = range(0, dataset_size, batch_size)
    
    if shuffle:
        # Create shuffled indices
        indices = np.random.permutation(dataset_size)
    else:
        indices = np.arange(dataset_size)
    
    for start_idx in valid_idx:
        end_idx = min(start_idx + batch_size, dataset_size)
        batch_indices = indices[start_idx:end_idx]
        
        # Get items for this batch
        batch = [dataset[int(i)] for i in batch_indices]
        
        # Collate the batch
        yield collater(batch)

def build_dataloader(path_list,
                     root_path,
                     validation=False,
                     OOD_data="Data/OOD_texts.txt",
                     min_length=50,
                     batch_size=4,
                     num_workers=1,
                     device=None,
                     collate_config={},
                     dataset_config={}):
    """Build a JAX-compatible dataloader"""
    
    # Create dataset
    dataset = FilePathDataset(path_list,
                              root_path,
                              OOD_data=OOD_data,
                              min_length=min_length,
                              validation=validation,
                              **dataset_config)
    
    # Return JAX dataloader generator function
    return lambda: jax_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=(not validation),
        drop_last=(not validation)
    )