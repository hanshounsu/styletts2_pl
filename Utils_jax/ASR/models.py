import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple, Optional, Dict, List, Sequence
from .layers import MFCC, Attention, LinearNorm, ConvNorm, ConvBlock

class ASRS2SJax(nn.Module):
    """Sequence-to-sequence ASR model component"""
    embedding_dim: int = 256
    hidden_dim: int = 512
    n_location_filters: int = 32
    location_kernel_size: int = 63
    n_token: int = 40
    
    def setup(self):
        # Text embedding layer
        self.embedding = nn.Embed(
            num_embeddings=self.n_token,
            features=self.embedding_dim
        )
        
        self.decoder_rnn_dim = self.hidden_dim
        self.project_to_n_symbols = LinearNorm(
            in_features=self.hidden_dim,
            out_features=self.n_token
        )
        
        # Attention mechanism
        self.attention_layer = Attention(
            attention_rnn_dim=self.decoder_rnn_dim,
            embedding_dim=self.hidden_dim,
            attention_dim=self.hidden_dim,
            attention_location_n_filters=self.n_location_filters,
            attention_location_kernel_size=self.location_kernel_size
        )
        
        # RNN decoder
        self.decoder_rnn = nn.LSTMCell(features=self.decoder_rnn_dim)

        self.decoder_dropout = nn.Dropout(rate=0.5)
        
        # Hidden projection layer
        self.project_to_hidden = nn.Sequential([
            LinearNorm(
                in_features=self.decoder_rnn_dim + self.hidden_dim,  # decoder hidden + context
                out_features=self.hidden_dim,
                w_init_gain='tanh'
            ),
            nn.tanh
        ])
        
        # Special tokens
        self.sos = 1  # Start of sequence token
        self.eos = 2  # End of sequence token
    
    def initialize_decoder_states(self, memory, mask):
        """Initialize decoder states for sequence generation"""
        B, L, H = memory.shape
        
        # Initialize states with zeros
        decoder_hidden = jnp.zeros((B, self.decoder_rnn_dim))
        decoder_cell = jnp.zeros((B, self.decoder_rnn_dim))
        attention_weights = jnp.zeros((B, L))
        attention_weights_cum = jnp.zeros((B, L))
        attention_context = jnp.zeros((B, H))
        
        # Process memory for attention - precompute for efficiency
        processed_memory = self.attention_layer.memory_layer(memory)
        
        return {
            'decoder_hidden': decoder_hidden,
            'decoder_cell': decoder_cell,
            'attention_weights': attention_weights,
            'attention_weights_cum': attention_weights_cum,
            'attention_context': attention_context,
            'memory': memory,
            'processed_memory': processed_memory,
            'mask': mask,
        }
    
    def decode_step(self, carry, x):
        """Single decoder step for use with jax.lax.scan"""
        # Unpack state and RNG
        states, rng = carry
        decoder_input = x
        
        # LSTM cell input: concatenate decoder input with previous attention context
        cell_input = jnp.concatenate([decoder_input, states['attention_context']], axis=-1)
        
        # Run LSTM cell - FIX: Properly unpack the return value
        lstm_state = (states['decoder_cell'], states['decoder_hidden'])
        new_carry, _ = self.decoder_rnn(lstm_state, cell_input)
        new_decoder_cell, new_decoder_hidden = new_carry  # Correctly unpack the carry tuple
    
        # Prepare attention weights for location-sensitive attention
        attention_weights_cat = jnp.concatenate([
            jnp.expand_dims(states['attention_weights'], 1),
            jnp.expand_dims(states['attention_weights_cum'], 1)
        ], axis=1)
        
        # Apply attention mechanism
        attention_context, attention_weights = self.attention_layer(
            new_decoder_hidden,
            states['memory'],
            states['processed_memory'],
            attention_weights_cat,
            states['mask']
        )
        
        # Update cumulative attention weights
        attention_weights_cum = states['attention_weights_cum'] + attention_weights
        
        # Project to hidden representation
        hidden_and_context = jnp.concatenate([new_decoder_hidden, attention_context], axis=-1)
        hidden = self.project_to_hidden(hidden_and_context)
        
        # Output projection (with dropout during training)
        dropout_rng, new_rng = jax.random.split(rng)
        hidden_dropout = self.decoder_dropout(hidden, deterministic=False, rng=dropout_rng)
        logit = self.project_to_n_symbols(hidden_dropout)
        
        # Update states
        new_states = {
            'decoder_hidden': new_decoder_hidden,
            'decoder_cell': new_decoder_cell,
            'attention_weights': attention_weights,
            'attention_weights_cum': attention_weights_cum,
            'attention_context': attention_context,
            'memory': states['memory'],
            'processed_memory': states['processed_memory'],
            'mask': states['mask'],
        }
        
        # Return updated state and outputs
        return (new_states, new_rng), (hidden, logit, attention_weights)
    
    def __call__(self, memory, memory_mask, text_input, rng, training=False):
        """Forward pass of the sequence-to-sequence model"""
        # Initialize decoder states
        states = self.initialize_decoder_states(memory, memory_mask)
        
        # Text random masking for training
        if training:
            mask_rng, rng = jax.random.split(rng)
            random_mask = jax.random.uniform(mask_rng, text_input.shape) < 0.1
            text_input = jnp.where(random_mask, 3, text_input)  # 3 is UNK token
        
        # Prepare decoder inputs
        decoder_inputs = self.embedding(text_input)
        decoder_inputs = jnp.transpose(decoder_inputs, (1, 0, 2))  # [T, B, dim]
        
        # Add start token embedding
        start_tokens = jnp.full((text_input.shape[0],), self.sos, dtype=jnp.int32)
        start_embedding = self.embedding(start_tokens)
        decoder_inputs = jnp.concatenate([jnp.expand_dims(start_embedding, 0), decoder_inputs], axis=0)
        
        # Run decoder steps using scan
        init_carry = (states, rng)
        (final_states, _), (hidden_outputs, logit_outputs, alignments) = jax.lax.scan(
            self.decode_step,
            init_carry,
            decoder_inputs
        )
        
        # Transpose outputs to batch-first
        hidden_outputs = jnp.transpose(hidden_outputs, (1, 0, 2))
        logit_outputs = jnp.transpose(logit_outputs, (1, 0, 2))
        alignments = jnp.transpose(alignments, (1, 0, 2))
        
        return hidden_outputs, logit_outputs, alignments

class ASRCNNJax(nn.Module):
    """CNN-based ASR model"""
    input_dim: int = 80
    hidden_dim: int = 256
    n_token: int = 35
    n_layers: int = 6
    token_embedding_dim: int = 256
    
    def setup(self):
        # Ensure this matches PyTorch model - with explicit 'dct_mat' param
        self.to_mfcc = MFCC(
            n_mfcc=self.input_dim//2,
            n_mels=self.input_dim
        )
        
        # Initial CNN layer - ensure structure matches PyTorch
        self.init_cnn = ConvNorm(
            in_channels=self.input_dim//2,
            out_channels=self.hidden_dim,
            kernel_size=7,
            padding=3,
            stride=2
        )
        
        # Use named 'cnns' to match PyTorch structure
        self.cnns = [
            [
                ConvBlock(
                    hidden_dim=self.hidden_dim,
                    n_conv=3,
                    dropout_p=0.2
                ),
                nn.GroupNorm(
                    num_groups=1,
                    epsilon=1e-5
                )
            ] 
            for _ in range(self.n_layers)
        ]
            
        # Keep these for JAX API compatibility
        self.conv_blocks = [block[0] for block in self.cnns]
        self.norms = [block[1] for block in self.cnns]
        
        # Projection layer
        self.projection = ConvNorm(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim // 2,
            kernel_size=1
        )
        
        # CTC Linear layers - make sure naming matches PyTorch
        self.ctc_linear = nn.Sequential([
            LinearNorm(
                in_features=self.hidden_dim // 2,
                out_features=self.hidden_dim
            ),
            nn.relu,
            LinearNorm(
                in_features=self.hidden_dim,
                out_features=self.n_token
            )
        ])
        
        
        # Sequence-to-sequence component
        self.asr_s2s = ASRS2SJax(
            embedding_dim=self.token_embedding_dim,
            hidden_dim=self.hidden_dim // 2,
            n_token=self.n_token
        )
    
    def __call__(self, x, src_key_padding_mask=None, text_input=None, rng=None, training=False):
        """Forward pass of ASR model"""
        # Convert input to MFCC features
        x = self.to_mfcc(x)
        x = self.init_cnn(x)
        
        # Apply CNN layers
        for i in range(self.n_layers):
            x = self.conv_blocks[i](x, training=training)
            x = self.norms[i](x)
        
        # Project features
        x = self.projection(x)
        x = jnp.transpose(x, (0, 2, 1))  # [B, T, C]
        
        # Apply CTC linear layers
        ctc_logit = self.ctc_linear(x)
        
        # If text input provided, run sequence-to-sequence model
        if text_input is not None:
            _, s2s_logit, s2s_attn = self.asr_s2s(
                x, src_key_padding_mask, text_input, rng, training
            )
            return ctc_logit, s2s_logit, s2s_attn
        else:
            return ctc_logit
    
    # Update get_feature method to match the new structure
    def get_feature(self, x):
        """Extract features without classification"""
        # Handle different input shapes
        if x.ndim > 2:
            x = jnp.squeeze(x, axis=1)
            
        x = self.to_mfcc(x)
        x = self.init_cnn(x)
        
        # Apply CNN layers
        for i in range(self.n_layers):
            x = self.conv_blocks[i](x, training=False)
            x = self.norms[i](x)
        
        x = self.projection(x)
        return x