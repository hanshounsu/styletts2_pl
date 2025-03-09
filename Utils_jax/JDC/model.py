"""
Implementation of model from:
Kum et al. - "Joint Detection and Classification of Singing Voice Melody Using
Convolutional Recurrent Neural Networks" (2019)
Link: https://www.semanticscholar.org/paper/Joint-Detection-and-Classification-of-Singing-Voice-Kum-Nam/60a2ad4c7db43bace75805054603747fcd062c0d
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Any, Optional

class ResBlockJax(nn.Module):
    in_channels: int
    out_channels: int
    leaky_relu_slope: float = 0.01
    
    @nn.compact
    def __call__(self, x, training=True):
        downsample = self.in_channels != self.out_channels
        
        # BN / LReLU / MaxPool layer before the conv layer
        pre_conv = nn.BatchNorm(
            use_running_average=not training, 
            momentum=0.9, 
            epsilon=1e-5
        )(x)
        pre_conv = nn.leaky_relu(pre_conv, negative_slope=self.leaky_relu_slope)
        pre_conv = nn.max_pool(pre_conv, window_shape=(1, 2), strides=(1, 2), padding='VALID')
        
        # Conv layers
        conv_out = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            use_bias=False
        )(pre_conv)
        
        conv_out = nn.BatchNorm(
            use_running_average=not training,
            momentum=0.9,
            epsilon=1e-5
        )(conv_out)
        
        conv_out = nn.leaky_relu(conv_out, negative_slope=self.leaky_relu_slope)
        
        conv_out = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            use_bias=False
        )(conv_out)
        
        # 1x1 convolution for dimension matching if needed
        if downsample:
            skip = nn.Conv(
                features=self.out_channels,
                kernel_size=(1, 1),
                use_bias=False
            )(pre_conv)
        else:
            skip = pre_conv
        
        return conv_out + skip

class JDCNetJax(nn.Module):
    """
    Joint Detection and Classification Network model for singing voice melody.
    """
    num_class: int = 722
    seq_len: int = 31 
    leaky_relu_slope: float = 0.01
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # x shape is (batch, 1, seq_len, 80)
        seq_len = x.shape[-2]
        
        x = jnp.transpose(x, (0, 2, 3, 1))
        # Conv block
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            use_bias=False
        )(x)
        
        x = nn.BatchNorm(
            use_running_average=not training,
            momentum=0.9,
            epsilon=1e-5
        )(x)
        
        x = nn.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            use_bias=False
        )(x)
        
        convblock_out = x
        
        # Res blocks
        resblock1_out = ResBlockJax(in_channels=64, out_channels=128, 
                                  leaky_relu_slope=self.leaky_relu_slope)(convblock_out, training)
        
        resblock2_out = ResBlockJax(in_channels=128, out_channels=192, 
                                  leaky_relu_slope=self.leaky_relu_slope)(resblock1_out, training)
        
        resblock3_out = ResBlockJax(in_channels=192, out_channels=256, 
                                  leaky_relu_slope=self.leaky_relu_slope)(resblock2_out, training)
        
        # Pool block - part 1 (BatchNorm + LeakyReLU)
        x = nn.BatchNorm(
            use_running_average=not training,
            momentum=0.9,
            epsilon=1e-5
        )(resblock3_out)
        
        x = nn.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        
        # Store GAN feature - no need to transpose since JAX already has channels last
        gan_feature = x
        
        # Pool block - part 2 (MaxPool + Dropout)
        x = nn.max_pool(x, window_shape=(1, 4), strides=(1, 4), padding='VALID')
        
        x = nn.Dropout(rate=0.2)(x, deterministic=not training)
        
        poolblock_out = x
        
        
        # Prepare for classification
        # In PyTorch: (b, 256, 31, 2) => (b, 31, 256, 2) => (b, 31, 512)
        # In JAX: (b, 31, 2, 256) -> reshape to (b, 31, 512)
        # Flatten the last two dimensions
        b, s, f, c = poolblock_out.shape
        classifier_out = jnp.reshape(poolblock_out, (b, s, f * c))
        print("Classifier input shape: ", classifier_out.shape)
        
        # BiLSTM classifier - using forward and backward LSTMs
        # Forward LSTM
        forward_lstm = nn.scan(nn.LSTMCell, 
                            variable_broadcast="params", 
                            split_rngs={"params": False},
                            in_axes=1, out_axes=1)(features=256)
        # Backward LSTM 
        backward_lstm = nn.scan(nn.LSTMCell, 
                            variable_broadcast="params", 
                            split_rngs={"params": False},
                            in_axes=1, out_axes=1, reverse=True)(features=256)

        # Use the dropout RNG which is typically available during both training and inference
        dropout_key = self.make_rng('dropout')
        lstm_key1, lstm_key2 = jax.random.split(dropout_key)
        
        # Initialize states
        batch_size = classifier_out.shape[0]
        input_dim = classifier_out.shape[-1]
        forward_carry = forward_lstm.initialize_carry(lstm_key1, (batch_size, input_dim))
        backward_carry = backward_lstm.initialize_carry(lstm_key2, (batch_size, input_dim))

        # Run LSTMs
        _, forward_outputs = forward_lstm(forward_carry, classifier_out)
        _, backward_outputs = backward_lstm(backward_carry, classifier_out)
        print("Forward LSTM output shape: ", forward_outputs.shape)
        print("Backward LSTM output shape: ", backward_outputs.shape)

        # Concatenate outputs
        classifier_out = jnp.concatenate([forward_outputs, backward_outputs], axis=-1)
        
        # Reshape for classification head
        classifier_out = jnp.reshape(classifier_out, (-1, 512))
        classifier_out = nn.Dense(features=self.num_class)(classifier_out)
        classifier_out = jnp.reshape(classifier_out, (-1, seq_len, self.num_class))
        
        # Final output
        classifier_out = jnp.abs(classifier_out)
        
        return classifier_out, gan_feature, poolblock_out
    
    def get_feature_GAN(self, x, training=False):
        seq_len = x.shape[-2]
        
        # Input shape: (batch, seq_len, 80)
        # Reshape to (batch, 1, seq_len, 80)
        x = jnp.expand_dims(x, axis=1)
        
        # Conv block
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            use_bias=False
        )(x)
        
        x = nn.BatchNorm(
            use_running_average=not training,
            momentum=0.9,
            epsilon=1e-5
        )(x)
        
        x = nn.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            use_bias=False
        )(x)
        
        convblock_out = x
        
        # Res blocks
        resblock1_out = ResBlockJax(in_channels=64, out_channels=128, 
                                  leaky_relu_slope=self.leaky_relu_slope)(convblock_out, training)
        
        resblock2_out = ResBlockJax(in_channels=128, out_channels=192, 
                                  leaky_relu_slope=self.leaky_relu_slope)(resblock1_out, training)
        
        resblock3_out = ResBlockJax(in_channels=192, out_channels=256, 
                                  leaky_relu_slope=self.leaky_relu_slope)(resblock2_out, training)
        
        # Pool block part 1 only (BatchNorm + LeakyReLU)
        poolblock_out = nn.BatchNorm(
            use_running_average=not training,
            momentum=0.9,
            epsilon=1e-5
        )(resblock3_out)
        
        poolblock_out = nn.leaky_relu(poolblock_out, negative_slope=self.leaky_relu_slope)
        
        # Return transpose to match PyTorch
        return jnp.transpose(poolblock_out, (0, 2, 1, 3))
        
    def get_feature(self, x, training=False):
        seq_len = x.shape[-2]
        
        # Input shape: (batch, seq_len, 80)
        # Reshape to (batch, 1, seq_len, 80)
        x = jnp.expand_dims(x, axis=1)
        
        # Conv block
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            use_bias=False
        )(x)
        
        x = nn.BatchNorm(
            use_running_average=not training,
            momentum=0.9,
            epsilon=1e-5
        )(x)
        
        x = nn.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            use_bias=False
        )(x)
        
        convblock_out = x
        
        # Res blocks
        resblock1_out = ResBlockJax(in_channels=64, out_channels=128, 
                                  leaky_relu_slope=self.leaky_relu_slope)(convblock_out, training)
        
        resblock2_out = ResBlockJax(in_channels=128, out_channels=192, 
                                  leaky_relu_slope=self.leaky_relu_slope)(resblock1_out, training)
        
        resblock3_out = ResBlockJax(in_channels=192, out_channels=256, 
                                  leaky_relu_slope=self.leaky_relu_slope)(resblock2_out, training)
        
        # Pool block - first two parts (BatchNorm + LeakyReLU + MaxPool)
        poolblock_out = nn.BatchNorm(
            use_running_average=not training,
            momentum=0.9,
            epsilon=1e-5
        )(resblock3_out)
        
        poolblock_out = nn.leaky_relu(poolblock_out, negative_slope=self.leaky_relu_slope)
        
        # MaxPool to get the feature
        return nn.max_pool(poolblock_out, window_shape=(1, 4), strides=(1, 4), padding='VALID')