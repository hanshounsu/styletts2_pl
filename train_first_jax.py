import os
import os.path as osp
import yaml
import shutil
import numpy as np
import click
import warnings
warnings.simplefilter('ignore')

# JAX imports
import jax
import jax.numpy as jnp
from jax import random, grad, jit, lax, pmap, value_and_grad
import flax
import optax
from flax.training import train_state, checkpoints
from flax.training.common_utils import shard_batch

# load packages
from munch import Munch
from tqdm import tqdm

from models_jax import *
from meldataset_jax import build_dataloader
from utils_jax import *
from losses_jax import *
from optimizers_jax import build_optimizer
import time

# Logging setup
import logging
import tensorboard as tb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))

    # Setup logging directory
    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    
    # Setup logging
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)
    
    # TensorBoard setup
    summary_writer = tb.summary.SummaryWriter(log_dir + "/tensorboard")
    
    # Training parameters
    batch_size = config.get('batch_size', 10)
    epochs = config.get('epochs_1st', 200)
    save_freq = config.get('save_freq', 2)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)
    
    # Data parameters
    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    OOD_data = data_params['OOD_data']
    max_len = config.get('max_len', 200)
    
    # Set up JAX devices and distributions
    devices = jax.local_devices()
    n_devices = len(devices)
    print(f"JAX devices: {devices}")
    
    # Initialize PRNG key
    rng = random.PRNGKey(42)
    
    # Load data
    train_list, val_list = get_data_path_list(train_path, val_path)
    
    # JAX DataLoader equivalent
    train_iter = build_dataloader(
        train_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        batch_size=batch_size * n_devices,  # global batch size
        num_workers=2,
        dataset_config={} # TODO : device 빠져있음 확인
    )
    
    val_iter = build_dataloader(
        val_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        batch_size=batch_size * n_devices,
        validation=True,
        num_workers=0,
        dataset_config={} # TODO : device 빠져있음 확인
    )

    # Load pretrained models
    rng, init_rng = random.split(rng)
    
    # Load ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner, text_aligner_params = load_ASR_models(ASR_path, ASR_config, init_rng)

    # Load F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor, pitch_extractor_params = load_F0_models(F0_path, init_rng)

    # Load BERT model
    from Utils.PLBERT.util import load_plbert
    BERT_path = config.get('PLBERT_dir', False)
    plbert, plbert_params = load_plbert(BERT_path, init_rng)

    # Initialize model
    rng, model_rng = random.split(rng)
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    
    # Build JAX model - initialize with dummy inputs
    model, variables = build_model(
        model_params, 
        text_aligner, 
        pitch_extractor, 
        plbert,
        model_rng
    )
    
    # Create optimizer
    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 1e-4)),
        "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_list) // (batch_size * n_devices),
    }
    
    optimizer = build_optimizer_jax(scheduler_params)
    
    # Initialize training state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer
    )
    
    # Load checkpoint if specified
    if config.get('pretrained_model', '') != '':
        state = load_checkpoint_jax(state, config['pretrained_model'])
        start_epoch = state.step // (len(train_list) // (batch_size * n_devices))
        iters = state.step
    else:
        start_epoch = 0
        iters = 0
    
    # Initialize loss functions
    loss_params = Munch(config['loss_params'])
    TMA_epoch = loss_params.TMA_epoch
    
    stft_loss_fn = create_stft_loss()
    generator_loss_fn = create_generator_loss()
    discriminator_loss_fn = create_discriminator_loss()
    wavlm_loss_fn = create_wavlm_loss(
        model_params.slm.model,
        sr,
        model_params.slm.sr
    )
    
    # Print model parameters
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(variables['params']))
    print(f'Total parameters: {total_params / 1e6:.2f}M')
    
    # Get model downsampling factor
    n_down = 5  # Assuming this is the default in JAX model
    
    # Define training step function
    @jit
    def train_step(state, batch, rng, epoch):
        """Train for a single step."""
        def loss_fn(params):
            # Unpack batch
            waves, texts, input_lengths, _, _, mels, mel_input_length, _ = batch
            
            # Create masks
            mask = length_to_mask_jax(mel_input_length // (2 ** n_down))
            text_mask = length_to_mask_jax(input_lengths)
            
            # Text aligner forward pass
            ppgs, s2s_pred, s2s_attn = model.apply(
                {'params': params['text_aligner']}, 
                mels, mask, texts, 
                method=model.text_aligner_forward
            )
            
            # Process attention
            s2s_attn = jnp.transpose(s2s_attn, (0, 2, 1))  # (b, L, T) -> (b, T, L)
            s2s_attn = s2s_attn[..., 1:]
            s2s_attn = jnp.transpose(s2s_attn, (0, 2, 1))
            
            # Create attention mask
            attn_mask = create_attention_mask(mask, text_mask)
            s2s_attn = jnp.where(attn_mask, 0.0, s2s_attn)
            
            # Create monotonic alignment
            mask_ST = mask_from_lens_jax(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
            s2s_attn_mono = maximum_path_jax(s2s_attn, mask_ST)
            
            # Text encoder
            t_en = model.apply(
                {'params': params['text_encoder']}, 
                texts, input_lengths, text_mask,
                method=model.text_encoder_forward
            )
            
            # Choose between monotonic and non-monotonic alignment randomly
            alignment_rng, clip_rng = random.split(rng)
            use_monotonic = random.bernoulli(alignment_rng, 0.5)
            asr = jnp.where(
                use_monotonic, 
                (t_en @ s2s_attn_mono), 
                (t_en @ s2s_attn)
            )
            
            # Get clips
            mel_len = jnp.min(mel_input_length // 2 - 1)
            mel_len = jnp.minimum(mel_len, max_len // 2)
            mel_len_st = jnp.min(mel_input_length // 2 - 1)
            
            # Cut random clips (vectorized JAX implementation)
            indices = jnp.arange(len(mel_input_length))
            mel_lengths = mel_input_length // 2
            
            # Generate random starts
            random_starts = random.randint(
                clip_rng, 
                shape=(len(mel_input_length),), 
                minval=0, 
                maxval=jnp.maximum(mel_lengths - mel_len, 1)
            )
            
            # Extract encoder outputs, ground truth mels, and waveforms
            en = jax.vmap(lambda i, start: asr[i, :, start:start+mel_len])(indices, random_starts)
            gt = jax.vmap(lambda i, start: mels[i, :, (start * 2):((start+mel_len) * 2)])(indices, random_starts)
            wav = jax.vmap(lambda i, start: waves[i][(start * 2) * 300:((start+mel_len) * 2) * 300])(indices, random_starts)
            
            # Style reference
            style_rng = random.fold_in(rng, 1)
            style_starts = random.randint(
                style_rng, 
                shape=(len(mel_input_length),), 
                minval=0, 
                maxval=jnp.maximum(mel_lengths - mel_len_st, 1)
            )
            st = jax.vmap(lambda i, start: mels[i, :, (start * 2):((start+mel_len_st) * 2)])(indices, style_starts)
            
            # Extract pitch and style
            real_norm = log_norm_jax(gt.unsqueeze(1)).squeeze(1)
            F0_real, _, _ = model.apply(
                {'params': params['pitch_extractor']}, 
                gt.unsqueeze(1),
                method=model.pitch_extractor_forward
            )
            
            style_input = st.unsqueeze(1) if multispeaker else gt.unsqueeze(1)
            s = model.apply(
                {'params': params['acoustic_style_encoder']}, 
                style_input,
                method=model.acoustic_style_encoder_forward
            )
            
            # Generate audio
            y_rec = model.apply(
                {'params': params['decoder']}, 
                en, F0_real, real_norm, s,
                method=model.decoder_forward
            )
            
            # Loss calculation
            loss_mel = stft_loss_fn(y_rec.squeeze(), wav)
            
            # TMA training after specific epoch
            if epoch >= TMA_epoch:
                # Speech-to-speech loss
                loss_s2s = jax.vmap(lambda pred, text, length: 
                    jnp.mean(optax.softmax_cross_entropy(
                        pred[:length], 
                        jax.nn.one_hot(text[:length], depth=pred.shape[-1])
                    ))
                )(s2s_pred, texts, input_lengths)
                loss_s2s = jnp.mean(loss_s2s)
                
                # Monotonic loss
                loss_mono = optax.l1_loss(s2s_attn, s2s_attn_mono).mean() * 10
                
                # Generator and SLM losses
                loss_gen_all = generator_loss_fn(wav.unsqueeze(1), y_rec)
                loss_slm = wavlm_loss_fn(wav, y_rec)
                
                # Combined loss
                loss = (
                    loss_params.lambda_mel * loss_mel + 
                    loss_params.lambda_mono * loss_mono + 
                    loss_params.lambda_s2s * loss_s2s + 
                    loss_params.lambda_gen * loss_gen_all + 
                    loss_params.lambda_slm * loss_slm
                )
                
                metrics = {
                    'loss_mel': loss_mel,
                    'loss_s2s': loss_s2s,
                    'loss_mono': loss_mono,
                    'loss_gen': loss_gen_all,
                    'loss_slm': loss_slm,
                    'loss': loss,
                }
            else:
                # Only mel loss before TMA epoch
                loss = loss_mel
                metrics = {
                    'loss_mel': loss_mel,
                    'loss': loss,
                    'loss_s2s': 0.0,
                    'loss_mono': 0.0,
                    'loss_gen': 0.0,
                    'loss_slm': 0.0,
                }
            
            return loss, metrics
        
        # Compute gradients
        grad_fn = value_and_grad(loss_fn, has_aux=True)
        (_, metrics), grads = grad_fn(state.params)
        
        # Update parameters
        new_state = state.apply_gradients(grads=grads)
        
        return new_state, metrics
    
    # Define eval step
    @jit
    def eval_step(state, batch):
        """Evaluation step."""
        waves, texts, input_lengths, _, _, mels, mel_input_length, _ = batch
        
        # Create masks
        mask = length_to_mask_jax(mel_input_length // (2 ** n_down))
        text_mask = length_to_mask_jax(input_lengths)
        
        # Forward passes
        ppgs, s2s_pred, s2s_attn = model.apply(
            {'params': state.params['text_aligner']}, 
            mels, mask, texts, 
            method=model.text_aligner_forward,
            training=False
        )
        
        # Process attention
        s2s_attn = jnp.transpose(s2s_attn, (0, 2, 1))
        s2s_attn = s2s_attn[..., 1:]
        s2s_attn = jnp.transpose(s2s_attn, (0, 2, 1))
        
        # Create attention mask
        attn_mask = create_attention_mask(mask, text_mask)
        s2s_attn = jnp.where(attn_mask, 0.0, s2s_attn)
        
        # Text encoder
        t_en = model.apply(
            {'params': state.params['text_encoder']}, 
            texts, input_lengths, text_mask,
            method=model.text_encoder_forward,
            training=False
        )
        
        # ASR product
        asr = (t_en @ s2s_attn)
        
        # Get clips
        mel_len = jnp.min(mel_input_length // 2 - 1)
        mel_len = jnp.minimum(mel_len, max_len // 2)
        
        # Validation clips
        indices = jnp.arange(len(mel_input_length))
        random_starts = jnp.zeros((len(mel_input_length),), dtype=jnp.int32)  # Fixed position for validation
        
        # Extract encoder outputs, ground truth mels, and waveforms
        en = jax.vmap(lambda i, start: asr[i, :, start:start+mel_len])(indices, random_starts)
        gt = jax.vmap(lambda i, start: mels[i, :, (start * 2):((start+mel_len) * 2)])(indices, random_starts)
        wav = jax.vmap(lambda i, start: waves[i][(start * 2) * 300:((start+mel_len) * 2) * 300])(indices, random_starts)
        
        # Extract pitch and style
        F0_real, _, F0 = model.apply(
            {'params': state.params['pitch_extractor']}, 
            gt.unsqueeze(1),
            method=model.pitch_extractor_forward,
            training=False
        )
        
        s = model.apply(
            {'params': state.params['acoustic_style_encoder']}, 
            gt.unsqueeze(1),
            method=model.acoustic_style_encoder_forward,
            training=False
        )
        
        real_norm = log_norm_jax(gt.unsqueeze(1)).squeeze(1)
        
        # Generate audio
        y_rec = model.apply(
            {'params': state.params['decoder']}, 
            en, F0_real, real_norm, s,
            method=model.decoder_forward,
            training=False
        )
        
        # Compute loss
        loss_mel = stft_loss_fn(y_rec.squeeze(), wav)
        
        return {
            'loss': loss_mel,
            'y_rec': y_rec,
            'wav': wav,
            's2s_attn': s2s_attn
        }
    
    # Parallel training on multiple devices
    p_train_step = pmap(train_step, axis_name='batch')
    p_eval_step = pmap(eval_step, axis_name='batch')
    
    # Replicate state across devices
    state = flax.jax_utils.replicate(state)
    
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        running_loss = 0.0
        
        # Training
        for i, batch in enumerate(tqdm(train_iter, desc=f"Epoch {epoch+1}/{epochs}")):
            # Prepare batch for JAX
            batch = shard_batch(batch)
            
            # Get new RNG key for each step
            rng, step_rng = random.split(rng)
            step_rngs = random.split(step_rng, n_devices)
            
            # Update model
            state, metrics = p_train_step(state, batch, step_rngs, epoch)
            
            # Update tracking variables
            running_loss += metrics['loss_mel'].mean()
            
            if (i+1) % log_interval == 0:
                # Log metrics
                avg_loss = running_loss / log_interval
                
                # Log to console and TensorBoard
                log_str = (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_list)//(batch_size*n_devices)}], '
                          f'Mel Loss: {avg_loss:.5f}, '
                          f'Gen Loss: {metrics["loss_gen"].mean():.5f}, '
                          f'Disc Loss: {0:.5f}, '
                          f'Mono Loss: {metrics["loss_mono"].mean():.5f}, '
                          f'S2S Loss: {metrics["loss_s2s"].mean():.5f}, '
                          f'SLM Loss: {metrics["loss_slm"].mean():.5f}')
                print(log_str)
                logger.info(log_str)
                
                # Add to TensorBoard
                step = epoch * (len(train_list)//(batch_size*n_devices)) + i
                summary_writer.scalar('train/mel_loss', avg_loss, step)
                summary_writer.scalar('train/gen_loss', metrics['loss_gen'].mean(), step)
                summary_writer.scalar('train/mono_loss', metrics['loss_mono'].mean(), step)
                summary_writer.scalar('train/s2s_loss', metrics['loss_s2s'].mean(), step)
                summary_writer.scalar('train/slm_loss', metrics['loss_slm'].mean(), step)
                
                running_loss = 0
                
                print('Time elapsed:', time.time() - start_time)
        
        # Validation
        val_loss = 0.0
        val_steps = 0
        
        for batch_idx, batch in enumerate(tqdm(val_iter, desc='Validation')):
            # Prepare batch for JAX
            batch = shard_batch(batch)
            
            # Run eval step
            eval_metrics = p_eval_step(state, batch)
            
            # Update validation loss
            val_loss += eval_metrics['loss'].mean()
            val_steps += 1
            
            # Save first batch outputs for visualization
            if batch_idx == 0:
                val_outputs = jax.device_get(eval_metrics)
        
        # Average validation loss
        val_loss /= val_steps
        
        # Log validation metrics
        print(f'Epoch {epoch+1}, Validation loss: {val_loss:.3f}')
        logger.info(f'Validation loss: {val_loss:.3f}')
        summary_writer.scalar('eval/mel_loss', val_loss, epoch+1)
        
        # Plot attention for visualization
        attn_fig = get_image_jax(val_outputs['s2s_attn'][0][0])
        summary_writer.image('eval/attn', attn_fig, epoch+1)
        
        # Add audio samples to TensorBoard
        for i in range(min(4, val_outputs['y_rec'].shape[0])):
            summary_writer.audio(
                f'eval/y{i}', 
                val_outputs['y_rec'][i], 
                epoch+1, 
                sample_rate=sr
            )
            
            if epoch == 0:
                summary_writer.audio(
                    f'gt/y{i}', 
                    val_outputs['wav'][i], 
                    0, 
                    sample_rate=sr
                )
        
        # Save checkpoint
        if epoch % saving_epoch == 0:
            if val_loss < best_loss:
                best_loss = val_loss
            
            # Get consolidated params from first device
            params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
            
            print('Saving checkpoint...')
            checkpoints.save_checkpoint(
                ckpt_dir=log_dir,
                target={'params': params, 'step': epoch},
                step=epoch,
                prefix='epoch_1st_',
                overwrite=True
            )
    
    # Save final model
    params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
    checkpoints.save_checkpoint(
        ckpt_dir=log_dir,
        target={'params': params, 'step': epoch},
        step=epochs,
        prefix=config.get('first_stage_path', 'first_stage'),
        overwrite=True
    )
    print('Training complete!')

if __name__ == "__main__":
    main()