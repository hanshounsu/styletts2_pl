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
# from flax.training.common_utils import shard_batch

# load packages
from munch import Munch
from tqdm import tqdm

from models_jax import load_ASR_models_jax, load_F0_models_jax, build_model_jax, load_checkpoint_jax, ModelWrapper
from meldataset_jax import build_dataloader
from utils_jax import maximum_path_jax, get_data_path_list, length_to_mask, log_norm, get_image, recursive_munch, slice_audio_for_training, create_attention_mask, shard_batch
from losses_jax import MultiResolutionSTFTLoss, GeneratorLoss, DiscriminatorLoss, WavLMLoss
from optimizers_jax import build_optimizer
import time

# Logging setup
import logging
import tensorboardX as tbx  # Using TensorboardX for better JAX compatibility

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def generate_audio(params, apply_fn, s2s_attn, texts, input_lengths, mels, mel_input_length, text_mask, max_len=200):
    """Generate audio without tracking gradients for discriminator update."""
    # Run text encoder
    t_en = apply_fn(
        {'params': params['text_encoder']},
        texts, input_lengths, text_mask,
        method=lambda module, *args, **kwargs: module.text_encoder_forward(*args, **kwargs)
    )
    
    # Apply attention to get hidden features
    asr = t_en @ s2s_attn
    
    # Extract a fixed segment for consistency
    mel_len = jnp.min(mel_input_length // 2 - 1)
    mel_len = jnp.minimum(mel_len, max_len // 2)
    
    en = asr[:, :, :mel_len]
    gt = mels[:, :, :(mel_len * 2)]
    
    # Extract pitch information
    F0_real, _, _ = apply_fn(
        {'params': params['pitch_extractor']},
        jnp.expand_dims(gt, axis=1),  # Add channel dimension
        method=lambda module, *args, **kwargs: module.pitch_extractor_forward(*args, **kwargs)
    )
    
    # Extract style embedding
    s = apply_fn(
        {'params': params['acoustic_style_encoder']},
        jnp.expand_dims(gt, axis=1),  # Add channel dimension
        method=lambda module, *args, **kwargs: module.acoustic_style_encoder_forward(*args, **kwargs)
    )
    
    # Get normalization
    real_norm = log_norm(jnp.expand_dims(gt, axis=1)).squeeze(1)
    
    # Generate audio
    y_rec = apply_fn(
        {'params': params['decoder']},
        en, F0_real, real_norm, s,
        method=lambda module, *args, **kwargs: module.decoder_forward(*args, **kwargs)
    )
    
    return y_rec


def update_model_state(state, g_updates, d_updates):
    """Apply updates to model state."""
    # Handle component-wise optimizers - create structure for new params and opt_state
    new_params = {k: v for k, v in state.params.items()}
    new_opt_state = {k: v for k, v in state.opt_state.items()}
    
    # Apply generator updates
    for k, update in g_updates.items():
        # Update parameters with the update
        new_params[k] = optax.apply_updates(new_params[k], update[1])
        # Store updated optimizer state
        new_opt_state[k] = update[0]
    
    # Apply discriminator updates
    for k, update in d_updates.items():
        if k in new_params:  # Safety check
            new_params[k] = optax.apply_updates(new_params[k], update[1])
            new_opt_state[k] = update[0]
    
    # Return updated state
    return state.replace(
        params=new_params,
        opt_state=new_opt_state,
        step=state.step + 1
    )


@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
@click.option('-l', '--log_dir', default='Models/LJSpeech/config.yml', type=str)
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
    summary_writer = tbx.SummaryWriter(log_dir + "/tensorboard")
    
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
        dataset_config={}
    )
    
    val_iter = build_dataloader(
        val_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        batch_size=batch_size * n_devices,
        validation=True,
        num_workers=0,
        dataset_config={}
    )

    # Load pretrained models
    rng, init_rng = random.split(rng)
    
    # Load ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner, text_aligner_params = load_ASR_models_jax(ASR_path, ASR_config, init_rng)

    # Load F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor, pitch_extractor_params = load_F0_models_jax(F0_path, init_rng)

    # Load BERT model
    from Utils_jax.PLBERT.util import load_plbert_jax
    BERT_path = config.get('PLBERT_dir', False)
    plbert, plbert_params = load_plbert_jax(BERT_path)

    # Initialize model
    rng, model_rng = random.split(rng)
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    
    # Build JAX model
    model, variables = build_model_jax(
        model_params, 
        text_aligner, 
        text_aligner_params,
        pitch_extractor, 
        pitch_extractor_params,
        plbert,
        plbert_params,
        model_rng
    )
    
    model = ModelWrapper(model)
    
    # Create optimizer with component-wise structure
    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 1e-4)),
        "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_list) // (batch_size * n_devices),
    }
    
    # Create optimizer dictionary for each model component
    optimizers = build_optimizer(scheduler_params, component_wise=True)
    opt_states = {k: optimizers[k].init(variables['params'][k]) 
                 for k in variables['params']}
    
    # Initialize training state
    state = train_state.TrainState(
        step=0,
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizers,  # Dictionary of optimizers
        opt_state=opt_states  # Dictionary of optimizer states
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
    
    stft_loss_fn = MultiResolutionSTFTLoss()
    generator_loss_fn = GeneratorLoss(mpd=model.models.mpd, msd=model.models.msd)
    discriminator_loss_fn = DiscriminatorLoss(mpd=model.models.mpd, msd=model.models.msd)
    # Placeholder for WavLM loss - we're not using it as mentioned
    def wavlm_loss_fn(wav_clips, y_rec):
        return jnp.array(0.0)
    
    # Print model parameters
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(variables['params']))
    print(f'Total parameters: {total_params / 1e6:.2f}M')
    
    # Get model downsampling factor
    n_down = 1  # Assuming this is the default in JAX model
    
    # Define training step function
    @jit
    def train_step(state, batch, rng, epoch):
        """Train for a single step with separate generator and discriminator updates."""
        # Unpack batch
        wav, texts, input_lengths, _, _, mels, mel_input_length, _ = batch
        
        # Common preprocessing (inference only, no gradients)
        mask = length_to_mask(mel_input_length // (2 ** n_down))
        text_mask = length_to_mask(input_lengths)
        
        # Create attention mask for the model
        attn_mask = create_attention_mask(mask, text_mask)
        
        # Forward pass through text aligner
        ppgs, s2s_pred, s2s_attn = state.apply_fn(
            {'params': state.params['text_aligner']},
            mels, mask, texts,
            method=lambda module, *args, **kwargs: module.text_aligner_forward(*args, **kwargs)
        )
        
        # Process attention
        s2s_attn = jnp.transpose(s2s_attn, (0, 2, 1))
        s2s_attn = s2s_attn[..., 1:]  # Remove first frame (similar to PyTorch)
        s2s_attn = jnp.transpose(s2s_attn, (0, 2, 1))
        
        # Apply mask to attention
        s2s_attn = jnp.where(attn_mask, 0.0, s2s_attn)
        
        # Generate monotonic attention
        s2s_attn_mono = maximum_path_jax(s2s_attn, ~attn_mask)
        
        # Discriminator step (only if past TMA epoch)
        if epoch >= TMA_epoch:
            def disc_loss_fn(disc_params):
                # Forward pass through generator components to get fake audio
                # (No gradients for generator during discriminator update)
                y_rec = generate_audio(state.params, state.apply_fn, s2s_attn, texts, 
                                    input_lengths, mels, mel_input_length, text_mask, max_len)
                
                # Compute discriminator loss
                d_loss = discriminator_loss_fn(disc_params, wav, y_rec)
                return d_loss, {'d_loss': d_loss}
            
            # Get discriminator parameters only
            disc_params = {
                'mpd': state.params['mpd'],
                'msd': state.params['msd']
            }
            
            # Compute discriminator gradients
            (_, d_metrics), d_grads = value_and_grad(disc_loss_fn, has_aux=True)(disc_params)
            
            # Create discriminator update
            d_updates = {}
            for k, g in d_grads.items():
                d_updates[k] = state.tx[k].update(g, state.opt_state[k], state.params[k])
        else:
            d_metrics = {'d_loss': jnp.array(0.0)}
            d_updates = {}
        
        # Generator step
        def gen_loss_fn(gen_params):
            # Text encoder forward pass
            t_en = state.apply_fn(
                {'params': gen_params['text_encoder']},
                texts, input_lengths, text_mask,
                method=lambda module, *args, **kwargs: module.text_encoder_forward(*args, **kwargs)
            )
            
            # Choose between regular or monotonic attention (random for training)
            use_mono = jax.random.bernoulli(rng, 0.5)
            asr = jnp.where(use_mono, 
                            t_en @ s2s_attn_mono,
                            t_en @ s2s_attn)
            
            # Cut random clips for training
            clips = slice_audio_for_training(
                asr, mels, wav, mel_input_length, max_len, rng
            )
            en, gt, wav_clips = clips
            
            # Extract pitch and style
            F0_real, _, _ = state.apply_fn(
                {'params': gen_params['pitch_extractor']},
                jnp.expand_dims(gt, axis=1),
                method=lambda module, *args, **kwargs: module.pitch_extractor_forward(*args, **kwargs)
            )
            
            # Get style 
            s = state.apply_fn(
                {'params': gen_params['acoustic_style_encoder']},
                jnp.expand_dims(gt, axis=1),
                method=lambda module, *args, **kwargs: module.acoustic_style_encoder_forward(*args, **kwargs)
            )
            
            # Get normalization
            real_norm = log_norm(jnp.expand_dims(gt, axis=1)).squeeze(1)
            
            # Generate audio
            y_rec = state.apply_fn(
                {'params': gen_params['decoder']},
                en, F0_real, real_norm, s,
                method=lambda module, *args, **kwargs: module.decoder_forward(*args, **kwargs)
            )
            
            # Calculate basic mel loss
            loss_mel = stft_loss_fn(jnp.squeeze(y_rec, axis=1), wav_clips)
            
            # Additional losses after TMA epoch
            if epoch >= TMA_epoch:
                # Speech-to-speech loss
                loss_s2s = jax.vmap(lambda pred, text, length: 
                    jnp.mean(optax.softmax_cross_entropy(
                        pred[:length], 
                        jax.nn.one_hot(text[:length], pred.shape[-1])
                    ))
                )(s2s_pred, texts, input_lengths)
                loss_s2s = jnp.mean(loss_s2s)
                
                # Monotonic alignment loss
                loss_mono = jnp.mean(jnp.abs(s2s_attn - s2s_attn_mono)) * 10
                
                # Generator adversarial loss
                loss_gen_all, _ = generator_loss_fn(jnp.expand_dims(wav_clips, axis=1), y_rec)
                
                # WavLM loss - currently a placeholder returning zero
                loss_slm = wavlm_loss_fn(wav_clips, y_rec)
                
                # Combined loss
                loss = (
                    loss_params.lambda_mel * loss_mel + 
                    loss_params.lambda_mono * loss_mono + 
                    loss_params.lambda_s2s * loss_s2s + 
                    loss_params.lambda_gen * loss_gen_all
                    # loss_params.lambda_slm * loss_slm  # Commented out as not used
                )
                
                metrics = {
                    'loss_mel': loss_mel,
                    'loss_s2s': loss_s2s,
                    'loss_mono': loss_mono,
                    'loss_gen': loss_gen_all,
                    'loss_slm': jnp.array(0.0),  # Placeholder
                    'loss': loss,
                }
            else:
                # Only mel loss before TMA epoch
                loss = loss_mel
                metrics = {
                    'loss_mel': loss_mel,
                    'loss': loss,
                    'loss_s2s': jnp.array(0.0),
                    'loss_mono': jnp.array(0.0),
                    'loss_gen': jnp.array(0.0),
                    'loss_slm': jnp.array(0.0),
                }
            
            return loss, metrics
    
        # Get generator parameters (all except discriminators)
        gen_params = {k: v for k, v in state.params.items() 
                     if k not in ['mpd', 'msd']}
        
        # Compute generator gradients
        (_, g_metrics), g_grads = value_and_grad(gen_loss_fn, has_aux=True)(gen_params)
        
        # Create generator updates
        g_updates = {}
        for k in gen_params:
            if k in ['text_aligner', 'pitch_extractor'] and epoch < TMA_epoch:
                # Skip updating these components before TMA epoch
                continue
            g_updates[k] = state.tx[k].update(g_grads[k], state.opt_state[k], state.params[k])
        
        # Combine all metrics
        metrics = {**g_metrics, **d_metrics}
        
        # Apply updates to state
        new_state = update_model_state(state, g_updates, d_updates)
        
        return new_state, metrics
    
    # Define eval step
    @jit
    def eval_step(state, batch):
        """Evaluation step."""
        waves, texts, input_lengths, _, _, mels, mel_input_length, _ = batch
        
        # Create masks
        mask = length_to_mask(mel_input_length // (2 ** n_down))
        text_mask = length_to_mask(input_lengths)
        
        # Forward passes
        ppgs, s2s_pred, s2s_attn = model.apply(
            {'params': state.params['text_aligner']}, 
            mels, mask, texts, 
            method=lambda module, *args, **kwargs: module.text_aligner_forward(*args, **kwargs),
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
            method=lambda module, *args, **kwargs: module.text_encoder_forward(*args, **kwargs),
            training=False
        )
        
        # ASR product
        asr = jnp.matmul(t_en, s2s_attn)
        
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
        gt_expanded = jnp.expand_dims(gt, axis=1)  # JAX equivalent of unsqueeze
        real_norm = log_norm(gt_expanded).squeeze(1)
        
        F0_real, _, F0 = model.apply(
            {'params': state.params['pitch_extractor']}, 
            gt_expanded,
            method=lambda module, *args, **kwargs: module.pitch_extractor_forward(*args, **kwargs),
            training=False
        )
        
        s = model.apply(
            {'params': state.params['acoustic_style_encoder']}, 
            gt_expanded,
            method=lambda module, *args, **kwargs: module.acoustic_style_encoder_forward(*args, **kwargs),
            training=False
        )
        
        # Generate audio
        y_rec = model.apply(
            {'params': state.params['decoder']}, 
            en, F0_real, real_norm, s,
            method=lambda module, *args, **kwargs: module.decoder_forward(*args, **kwargs),
            training=False
        )
        
        # Compute loss
        loss_mel = stft_loss_fn(jnp.squeeze(y_rec, axis=1), wav)
        
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
        for i, batch in enumerate(tqdm(train_iter(), desc=f"Epoch {epoch+1}/{epochs}")):
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
                          f'Disc Loss: {metrics["d_loss"].mean():.5f}, '
                          f'Mono Loss: {metrics["loss_mono"].mean():.5f}, '
                          f'S2S Loss: {metrics["loss_s2s"].mean():.5f}')
                print(log_str)
                logger.info(log_str)
                
                # Add to TensorBoard
                step = epoch * (len(train_list)//(batch_size*n_devices)) + i
                summary_writer.add_scalar('train/mel_loss', float(avg_loss), step)
                summary_writer.add_scalar('train/gen_loss', float(metrics['loss_gen'].mean()), step)
                summary_writer.add_scalar('train/disc_loss', float(metrics['d_loss'].mean()), step)
                summary_writer.add_scalar('train/mono_loss', float(metrics['loss_mono'].mean()), step)
                summary_writer.add_scalar('train/s2s_loss', float(metrics['loss_s2s'].mean()), step)
                
                running_loss = 0
                
                print('Time elapsed:', time.time() - start_time)
        
        # Validation
        val_loss = 0.0
        val_steps = 0
        val_outputs = None
        
        for batch_idx, batch in enumerate(tqdm(val_iter(), desc='Validation')):
            # Prepare batch for JAX
            batch = shard_batch(batch)
            
            # Run eval step
            eval_metrics = p_eval_step(state, batch)
            
            # Update validation loss
            val_loss += jax.device_get(eval_metrics['loss'].mean())
            val_steps += 1
            
            # Save first batch outputs for visualization
            if batch_idx == 0:
                val_outputs = jax.device_get(eval_metrics)
        
        # Average validation loss
        val_loss /= val_steps
        
        # Log validation metrics
        print(f'Epoch {epoch+1}, Validation loss: {val_loss:.3f}')
        logger.info(f'Validation loss: {val_loss:.3f}')
        summary_writer.add_scalar('eval/mel_loss', float(val_loss), epoch+1)
        
        # Plot attention for visualization
        if val_outputs is not None:
            attn_fig = get_image(val_outputs['s2s_attn'][0][0])
            summary_writer.add_figure('eval/attn', attn_fig, epoch+1)
            
            # Add audio samples to TensorBoard
            for i in range(min(4, val_outputs['y_rec'].shape[0])):
                summary_writer.add_audio( f'eval/y{i}', 
                    val_outputs['y_rec'][i].squeeze(), 
                    epoch+1, sample_rate=sr)
                
                if epoch == 0:
                    summary_writer.add_audio(
                        f'gt/y{i}', 
                        val_outputs['wav'][i], 
                        0, sample_rate=sr)
        
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
        target={'params': params, 'step': epochs},
        step=epochs,
        prefix=config.get('first_stage_path', 'first_stage'),
        overwrite=True
    )
    print('Training complete!')


if __name__ == "__main__":
    main()