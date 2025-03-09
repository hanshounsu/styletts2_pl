#coding:utf-8
import os, sys
import os.path as osp
import numpy as np
import jax
import jax.numpy as jnp
import optax
from functools import reduce
from typing import Dict, Any, List, Tuple, Callable, Optional

class MultiOptimizer:
    def __init__(self, optimizers={}, schedulers={}):
        self.optimizers = optimizers  # Dictionary of optax optimizers
        self.schedulers = schedulers  # Dictionary of scheduler functions
        self.keys = list(optimizers.keys())
        self.states = {k: opt.init(params) for k, (opt, params) in optimizers.items()}
    
    def state_dict(self):
        """Return state dictionaries for all optimizers"""
        state_dicts = [(key, self.states[key]) for key in self.keys]
        return state_dicts
    
    def load_state_dict(self, state_dict):
        """Load optimizer states from state_dict"""
        for key, val in state_dict:
            try:
                self.states[key] = val
            except:
                print(f"Unloaded {key}")
    
    def step(self, key=None, grads=None):
        """
        Update parameters with gradients
        Args:
            key: optimizer key to update (if None, update all)
            grads: dictionary of gradients, keyed by optimizer keys
        """
        if key is not None:
            return self._step(key, grads[key])
        else:
            updates = {}
            for k in self.keys:
                if k in grads:
                    updates[k] = self._step(k, grads[k])
            return updates
    
    def _step(self, key, grads):
        """
        Single optimizer step
        Args:
            key: optimizer key
            grads: gradients for this optimizer
        """
        opt, params = self.optimizers[key]
        updates, new_state = opt.update(grads, self.states[key])
        self.states[key] = new_state
        return updates
    
    def zero_grad(self, key=None):
        """No-op in JAX (for API compatibility)"""
        # In JAX, we don't need to zero gradients as they're computed from scratch each time
        pass
    
    def scheduler(self, step, key=None):
        """Update learning rate schedulers"""
        if key is not None:
            if key in self.schedulers:
                self.schedulers[key](step)
        else:
            for key in self.keys:
                if key in self.schedulers:
                    self.schedulers[key](step)

# def define_scheduler(params):
#     """
#     Create a JAX learning rate scheduler
#     Args:
#         params: dictionary of scheduler parameters
#     Returns:
#         scheduler function
#     """
#     max_lr = params.get('max_lr', 2e-4)
#     epochs = params.get('epochs', 200)
#     steps_per_epoch = params.get('steps_per_epoch', 1000)
#     pct_start = params.get('pct_start', 0.0)
    
#     # Create an OneCycleLR equivalent in optax
#     total_steps = epochs * steps_per_epoch
#     warmup_steps = int(total_steps * pct_start)
    
#     # OneCycleLR-like schedule
#     schedule_fn = optax.join_schedules([
#         optax.linear_schedule(
#             init_value=max_lr,
#             end_value=max_lr,
#             transition_steps=warmup_steps if warmup_steps > 0 else 1
#         ),
#         optax.linear_schedule(
#             init_value=max_lr,
#             end_value=max_lr,  # In original: final_div_factor=1 means final lr equals initial
#             transition_steps=total_steps - warmup_steps if total_steps - warmup_steps > 0 else 1
#         )
#     ], [warmup_steps])
    
#     return schedule_fn

def build_optimizer(scheduler_params, component_wise=False):
    """
    Build optimizer(s) for the model.
    
    Args:
        scheduler_params: Dictionary with learning rate scheduler parameters
        component_wise: If True, return a dictionary of optimizers for each component
    
    Returns:
        If component_wise=True: Dictionary of optimizers
        Otherwise: A single optimizer
    """
    # Extract parameters
    max_lr = scheduler_params.get("max_lr", 1e-4)
    pct_start = scheduler_params.get("pct_start", 0.0)
    epochs = scheduler_params.get("epochs", 100)
    steps_per_epoch = scheduler_params.get("steps_per_epoch", 100)
    
    # Calculate total steps
    total_steps = epochs * steps_per_epoch
    
    # Create learning rate schedule
    if pct_start > 0:
        # One-cycle learning rate schedule
        schedule_fn = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=max_lr,
            warmup_steps=int(total_steps * pct_start),
            decay_steps=total_steps,
            end_value=max_lr * 0.1
        )
    else:
        # Constant learning rate
        schedule_fn = max_lr
    
    # Create base optimizer with weight decay
    optimizer = optax.adamw(
        learning_rate=schedule_fn,
        b1=0.9,
        b2=0.99,
        weight_decay=0.01
    )
    
    if not component_wise:
        # Return a single optimizer
        return optimizer
    else:
        # Return a dictionary of optimizers for component-wise optimization
        # Define components that need their own optimizers
        component_list = [
            'text_encoder', 'duration_prosody_predictor', 'decoder',
            'prosodic_style_encoder', 'acoustic_style_encoder',
            'text_aligner', 'pitch_extractor', 'bert', 'bert_encoder',
            'mpd', 'msd', 'wd', 'diffusion', 'transformer'
        ]
        
        # Create a dictionary with an optimizer for each component
        optimizers = {component: optimizer for component in component_list}
        return optimizers