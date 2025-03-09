import os
import yaml
import jax
import flax
import torch  # Still needed for loading PyTorch weights
import numpy as np
from transformers import AlbertConfig, FlaxAlbertModel
from flax.training import checkpoints
import jax.numpy as jnp
from flax.core import unfreeze, freeze

from weight_transfer_jax import transfer_plbert_weights_dict

class CustomFlaxAlbert(FlaxAlbertModel):
    def __call__(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().__call__(*args, **kwargs)
        
        # Only return the last_hidden_state
        return outputs.last_hidden_state


def load_plbert_jax(log_dir):
    config_path = os.path.join(log_dir, "config.yml")
    plbert_config = yaml.safe_load(open(config_path))
    
    albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    bert = CustomFlaxAlbert(config=albert_base_configuration)

    # Find the latest checkpoint
    files = os.listdir(log_dir)
    ckpts = [f for f in files if f.startswith("step_")]
    iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
    iters = sorted(iters)[-1]

    # Load PyTorch checkpoint
    checkpoint = torch.load(log_dir + "/step_" + str(iters) + ".t7", map_location='cpu')
    state_dict = checkpoint['net']
    
    # Process PyTorch state dict
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        if name.startswith('encoder.'):
            name = name[8:]  # remove `encoder.`
            new_state_dict[name] = v
    
    # Delete position IDs as in the original code
    if "embeddings.position_ids" in new_state_dict:
        del new_state_dict["embeddings.position_ids"]
    
    # Initialize JAX model with random seed
    rng = jax.random.PRNGKey(0)
    
    # Create dummy input for initialization
    dummy_input_ids = jnp.ones((1, 128), dtype=jnp.int32)
    dummy_attention_mask = jnp.ones((1, 128), dtype=jnp.int32)
    
    # Initialize JAX model parameters
    variables = bert.module.init(rng, dummy_input_ids, dummy_attention_mask)
    
    # Convert PyTorch weights to JAX
    params = transfer_plbert_weights_dict(new_state_dict, variables)
    
    return bert, params
