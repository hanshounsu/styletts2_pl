from flax.core import unfreeze, freeze
import numpy as np
import re

def transfer_jdcnet_weights(pytorch_model, jax_variables):
    """Transfer weights from PyTorch JDCNet to JAX JDCNet"""
    params = unfreeze(jax_variables)
    
    # First convolutional block
    params["params"]["Conv_0"]["kernel"] = np.transpose(
        pytorch_model.conv_block[0].weight.detach().cpu().numpy(), (2, 3, 1, 0)
    )
    params["params"]["Conv_0"]["bias"] = pytorch_model.conv_block[0].bias.detach().cpu().numpy() if pytorch_model.conv_block[0].bias is not None else np.zeros(64)
    
    # First batch norm
    bn = pytorch_model.conv_block[1]
    params["params"]["BatchNorm_0"]["scale"] = bn.weight.detach().cpu().numpy()
    params["params"]["BatchNorm_0"]["bias"] = bn.bias.detach().cpu().numpy()
    params["batch_stats"]["BatchNorm_0"]["mean"] = bn.running_mean.detach().cpu().numpy()
    params["batch_stats"]["BatchNorm_0"]["var"] = bn.running_var.detach().cpu().numpy()
    
    # Second conv
    params["params"]["Conv_1"]["kernel"] = np.transpose(
        pytorch_model.conv_block[3].weight.detach().cpu().numpy(), (2, 3, 1, 0)
    )
    params["params"]["Conv_1"]["bias"] = pytorch_model.conv_block[3].bias.detach().cpu().numpy() if pytorch_model.conv_block[3].bias is not None else np.zeros(64)
    
    # Residual blocks (3 of them)
    resblocks_pt = [pytorch_model.res_block1, pytorch_model.res_block2, pytorch_model.res_block3]
    for i, resblock_pt in enumerate(resblocks_pt):
        # First batch norm in resblock (from pre_conv.0)
        bn = resblock_pt.pre_conv[0]
        params["params"][f"ResBlockJax_{i}"]["BatchNorm_0"]["scale"] = bn.weight.detach().cpu().numpy()
        params["params"][f"ResBlockJax_{i}"]["BatchNorm_0"]["bias"] = bn.bias.detach().cpu().numpy()
        params["batch_stats"][f"ResBlockJax_{i}"]["BatchNorm_0"]["mean"] = bn.running_mean.detach().cpu().numpy()
        params["batch_stats"][f"ResBlockJax_{i}"]["BatchNorm_0"]["var"] = bn.running_var.detach().cpu().numpy()
        
        # First conv in resblock (from conv.0)
        conv = resblock_pt.conv[0]
        params["params"][f"ResBlockJax_{i}"]["Conv_0"]["kernel"] = np.transpose(
            conv.weight.detach().cpu().numpy(), (2, 3, 1, 0)
        )
        params["params"][f"ResBlockJax_{i}"]["Conv_0"]["bias"] = conv.bias.detach().cpu().numpy() if conv.bias is not None else np.zeros(conv.out_channels)
        
        # Second batch norm in resblock (from conv.1)
        bn = resblock_pt.conv[1]
        params["params"][f"ResBlockJax_{i}"]["BatchNorm_1"]["scale"] = bn.weight.detach().cpu().numpy()
        params["params"][f"ResBlockJax_{i}"]["BatchNorm_1"]["bias"] = bn.bias.detach().cpu().numpy()
        params["batch_stats"][f"ResBlockJax_{i}"]["BatchNorm_1"]["mean"] = bn.running_mean.detach().cpu().numpy()
        params["batch_stats"][f"ResBlockJax_{i}"]["BatchNorm_1"]["var"] = bn.running_var.detach().cpu().numpy()
        
        # Second conv in resblock (from conv.3)
        conv = resblock_pt.conv[3]
        params["params"][f"ResBlockJax_{i}"]["Conv_1"]["kernel"] = np.transpose(
            conv.weight.detach().cpu().numpy(), (2, 3, 1, 0)
        )
        params["params"][f"ResBlockJax_{i}"]["Conv_1"]["bias"] = conv.bias.detach().cpu().numpy() if conv.bias is not None else np.zeros(conv.out_channels)
        
        # Shortcut conv (from conv1by1)
        params["params"][f"ResBlockJax_{i}"]["Conv_2"]["kernel"] = np.transpose(
            resblock_pt.conv1by1.weight.detach().cpu().numpy(), (2, 3, 1, 0)
        )
        params["params"][f"ResBlockJax_{i}"]["Conv_2"]["bias"] = (
            resblock_pt.conv1by1.bias.detach().cpu().numpy() 
            if resblock_pt.conv1by1.bias is not None 
            else np.zeros(resblock_pt.conv1by1.out_channels)
        )
    
    # Pool block's batch norm
    bn = pytorch_model.pool_block[0]
    params["params"]["BatchNorm_1"]["scale"] = bn.weight.detach().cpu().numpy()
    params["params"]["BatchNorm_1"]["bias"] = bn.bias.detach().cpu().numpy()
    params["batch_stats"]["BatchNorm_1"]["mean"] = bn.running_mean.detach().cpu().numpy()
    params["batch_stats"]["BatchNorm_1"]["var"] = bn.running_var.detach().cpu().numpy()
    
    # LSTM cells - need to handle PyTorch's concatenated format vs JAX's separate gates
    # Forward LSTM
    lstm_fw = pytorch_model.bilstm_classifier
    fw_ih = lstm_fw.weight_ih_l0.detach().cpu().numpy()
    fw_hh = lstm_fw.weight_hh_l0.detach().cpu().numpy()
    
    # Split the gates (PyTorch concatenates [i, f, g, o] gates)
    hidden_size = fw_ih.shape[0] // 4
    
    # Input gates
    params["params"]["ScanLSTMCell_0"]["ii"]["kernel"] = fw_ih[:hidden_size, :].T
    params["params"]["ScanLSTMCell_0"]["if"]["kernel"] = fw_ih[hidden_size:2*hidden_size, :].T
    params["params"]["ScanLSTMCell_0"]["ig"]["kernel"] = fw_ih[2*hidden_size:3*hidden_size, :].T
    params["params"]["ScanLSTMCell_0"]["io"]["kernel"] = fw_ih[3*hidden_size:, :].T
    
    # Hidden gates
    params["params"]["ScanLSTMCell_0"]["hi"]["kernel"] = fw_hh[:hidden_size, :].T
    params["params"]["ScanLSTMCell_0"]["hf"]["kernel"] = fw_hh[hidden_size:2*hidden_size, :].T
    params["params"]["ScanLSTMCell_0"]["hg"]["kernel"] = fw_hh[2*hidden_size:3*hidden_size, :].T
    params["params"]["ScanLSTMCell_0"]["ho"]["kernel"] = fw_hh[3*hidden_size:, :].T
    
    # Biases (PyTorch has separate biases for input and hidden, JAX combines them)
    fw_ih_bias = lstm_fw.bias_ih_l0.detach().cpu().numpy()
    fw_hh_bias = lstm_fw.bias_hh_l0.detach().cpu().numpy()
    
    params["params"]["ScanLSTMCell_0"]["hi"]["bias"] = fw_ih_bias[:hidden_size] + fw_hh_bias[:hidden_size]
    params["params"]["ScanLSTMCell_0"]["hf"]["bias"] = fw_ih_bias[hidden_size:2*hidden_size] + fw_hh_bias[hidden_size:2*hidden_size]
    params["params"]["ScanLSTMCell_0"]["hg"]["bias"] = fw_ih_bias[2*hidden_size:3*hidden_size] + fw_hh_bias[2*hidden_size:3*hidden_size]
    params["params"]["ScanLSTMCell_0"]["ho"]["bias"] = fw_ih_bias[3*hidden_size:] + fw_hh_bias[3*hidden_size:]
    
    # Backward LSTM
    bw_ih = lstm_fw.weight_ih_l0_reverse.detach().cpu().numpy()
    bw_hh = lstm_fw.weight_hh_l0_reverse.detach().cpu().numpy()
    
    # Input gates
    params["params"]["ScanLSTMCell_1"]["ii"]["kernel"] = bw_ih[:hidden_size, :].T
    params["params"]["ScanLSTMCell_1"]["if"]["kernel"] = bw_ih[hidden_size:2*hidden_size, :].T
    params["params"]["ScanLSTMCell_1"]["ig"]["kernel"] = bw_ih[2*hidden_size:3*hidden_size, :].T
    params["params"]["ScanLSTMCell_1"]["io"]["kernel"] = bw_ih[3*hidden_size:, :].T
    
    # Hidden gates
    params["params"]["ScanLSTMCell_1"]["hi"]["kernel"] = bw_hh[:hidden_size, :].T
    params["params"]["ScanLSTMCell_1"]["hf"]["kernel"] = bw_hh[hidden_size:2*hidden_size, :].T
    params["params"]["ScanLSTMCell_1"]["hg"]["kernel"] = bw_hh[2*hidden_size:3*hidden_size, :].T
    params["params"]["ScanLSTMCell_1"]["ho"]["kernel"] = bw_hh[3*hidden_size:, :].T
    
    # Biases
    bw_ih_bias = lstm_fw.bias_ih_l0_reverse.detach().cpu().numpy()
    bw_hh_bias = lstm_fw.bias_hh_l0_reverse.detach().cpu().numpy()
    
    params["params"]["ScanLSTMCell_1"]["hi"]["bias"] = bw_ih_bias[:hidden_size] + bw_hh_bias[:hidden_size]
    params["params"]["ScanLSTMCell_1"]["hf"]["bias"] = bw_ih_bias[hidden_size:2*hidden_size] + bw_hh_bias[hidden_size:2*hidden_size]
    params["params"]["ScanLSTMCell_1"]["hg"]["bias"] = bw_ih_bias[2*hidden_size:3*hidden_size] + bw_hh_bias[2*hidden_size:3*hidden_size]
    params["params"]["ScanLSTMCell_1"]["ho"]["bias"] = bw_ih_bias[3*hidden_size:] + bw_hh_bias[3*hidden_size:]
    
    # Classification layer
    params["params"]["Dense_0"]["kernel"] = pytorch_model.classifier.weight.detach().cpu().numpy().T
    params["params"]["Dense_0"]["bias"] = pytorch_model.classifier.bias.detach().cpu().numpy()
    
    return freeze(params)

import numpy as np
from flax.core import freeze, unfreeze
def transfer_asrcnn_weights(pytorch_model, jax_variables):
    """Transfer weights from PyTorch ASRCNN to JAX ASRCNN"""
    params = unfreeze(jax_variables)

    # print("JAX MODEL STRUCTURE:")
    # print_nested_dict(params["params"])
    # print("\nPYTORCH MODEL STRUCTURE:")
    # print_model_structure(pytorch_model)
    # print("\nPYTORCH MODEL PARAMETERS:")
    # print_pytorch_params(pytorch_model)
    
    # 1. Initial CNN
    init_cnn = pytorch_model.init_cnn
    params["params"]["init_cnn"]["weight"] = np.transpose(
        init_cnn.conv.weight.detach().cpu().numpy(), (2, 1, 0)
    )
    params["params"]["init_cnn"]["bias"] = init_cnn.conv.bias.detach().cpu().numpy()
    
    # 2. CNN layers - handle the exact JAX parameter structure
    for n in range(len(pytorch_model.cnns)):
        # 2.1 ConvBlock (indexed as cnns_n_0)
        conv_block = pytorch_model.cnns[n][0]  # ConvBlock with ModuleList of Sequential blocks
        
        # JAX path with index formatting as seen in the output
        jax_block_path = params["params"][f"cnns_{n}_0"]
        
        # Process each block in ModuleList
        for i, block in enumerate(conv_block.blocks):
            # First Conv layer (index 0)
            first_conv = block[0]  # First ConvNorm
            jax_block_path[f"conv_layers_{i}_conv1"]["weight"] = np.transpose(
                first_conv.conv.weight.detach().cpu().numpy(), (2, 1, 0)
            )
            jax_block_path[f"conv_layers_{i}_conv1"]["bias"] = first_conv.conv.bias.detach().cpu().numpy()
            
            # GroupNorm layer (index 2)
            group_norm = block[2]
            jax_block_path[f"conv_layers_{i}_norm1"]["scale"] = group_norm.weight.detach().cpu().numpy()
            jax_block_path[f"conv_layers_{i}_norm1"]["bias"] = group_norm.bias.detach().cpu().numpy()
            
            # Second Conv layer (index 4)
            second_conv = block[4]
            jax_block_path[f"conv_layers_{i}_conv2"]["weight"] = np.transpose(
                second_conv.conv.weight.detach().cpu().numpy(), (2, 1, 0)
            )
            jax_block_path[f"conv_layers_{i}_conv2"]["bias"] = second_conv.conv.bias.detach().cpu().numpy()
        
        # 2.2 GroupNorm after ConvBlock (indexed as cnns_n_1)
        group_norm = pytorch_model.cnns[n][1]
        params["params"][f"cnns_{n}_1"]["scale"] = group_norm.weight.detach().cpu().numpy()
        params["params"][f"cnns_{n}_1"]["bias"] = group_norm.bias.detach().cpu().numpy()
    
    # 3. Projection layer
    projection = pytorch_model.projection
    params["params"]["projection"]["weight"] = np.transpose(
        projection.conv.weight.detach().cpu().numpy(), (2, 1, 0)
    )
    params["params"]["projection"]["bias"] = projection.conv.bias.detach().cpu().numpy()
    
    # 4. CTC Linear layers
    ctc_linear1 = pytorch_model.ctc_linear[0]  # First linear in Sequential
    params["params"]["ctc_linear"]["layers_0"]["weight"] = ctc_linear1.linear_layer.weight.detach().cpu().numpy().T
    params["params"]["ctc_linear"]["layers_0"]["bias"] = ctc_linear1.linear_layer.bias.detach().cpu().numpy()
    
    ctc_linear2 = pytorch_model.ctc_linear[2]  # Third element (after activation)
    params["params"]["ctc_linear"]["layers_2"]["weight"] = ctc_linear2.linear_layer.weight.detach().cpu().numpy().T
    params["params"]["ctc_linear"]["layers_2"]["bias"] = ctc_linear2.linear_layer.bias.detach().cpu().numpy()
    
    # 5. ASR S2S components - only transfer if needed
    if hasattr(pytorch_model, 'asr_s2s') and "asr_s2s" in params["params"]:
        # Transfer embedding weights
        params["params"]["asr_s2s"]["embedding"]["embedding"] = \
            pytorch_model.asr_s2s.embedding.weight.detach().cpu().numpy()
        
        # Transfer attention components
        pt_attention = pytorch_model.asr_s2s.attention_layer
        jax_attention = params["params"]["asr_s2s"]["attention_layer"]
        
        # Memory layer
        if hasattr(pt_attention, 'memory_layer'):
            jax_attention["memory_layer"]["weight"] = \
                pt_attention.memory_layer.linear_layer.weight.detach().cpu().numpy().T
        
        
        # Query layer (missing)
        if "query_layer" in jax_attention:
            jax_attention["query_layer"]["weight"] = \
                pt_attention.query_layer.linear_layer.weight.detach().cpu().numpy().T

        # Location layer components (missing)
        if "location_layer" in jax_attention:
            if "conv" in jax_attention["location_layer"]:
                jax_attention["location_layer"]["conv"]["weight"] = np.transpose(
                    pt_attention.location_layer.location_conv.conv.weight.detach().cpu().numpy(), 
                    (2, 1, 0)
                )
                if hasattr(pt_attention.location_layer.location_conv.conv, 'bias'):
                    if pt_attention.location_layer.location_conv.conv.bias is not None:
                        jax_attention["location_layer"]["conv"]["bias"] = \
                            pt_attention.location_layer.location_conv.conv.bias.detach().cpu().numpy()
            
            if "linear" in jax_attention["location_layer"]:
                jax_attention["location_layer"]["linear"]["weight"] = \
                    pt_attention.location_layer.location_dense.linear_layer.weight.detach().cpu().numpy().T
                if hasattr(pt_attention.location_layer.location_dense.linear_layer, 'bias'):
                    if pt_attention.location_layer.location_dense.linear_layer.bias is not None:
                        jax_attention["location_layer"]["linear"]["bias"] = \
                            pt_attention.location_layer.location_dense.linear_layer.bias.detach().cpu().numpy()

        # V layer (missing)
        if "v" in jax_attention:
            jax_attention["v"]["weight"] = \
                pt_attention.v.linear_layer.weight.detach().cpu().numpy().T
            if hasattr(pt_attention.v.linear_layer, 'bias'):
                if pt_attention.v.linear_layer.bias is not None:
                    jax_attention["v"]["bias"] = \
                        pt_attention.v.linear_layer.bias.detach().cpu().numpy()

        # Decoder RNN LSTM cell transfers
        if "decoder_rnn" in params["params"]["asr_s2s"]:
            jax_lstm = params["params"]["asr_s2s"]["decoder_rnn"]
            pt_lstm = pytorch_model.asr_s2s.decoder_rnn
            
            # Input-to-hidden weights
            if hasattr(pt_lstm, 'weight_ih') and "ii" in jax_lstm:
                # In PyTorch, weight_ih contains [i, f, g, o] gates concatenated
                # We need to split them for JAX
                ih_weights = pt_lstm.weight_ih.detach().cpu().numpy()
                hidden_size = ih_weights.shape[0] // 4
                
                jax_lstm["ii"]["kernel"] = ih_weights[:hidden_size].T
                jax_lstm["if"]["kernel"] = ih_weights[hidden_size:2*hidden_size].T
                jax_lstm["ig"]["kernel"] = ih_weights[2*hidden_size:3*hidden_size].T
                jax_lstm["io"]["kernel"] = ih_weights[3*hidden_size:].T
            
            # Hidden-to-hidden weights
            if hasattr(pt_lstm, 'weight_hh') and "hi" in jax_lstm:
                # In PyTorch, weight_hh contains [i, f, g, o] gates concatenated
                hh_weights = pt_lstm.weight_hh.detach().cpu().numpy()
                hidden_size = hh_weights.shape[0] // 4
                
                jax_lstm["hi"]["kernel"] = hh_weights[:hidden_size].T
                jax_lstm["hf"]["kernel"] = hh_weights[hidden_size:2*hidden_size].T
                jax_lstm["hg"]["kernel"] = hh_weights[2*hidden_size:3*hidden_size].T
                jax_lstm["ho"]["kernel"] = hh_weights[3*hidden_size:].T
            
            # Biases
            if hasattr(pt_lstm, 'bias_ih') and hasattr(pt_lstm, 'bias_hh'):
                # In PyTorch, biases are also split by gates
                ih_bias = pt_lstm.bias_ih.detach().cpu().numpy()
                hh_bias = pt_lstm.bias_hh.detach().cpu().numpy()
                hidden_size = ih_bias.shape[0] // 4
                
                # JAX combines the biases
                jax_lstm["hi"]["bias"] = ih_bias[:hidden_size] + hh_bias[:hidden_size]
                jax_lstm["hf"]["bias"] = ih_bias[hidden_size:2*hidden_size] + hh_bias[hidden_size:2*hidden_size]
                jax_lstm["hg"]["bias"] = ih_bias[2*hidden_size:3*hidden_size] + hh_bias[2*hidden_size:3*hidden_size]
                jax_lstm["ho"]["bias"] = ih_bias[3*hidden_size:] + hh_bias[3*hidden_size:]
                
                # Skip dynamic jaxpr tracers for now - these will be part of the model graph
                # and don't need explicit transfer
                

        # Project to hidden layer
        if "project_to_hidden" in params["params"]["asr_s2s"]:
            hidden_proj = params["params"]["asr_s2s"]["project_to_hidden"]
            pt_hidden_proj = pytorch_model.asr_s2s.project_to_hidden
            
            if "layers_0" in hidden_proj: # ignore the Tanh layer and bring only the LinearNorm [0]
                hidden_proj["layers_0"]["weight"] = pt_hidden_proj[0].linear_layer.weight.detach().cpu().numpy().T
                hidden_proj["layers_0"]["bias"] = pt_hidden_proj[0].linear_layer.bias.detach().cpu().numpy()

        # Project to n_symbols layer
        if "project_to_n_symbols" in params["params"]["asr_s2s"]:
            symbol_proj = params["params"]["asr_s2s"]["project_to_n_symbols"]
            pt_symbol_proj = pytorch_model.asr_s2s.project_to_n_symbols
            
            symbol_proj["weight"] = pt_symbol_proj.weight.detach().cpu().numpy().T
            symbol_proj["bias"] = pt_symbol_proj.bias.detach().cpu().numpy()
    
    return freeze(params)

def print_nested_dict(d, prefix=''):
    """Helper function to print nested dictionary structure"""
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{prefix}{k}/")
            print_nested_dict(v, prefix + '  ')
        else:
            shape_info = getattr(v, 'shape', None)
            print(f"{prefix}{k}: {type(v).__name__} {shape_info}")

def print_model_structure(model, prefix=""):
    """Print the structure of a PyTorch model"""
    for name, child in model.named_children():
        print(f"{prefix}{name}: {type(child)}")
        print_model_structure(child, prefix + "  ")

def print_pytorch_params(model, prefix=""):
    """Print all parameters in a PyTorch model with their shapes"""
    for name, param in model.named_parameters():
        print(f"{prefix}{name}: {param.shape}")


def transfer_plbert_weights(pytorch_bert, jax_params):
    """Transfer weights from PyTorch BERT to JAX BERT"""
    params = unfreeze(jax_params)

    print("JAX MODEL STRUCTURE:")
    print_nested_dict(params["params"])
    print("\nPYTORCH MODEL STRUCTURE:")
    print_model_structure(pytorch_bert)
    print("\nPYTORCH MODEL PARAMETERS:")
    print_pytorch_params(pytorch_bert)
    
    # Map embeddings
    if hasattr(pytorch_bert, 'embeddings'):
        emb = pytorch_bert.embeddings
        if hasattr(emb, 'word_embeddings'):
            params['params']['embeddings']['word_embeddings']['embedding'] = emb.word_embeddings.weight.detach().cpu().numpy()
        
        if hasattr(emb, 'position_embeddings'):
            params['params']['embeddings']['position_embeddings']['embedding'] = emb.position_embeddings.weight.detach().cpu().numpy()
        
        if hasattr(emb, 'token_type_embeddings'):
            params['params']['embeddings']['token_type_embeddings']['embedding'] = emb.token_type_embeddings.weight.detach().cpu().numpy()
        
        if hasattr(emb, 'LayerNorm'):
            params['params']['embeddings']['LayerNorm']['scale'] = emb.LayerNorm.weight.detach().cpu().numpy()
            params['params']['embeddings']['LayerNorm']['bias'] = emb.LayerNorm.bias.detach().cpu().numpy()
    
    # Map encoder layers
    if hasattr(pytorch_bert, 'encoder'):
        for i, layer in enumerate(pytorch_bert.encoder.layer):
            layer_name = f"{i}"  # May need to adjust based on JAX model structure
            
            # Map attention components
            if hasattr(layer, 'attention'):
                # Self-attention
                att_self = layer.attention.self
                params['params']['encoder']['layer'][layer_name]['attention']['self']['query']['kernel'] = att_self.query.weight.detach().cpu().numpy().T
                params['params']['encoder']['layer'][layer_name]['attention']['self']['query']['bias'] = att_self.query.bias.detach().cpu().numpy()
                
                params['params']['encoder']['layer'][layer_name]['attention']['self']['key']['kernel'] = att_self.key.weight.detach().cpu().numpy().T
                params['params']['encoder']['layer'][layer_name]['attention']['self']['key']['bias'] = att_self.key.bias.detach().cpu().numpy()
                
                params['params']['encoder']['layer'][layer_name]['attention']['self']['value']['kernel'] = att_self.value.weight.detach().cpu().numpy().T
                params['params']['encoder']['layer'][layer_name]['attention']['self']['value']['bias'] = att_self.value.bias.detach().cpu().numpy()
                
                # Attention output
                att_output = layer.attention.output
                params['params']['encoder']['layer'][layer_name]['attention']['output']['dense']['kernel'] = att_output.dense.weight.detach().cpu().numpy().T
                params['params']['encoder']['layer'][layer_name]['attention']['output']['dense']['bias'] = att_output.dense.bias.detach().cpu().numpy()
                
                # Layer norm
                params['params']['encoder']['layer'][layer_name]['attention']['output']['LayerNorm']['scale'] = att_output.LayerNorm.weight.detach().cpu().numpy()
                params['params']['encoder']['layer'][layer_name]['attention']['output']['LayerNorm']['bias'] = att_output.LayerNorm.bias.detach().cpu().numpy()
            
            # Map intermediate layer
            if hasattr(layer, 'intermediate'):
                inter = layer.intermediate
                params['params']['encoder']['layer'][layer_name]['intermediate']['dense']['kernel'] = inter.dense.weight.detach().cpu().numpy().T
                params['params']['encoder']['layer'][layer_name]['intermediate']['dense']['bias'] = inter.dense.bias.detach().cpu().numpy()
            
            # Map output layer
            if hasattr(layer, 'output'):
                out = layer.output
                params['params']['encoder']['layer'][layer_name]['output']['dense']['kernel'] = out.dense.weight.detach().cpu().numpy().T
                params['params']['encoder']['layer'][layer_name]['output']['dense']['bias'] = out.dense.bias.detach().cpu().numpy()
                
                params['params']['encoder']['layer'][layer_name]['output']['LayerNorm']['scale'] = out.LayerNorm.weight.detach().cpu().numpy()
                params['params']['encoder']['layer'][layer_name]['output']['LayerNorm']['bias'] = out.LayerNorm.bias.detach().cpu().numpy()
    
    # Map pooler if present
    if hasattr(pytorch_bert, 'pooler') and 'pooler' in params['params']:
        params['params']['pooler']['dense']['kernel'] = pytorch_bert.pooler.dense.weight.detach().cpu().numpy().T
        params['params']['pooler']['dense']['bias'] = pytorch_bert.pooler.dense.bias.detach().cpu().numpy()
    
    return freeze(params)

# def transfer_plbert_weights_dict(pytorch_state_dict, jax_params):
#     """Transfer weights from PyTorch PLBERT state_dict to JAX ALBERT parameters
    
#     Args:
#         pytorch_state_dict: OrderedDict containing PyTorch model weights
#         jax_params: JAX model parameters from bert.module.init()
        
#     Returns:
#         Updated JAX parameters with weights from PyTorch model
#     """
#     params = unfreeze(jax_params)

#     # Optional: Print structures for debugging
#     print("JAX MODEL STRUCTURE:")
#     print_nested_dict(params["params"])
#     print("\nPYTORCH STATE DICT KEYS:")
#     for key in pytorch_state_dict.keys():
#         print(key)
    
#     # Track converted parameters
#     converted_params = set()
#     missing_in_jax = set()
    
#     # Helper function to safely navigate nested dict paths
#     def set_nested_param(param_dict, path_parts, value, transpose=False):
#         current = param_dict
#         for i, part in enumerate(path_parts[:-1]):
#             if part not in current:
#                 path_so_far = "/".join(path_parts[:i+1])
#                 missing_in_jax.add(path_so_far)
#                 return False
#             current = current[part]
        
#         if path_parts[-1] not in current:
#             path_so_far = "/".join(path_parts)
#             missing_in_jax.add(path_so_far)
#             return False
            
#         if transpose and value.ndim >= 2:
#             current[path_parts[-1]] = value.T
#         else:
#             current[path_parts[-1]] = value
#         return True
    
#     for pt_key, pt_param in pytorch_state_dict.items():
#         if pt_param is None:
#             continue
            
#         # Convert PyTorch tensor to numpy
#         np_param = pt_param.detach().cpu().numpy()
        
#         # Map embeddings
#         if pt_key.startswith('embeddings'):
#             parts = pt_key.split('.')
            
#             if parts[1] == 'word_embeddings' and parts[2] == 'weight':
#                 if set_nested_param(params['params'], ['embeddings', 'word_embeddings', 'embedding'], np_param):
#                     converted_params.add(pt_key)
                    
#             elif parts[1] == 'position_embeddings' and parts[2] == 'weight':
#                 if set_nested_param(params['params'], ['embeddings', 'position_embeddings', 'embedding'], np_param):
#                     converted_params.add(pt_key)
                    
#             elif parts[1] == 'token_type_embeddings' and parts[2] == 'weight':
#                 if set_nested_param(params['params'], ['embeddings', 'token_type_embeddings', 'embedding'], np_param):
#                     converted_params.add(pt_key)
                    
#             elif parts[1] == 'LayerNorm':
#                 if parts[2] == 'weight':
#                     if set_nested_param(params['params'], ['embeddings', 'LayerNorm', 'scale'], np_param):
#                         converted_params.add(pt_key)
#                 elif parts[2] == 'bias':
#                     if set_nested_param(params['params'], ['embeddings', 'LayerNorm', 'bias'], np_param):
#                         converted_params.add(pt_key)
        
#         # Map embedding projection (specific to ALBERT)
#         elif 'embedding_hidden_mapping_in' in pt_key:
#             parts = pt_key.split('.')
#             last_part = parts[-1]
            
#             if last_part == 'weight':
#                 if set_nested_param(params['params'], ['encoder', 'embedding_hidden_mapping_in', 'kernel'], np_param, transpose=True):
#                     converted_params.add(pt_key)
#             elif last_part == 'bias':
#                 if set_nested_param(params['params'], ['encoder', 'embedding_hidden_mapping_in', 'bias'], np_param):
#                     converted_params.add(pt_key)
        
#         # Handle ALBERT layer groups
#         elif 'albert_layer_groups' in pt_key:
#             # Extract group index and layer index
#             match_group = re.search(r'albert_layer_groups\.(\d+)\.albert_layers\.(\d+)\.', pt_key)
#             if match_group:
#                 group_idx, layer_idx = match_group.groups()
                
#                 # Process the remainder of the key to find what component this is
#                 remainder = pt_key.split(f'albert_layer_groups.{group_idx}.albert_layers.{layer_idx}.')[-1]
#                 parts = remainder.split('.')
                
#                 # Handle attention components
#                 if parts[0] == 'attention':
#                     # Self attention
#                     if len(parts) >= 3 and parts[1] == 'self':
#                         component = parts[2]
#                         param_type = parts[3]
                        
#                         # Map to appropriate JAX parameter
#                         jax_path = ['encoder', f'albert_layer_groups_{group_idx}', 
#                                    f'albert_layers_{layer_idx}', 'attention', 'self', component]
                        
#                         if param_type == 'weight':
#                             if set_nested_param(params['params'], jax_path + ['kernel'], np_param, transpose=True):
#                                 converted_params.add(pt_key)
#                         elif param_type == 'bias':
#                             if set_nested_param(params['params'], jax_path + ['bias'], np_param):
#                                 converted_params.add(pt_key)
                    
#                     # Attention output
#                     elif len(parts) >= 3 and parts[1] == 'output':
#                         if parts[2] == 'dense':
#                             param_type = parts[3]
#                             jax_path = ['encoder', f'albert_layer_groups_{group_idx}', 
#                                       f'albert_layers_{layer_idx}', 'attention', 'output', 'dense']
                            
#                             if param_type == 'weight':
#                                 if set_nested_param(params['params'], jax_path + ['kernel'], np_param, transpose=True):
#                                     converted_params.add(pt_key)
#                             elif param_type == 'bias':
#                                 if set_nested_param(params['params'], jax_path + ['bias'], np_param):
#                                     converted_params.add(pt_key)
                                    
#                         elif parts[2] == 'LayerNorm':
#                             param_type = parts[3]
#                             jax_path = ['encoder', f'albert_layer_groups_{group_idx}', 
#                                       f'albert_layers_{layer_idx}', 'attention', 'output', 'LayerNorm']
                            
#                             if param_type == 'weight':
#                                 if set_nested_param(params['params'], jax_path + ['scale'], np_param):
#                                     converted_params.add(pt_key)
#                             elif param_type == 'bias':
#                                 if set_nested_param(params['params'], jax_path + ['bias'], np_param):
#                                     converted_params.add(pt_key)
                
#                 # Handle FFN components
#                 elif parts[0] == 'ffn' or parts[0] == 'intermediate':
#                     if parts[0] == 'intermediate' and parts[1] == 'dense':
#                         param_type = parts[2]
#                         jax_path = ['encoder', f'albert_layer_groups_{group_idx}', 
#                                   f'albert_layers_{layer_idx}', 'intermediate', 'dense']
                        
#                         if param_type == 'weight':
#                             if set_nested_param(params['params'], jax_path + ['kernel'], np_param, transpose=True):
#                                 converted_params.add(pt_key)
#                         elif param_type == 'bias':
#                             if set_nested_param(params['params'], jax_path + ['bias'], np_param):
#                                 converted_params.add(pt_key)
                    
#                     elif parts[0] == 'ffn' and parts[1] == 'intermediate':
#                         # Sometimes ffn.intermediate is used instead of just intermediate
#                         if parts[2] == 'dense':
#                             param_type = parts[3]
#                             jax_path = ['encoder', f'albert_layer_groups_{group_idx}', 
#                                       f'albert_layers_{layer_idx}', 'intermediate', 'dense']
                            
#                             if param_type == 'weight':
#                                 if set_nested_param(params['params'], jax_path + ['kernel'], np_param, transpose=True):
#                                     converted_params.add(pt_key)
#                             elif param_type == 'bias':
#                                 if set_nested_param(params['params'], jax_path + ['bias'], np_param):
#                                     converted_params.add(pt_key)
                    
#                     elif parts[0] == 'ffn' and parts[1] == 'output':
#                         if parts[2] == 'dense':
#                             param_type = parts[3]
#                             jax_path = ['encoder', f'albert_layer_groups_{group_idx}', 
#                                       f'albert_layers_{layer_idx}', 'output', 'dense']
                            
#                             if param_type == 'weight':
#                                 if set_nested_param(params['params'], jax_path + ['kernel'], np_param, transpose=True):
#                                     converted_params.add(pt_key)
#                             elif param_type == 'bias':
#                                 if set_nested_param(params['params'], jax_path + ['bias'], np_param):
#                                     converted_params.add(pt_key)
                        
#                         elif parts[2] == 'LayerNorm':
#                             param_type = parts[3]
#                             jax_path = ['encoder', f'albert_layer_groups_{group_idx}', 
#                                       f'albert_layers_{layer_idx}', 'output', 'LayerNorm']
                            
#                             if param_type == 'weight':
#                                 if set_nested_param(params['params'], jax_path + ['scale'], np_param):
#                                     converted_params.add(pt_key)
#                             elif param_type == 'bias':
#                                 if set_nested_param(params['params'], jax_path + ['bias'], np_param):
#                                     converted_params.add(pt_key)
                
#                 # Output components (if not part of ffn)
#                 elif parts[0] == 'output':
#                     if parts[1] == 'dense':
#                         param_type = parts[2]
#                         jax_path = ['encoder', f'albert_layer_groups_{group_idx}', 
#                                   f'albert_layers_{layer_idx}', 'output', 'dense']
                        
#                         if param_type == 'weight':
#                             if set_nested_param(params['params'], jax_path + ['kernel'], np_param, transpose=True):
#                                 converted_params.add(pt_key)
#                         elif param_type == 'bias':
#                             if set_nested_param(params['params'], jax_path + ['bias'], np_param):
#                                 converted_params.add(pt_key)
                    
#                     elif parts[1] == 'LayerNorm':
#                         param_type = parts[2]
#                         jax_path = ['encoder', f'albert_layer_groups_{group_idx}', 
#                                   f'albert_layers_{layer_idx}', 'output', 'LayerNorm']
                        
#                         if param_type == 'weight':
#                             if set_nested_param(params['params'], jax_path + ['scale'], np_param):
#                                 converted_params.add(pt_key)
#                         elif param_type == 'bias':
#                             if set_nested_param(params['params'], jax_path + ['bias'], np_param):
#                                 converted_params.add(pt_key)
        
#         # Handle standard BERT-style layer format (sometimes used in ALBERT models too)
#         elif 'encoder.layer.' in pt_key:
#             # Extract layer index
#             match_layer = re.search(r'encoder\.layer\.(\d+)\.', pt_key)
#             if match_layer:
#                 layer_idx = match_layer.group(1)
                
#                 # Get the remainder of the path
#                 remainder = pt_key.split(f'encoder.layer.{layer_idx}.')[-1]
#                 parts = remainder.split('.')
                
#                 # Map to appropriate JAX parameter paths
#                 if parts[0] == 'attention':
#                     # Self attention components
#                     if len(parts) >= 3 and parts[1] == 'self':
#                         component = parts[2]
#                         param_type = parts[3]
                        
#                         jax_path = ['encoder', 'layer', layer_idx, 'attention', 'self', component]
                        
#                         if param_type == 'weight':
#                             if set_nested_param(params['params'], jax_path + ['kernel'], np_param, transpose=True):
#                                 converted_params.add(pt_key)
#                         elif param_type == 'bias':
#                             if set_nested_param(params['params'], jax_path + ['bias'], np_param):
#                                 converted_params.add(pt_key)
                    
#                     # Attention output
#                     elif len(parts) >= 3 and parts[1] == 'output':
#                         if parts[2] == 'dense':
#                             param_type = parts[3]
#                             jax_path = ['encoder', 'layer', layer_idx, 'attention', 'output', 'dense']
                            
#                             if param_type == 'weight':
#                                 if set_nested_param(params['params'], jax_path + ['kernel'], np_param, transpose=True):
#                                     converted_params.add(pt_key)
#                             elif param_type == 'bias':
#                                 if set_nested_param(params['params'], jax_path + ['bias'], np_param):
#                                     converted_params.add(pt_key)
                        
#                         elif parts[2] == 'LayerNorm':
#                             param_type = parts[3]
#                             jax_path = ['encoder', 'layer', layer_idx, 'attention', 'output', 'LayerNorm']
                            
#                             if param_type == 'weight':
#                                 if set_nested_param(params['params'], jax_path + ['scale'], np_param):
#                                     converted_params.add(pt_key)
#                             elif param_type == 'bias':
#                                 if set_nested_param(params['params'], jax_path + ['bias'], np_param):
#                                     converted_params.add(pt_key)
                
#                 # Handle intermediate layer
#                 elif parts[0] == 'intermediate':
#                     if parts[1] == 'dense':
#                         param_type = parts[2]
#                         jax_path = ['encoder', 'layer', layer_idx, 'intermediate', 'dense']
                        
#                         if param_type == 'weight':
#                             if set_nested_param(params['params'], jax_path + ['kernel'], np_param, transpose=True):
#                                 converted_params.add(pt_key)
#                         elif param_type == 'bias':
#                             if set_nested_param(params['params'], jax_path + ['bias'], np_param):
#                                 converted_params.add(pt_key)
                
#                 # Handle output layer
#                 elif parts[0] == 'output':
#                     if parts[1] == 'dense':
#                         param_type = parts[2]
#                         jax_path = ['encoder', 'layer', layer_idx, 'output', 'dense']
                        
#                         if param_type == 'weight':
#                             if set_nested_param(params['params'], jax_path + ['kernel'], np_param, transpose=True):
#                                 converted_params.add(pt_key)
#                         elif param_type == 'bias':
#                             if set_nested_param(params['params'], jax_path + ['bias'], np_param):
#                                 converted_params.add(pt_key)
                    
#                     elif parts[1] == 'LayerNorm':
#                         param_type = parts[2]
#                         jax_path = ['encoder', 'layer', layer_idx, 'output', 'LayerNorm']
                        
#                         if param_type == 'weight':
#                             if set_nested_param(params['params'], jax_path + ['scale'], np_param):
#                                 converted_params.add(pt_key)
#                         elif param_type == 'bias':
#                             if set_nested_param(params['params'], jax_path + ['bias'], np_param):
#                                 converted_params.add(pt_key)
        
#         # Map pooler (if present in both models)
#         elif pt_key.startswith('pooler.'):
#             if pt_key.endswith('dense.weight'):
#                 if set_nested_param(params['params'], ['pooler', 'dense', 'kernel'], np_param, transpose=True):
#                     converted_params.add(pt_key)
#             elif pt_key.endswith('dense.bias'):
#                 if set_nested_param(params['params'], ['pooler', 'dense', 'bias'], np_param):
#                     converted_params.add(pt_key)
    
#     # Report conversion statistics
#     if missing_in_jax:
#         print(f"Warning: Some paths were missing in the JAX model: {missing_in_jax}")
    
#     missing_params = set(pytorch_state_dict.keys()) - converted_params
#     if missing_params:
#         print(f"Warning: {len(missing_params)}/{len(pytorch_state_dict)} parameters were not transferred")
#         if len(missing_params) <= 10:
#             for param in missing_params:
#                 print(f"  Missing: {param}")
#         else:
#             for param in list(missing_params):
#                 print(f"  Missing: {param}")
#             # print(f"  ... and {len(missing_params) - 5} more")
#     else:
#         print(f"Successfully transferred all {len(converted_params)} parameters")
    
#     return freeze(params)


def transfer_plbert_weights_dict(pytorch_state_dict, jax_variables):
    """Transfer weights from PyTorch PLBERT state_dict to JAX ALBERT parameters"""
    params = unfreeze(jax_variables)
    
    # Debug: Print JAX model structure to understand its organization
    # print("JAX MODEL STRUCTURE (first few levels):")
    # debug_print_structure(params["params"], max_depth=4)
    
    # Track converted parameters
    converted_params = set()
    missing_in_jax = set()
    
    # Helper function to safely navigate nested dict paths
    def set_nested_param(param_dict, path_parts, value, transpose=False):
        current = param_dict
        for i, part in enumerate(path_parts[:-1]):
            if part not in current:
                path_so_far = "/".join(path_parts[:i+1])
                missing_in_jax.add(path_so_far)
                return False
            current = current[part]
        
        if path_parts[-1] not in current:
            path_so_far = "/".join(path_parts)
            missing_in_jax.add(path_so_far)
            return False
            
        if transpose and value.ndim >= 2:
            current[path_parts[-1]] = value.T
        else:
            current[path_parts[-1]] = value
        return True
    
    # Create mapping dictionaries for different parameter paths
    attention_map = {
        "query.weight": ("attention", "self", "query", "kernel"),
        "query.bias": ("attention", "self", "query", "bias"),
        "key.weight": ("attention", "self", "key", "kernel"),
        "key.bias": ("attention", "self", "key", "bias"),
        "value.weight": ("attention", "self", "value", "kernel"),
        "value.bias": ("attention", "self", "value", "bias"),
        "dense.weight": ("attention", "output", "dense", "kernel"),
        "dense.bias": ("attention", "output", "dense", "bias"),
        "LayerNorm.weight": ("attention", "output", "LayerNorm", "scale"),
        "LayerNorm.bias": ("attention", "output", "LayerNorm", "bias")
    }
    
    ffn_map = {
        "ffn.weight": ("intermediate", "dense", "kernel"),
        "ffn.bias": ("intermediate", "dense", "bias"),
        "ffn_output.weight": ("output", "dense", "kernel"),
        "ffn_output.bias": ("output", "dense", "bias"),
        "full_layer_layer_norm.weight": ("output", "LayerNorm", "scale"),
        "full_layer_layer_norm.bias": ("output", "LayerNorm", "bias")
    }
    
    for pt_key, pt_param in pytorch_state_dict.items():
        if pt_param is None:
            continue
            
        # Convert PyTorch tensor to numpy
        np_param = pt_param.detach().cpu().numpy()
        
        # Handle embeddings
        if pt_key.startswith('embeddings'):
            parts = pt_key.split('.')
            
            if parts[1] == 'word_embeddings' and parts[2] == 'weight':
                if set_nested_param(params['params'], ['embeddings', 'word_embeddings', 'embedding'], np_param):
                    converted_params.add(pt_key)
                    
            elif parts[1] == 'position_embeddings' and parts[2] == 'weight':
                if set_nested_param(params['params'], ['embeddings', 'position_embeddings', 'embedding'], np_param):
                    converted_params.add(pt_key)
                    
            elif parts[1] == 'token_type_embeddings' and parts[2] == 'weight':
                if set_nested_param(params['params'], ['embeddings', 'token_type_embeddings', 'embedding'], np_param):
                    converted_params.add(pt_key)
                    
            elif parts[1] == 'LayerNorm':
                if parts[2] == 'weight':
                    if set_nested_param(params['params'], ['embeddings', 'LayerNorm', 'scale'], np_param):
                        converted_params.add(pt_key)
                elif parts[2] == 'bias':
                    if set_nested_param(params['params'], ['embeddings', 'LayerNorm', 'bias'], np_param):
                        converted_params.add(pt_key)
        
        # Handle embedding projection
        elif 'embedding_hidden_mapping_in' in pt_key:
            parts = pt_key.split('.')
            last_part = parts[-1]
            
            if last_part == 'weight':
                if set_nested_param(params['params'], ['encoder', 'embedding_hidden_mapping_in', 'kernel'], np_param, transpose=True):
                    converted_params.add(pt_key)
            elif last_part == 'bias':
                if set_nested_param(params['params'], ['encoder', 'embedding_hidden_mapping_in', 'bias'], np_param):
                    converted_params.add(pt_key)
        
        # Handle ALBERT layer groups
        elif 'albert_layer_groups' in pt_key:
            # Extract group and layer indices
            match = re.search(r'albert_layer_groups\.(\d+)\.albert_layers\.(\d+)\.(.+)', pt_key)
            if match:
                group_idx, layer_idx, remainder = match.groups()
                
                # Try different path formats based on JAX model structure
                path_formats = [
                    # Format 1: Original JAX path format
                    ['encoder', f'albert_layer_groups_{group_idx}', f'albert_layers_{layer_idx}'],
                    
                    # Format 2: Alternative format seen in some JAX models
                    ['encoder', 'albert_layer_groups', group_idx, 'albert_layers', layer_idx],
                    
                    # Format 3: Flattened structure for single layer group
                    ['encoder', 'layer', layer_idx]
                ]
                
                # Check for attention components
                for attention_key, attention_path in attention_map.items():
                    if remainder.endswith(attention_key):
                        for base_path in path_formats:
                            full_path = base_path + list(attention_path)
                            transpose = attention_key.endswith('.weight')
                            if set_nested_param(params['params'], full_path, np_param, transpose=transpose):
                                converted_params.add(pt_key)
                                break
                
                # Check for FFN components
                for ffn_key, ffn_path in ffn_map.items():
                    if remainder.endswith(ffn_key):
                        for base_path in path_formats:
                            full_path = base_path + list(ffn_path)
                            transpose = ffn_key.endswith('.weight')
                            if set_nested_param(params['params'], full_path, np_param, transpose=transpose):
                                converted_params.add(pt_key)
                                break
        
        # Handle pooler
        elif pt_key.startswith('pooler.'):
            if pt_key.endswith('weight'):
                if set_nested_param(params['params'], ['pooler', 'dense', 'kernel'], np_param, transpose=True):
                    converted_params.add(pt_key)
            elif pt_key.endswith('bias'):
                if set_nested_param(params['params'], ['pooler', 'dense', 'bias'], np_param):
                    converted_params.add(pt_key)
    
    # Report transfer statistics
    untransferred = set(pytorch_state_dict.keys()) - converted_params
    print(f"{len(converted_params)}/{len(pytorch_state_dict)} parameters transferred")
    
    if untransferred:
        print(f"Warning: {len(untransferred)}/{len(pytorch_state_dict)} parameters were not transferred")
        for param in sorted(list(untransferred)):
            print(f"  Missing: {param}")
        # if len(untransferred) > 10:
        #     print(f"  ... and {len(untransferred)-10} more")
    
    return freeze(params)

def debug_print_structure(d, prefix='', max_depth=None, current_depth=0):
    """Print structure of nested dictionary with depth limit"""
    if max_depth is not None and current_depth >= max_depth:
        print(f"{prefix}... (max depth reached)")
        return
        
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            print(f"{prefix}{k}/")
            debug_print_structure(v, prefix + '  ', max_depth, current_depth + 1)
        else:
            shape_info = getattr(v, 'shape', None)
            print(f"{prefix}{k}: {shape_info}")