import torch
import torch.nn as nn
import numpy as np 
from deepthermomix.model.architecture import DTMPNN 

def infer_model_architecture(state_dict):

    node_dim = state_dict['graph_block.layers.0.lin_node.weight'].shape[1]
    edge_dim = state_dict['graph_block.layers.0.lin_edge.weight'].shape[1]
    graph_hidden_dim = state_dict['graph_block.layers.0.lin_node.weight'].shape[0]

    context_dim = state_dict['mixture_layer.interaction_mlp.0.weight'].shape[0]
    latent_dim = state_dict['mixture_layer.gate_mlp.0.weight'].shape[0]

    layer_indices = [int(k.split('.')[2]) for k in state_dict.keys() 
                     if k.startswith('graph_block.layers.') and k.split('.')[2].isdigit()]
    graph_layers = max(layer_indices) + 1 if layer_indices else 1

    return {
        'node_dim': node_dim,
        'edge_dim': edge_dim,
        'graph_hidden_dim': graph_hidden_dim,
        'latent_dim': latent_dim,
        'context_dim': context_dim,
        'graph_layers': graph_layers,
    }

def load_model(checkpoint_path, constraint_type='soft', verbose=False):
    try:
        if hasattr(np, 'core') and hasattr(np.core, 'multiarray'):
             torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except Exception:
        pass

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            if 'graph_block.layers.0.lin_node.weight' in checkpoint:
                state_dict = checkpoint
            else:
                state_dict = checkpoint 
    else:
        state_dict = checkpoint
    
    model_params = infer_model_architecture(state_dict)
    
    model_params['constraint_type'] = constraint_type
    if isinstance(checkpoint, dict):
        saved_args = checkpoint.get('args', {}) or checkpoint.get('config', {})
        if isinstance(saved_args, dict) and 'constraint_type' in saved_args:
            model_params['constraint_type'] = saved_args['constraint_type']
            if verbose == True: 
                print(f"Found constraint_type='{model_params['constraint_type']}' in checkpoint config.")

    if verbose == True:
        print("Inferred model parameters:")
        for k, v in model_params.items():
            print(f"  {k}: {v}")
    
    model = DTMPNN(**model_params)
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Warning: strict loading failed ({e}). Retrying with strict=False...")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    
    if verbose == True:
        print("\nModel loaded successfully!")

    return model