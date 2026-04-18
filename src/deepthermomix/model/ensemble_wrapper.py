import torch
import torch.nn as nn
import os
import glob
from deepthermomix.model.model_loader import load_model

class EnsembleWrapper(nn.Module):
    def __init__(self, models):
        super(EnsembleWrapper, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        outputs = []
        for model in self.models:
            out = model(*args, **kwargs)
            
            if isinstance(out, tuple):
                outputs.append(out[0])
            else:
                outputs.append(out)
                
        stacked_outputs = torch.stack(outputs)
        mean_output = torch.mean(stacked_outputs, dim=0)
        
        return mean_output, None, None

    def get_uncertainty(self, *args, **kwargs):
        outputs = []
        for model in self.models:
            out = model(*args, **kwargs)
            
            if isinstance(out, tuple):
                outputs.append(out[0])
            else:
                outputs.append(out)
        
        stacked_outputs = torch.stack(outputs)
        mean_output = torch.mean(stacked_outputs, dim=0)
        std_output = torch.std(stacked_outputs, dim=0)
        
        return mean_output, std_output

def load_ensemble(model_dir, constraint_type='soft', device='cpu'):
    if not os.path.isdir(model_dir):
        raise ValueError(f"Directory not found: {model_dir}")
    
    model_paths = glob.glob(os.path.join(model_dir, "*.pt"))
    model_paths.sort()

    if not model_paths:
        raise ValueError(f"No .pt files found in {model_dir}")

    print(f"Loading Ensemble from {len(model_paths)} models in {model_dir}...")
    
    loaded_models = []
    for path in model_paths:
        m = load_model(path, constraint_type=constraint_type)
        m = m.to(device)
        m.eval()
        loaded_models.append(m)

    if not loaded_models:
        raise RuntimeError("Could not load any models for the ensemble.")

    ensemble = EnsembleWrapper(loaded_models)
    return ensemble.to(device)