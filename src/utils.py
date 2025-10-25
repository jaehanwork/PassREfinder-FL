import os
import json
import torch

def save_model(save_path, model_name, model, training_state=None):
    state_dict = model.state_dict()
    
    torch.save(state_dict, os.path.join(save_path, model_name))
    with open(os.path.join(save_path, 'training_state.json'), 'w') as f:
        json.dump(training_state, f)

def load_model(load_path, model):
    state_dict = torch.load(os.path.join(load_path))
    
    model.load_state_dict(state_dict)
    return model
