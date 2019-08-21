import os
import torch

def load_ckpt(path, name, model):
    ckpt_file = f'{name}.pt'
    state = torch.load(os.path.join(path, ckpt_file))
    model.load_state_dict(state['model_state_dict'])