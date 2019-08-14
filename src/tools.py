import os
import torch

def find_ckpt(path, name):
    epo = 0
    ckpt_list = list(os.walk(path))[0][2]
    if f'{name}.pt' in ckpt_list:
        return f'{name}.pt'
    for ckpt_file in ckpt_list:
        if ckpt_file.startswith(name) and ckpt_file.endswith('.pt'):
            ep = int(ckpt_file[:-3].split('-')[1])
            if ep > epo:
                epo = ep
    return f'{name}-{epo}.pt'

def save_ckpt(path, name, model, optimizer, scheduler, epoch, step):
    torch.save({
        'epoch': epoch, 
        'step': step, 
        'model_state_dict': model.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(), 
        'scheduler_state_dict': scheduler.state_dict(), 
    }, os.path.join(path, f'{name}-{epoch}.pt'))

def load_ckpt(path, name, model, optimizer=None, scheduler=None, epoch=0):
    if epoch > 0:
        ckpt_file = f'{name}-{epoch}.pt'
    else:
        ckpt_file = find_ckpt(path, name)
    state = torch.load(os.path.join(path, ckpt_file))
    model.load_state_dict(state['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler_state_dict'])
    return state['epoch'], state['step']