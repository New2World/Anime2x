import os, cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision as tv
import numpy as np

import model

c_mean = [0.396, 0.504, 0.570]
c_std = [0.220, 0.219, 0.215]

class SuperResolutionDataset(Dataset):
    def __init__(self):
        self.n_samples = len(list(os.walk("data"))[0][2])
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        I = cv2.imread(f"General-100/im_{idx}.png")
        image = cv2.imread(f"data/im_{idx}.png")
        I = torch.from_numpy(I.transpose(2,0,1)).type(torch.FloatTensor)/255.
        image = torch.from_numpy(image.transpose(2,0,1)).type(torch.FloatTensor)/255.
        for i in range(3):
            I[i] = (I[i]-c_mean[i])/c_std[i]
            image[i] = (image[i]-c_mean[i])/c_std[i]
        if torch.cuda.is_available():
            image = image.cuda()
            I = I.cuda()
        return {"low":image, "high":I}

def training_loop(model, dataloader, summary, epoch=20, lr1=1e-3, lr2=1e-4):
    criteria = nn.MSELoss()
    if torch.cuda.is_available():
        model.cuda()
        criteria.cuda()
    optimizer = optim.Adam([{'params': model.encoder.parameters(), 'lr':lr1},
                            {'params': model.mapping.parameters(), 'lr':lr1},
                            {'params': model.decoder.parameters(), 'lr':lr2}])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=140)
    for ep in range(epoch):
        avg_loss = 0.
        batch = 0
        for sample in dataloader:
            inp = sample["low"]
            gt = sample["high"]
            outp = model(inp)
            mse_loss = criteria(gt, outp)
            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()
            avg_loss += mse_loss
            batch += 1
        lr_scheduler.step()
        avg_loss /= batch
        print(f"Ep.{ep+1:3d}/{epoch} - loss: {avg_loss:.6f}")
        summary.add_scalar("loss", avg_loss, global_step=ep)
        if (ep+1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/fsrcnn_{ep+1}.pt")

dataset = SuperResolutionDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
fsrcnn = model.FSRCNN()
if torch.cuda.is_available():
    fsrcnn.cuda()
summary = SummaryWriter(log_dir="logdir")
training_loop(fsrcnn, dataloader, summary, epoch=200)
summary.close()