import os
import skimage.io as si
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
import torchvision as tv
import numpy as np

import model

c_mean = [0.513, 0.476, 0.471]
c_std = [0.367, 0.355, 0.348]

class SuperResolutionDataset(Dataset):
    def __init__(self):
        self.n_samples = len(list(os.walk("data"))[0][2])
        self.preprocess = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.RandomCrop(256, padding=4, padding_mode='symmetric'),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomVerticalFlip()
        ])
        self.resize = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.Resize(128)
        ])
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        I = si.imread(f"data/img_{idx}.jpg")
        I = np.array(self.preprocess(I))
        image = np.array(self.resize(I))
        I = torch.from_numpy(I.transpose(2,0,1)).type(torch.FloatTensor)
        image = torch.from_numpy(image.transpose(2,0,1)).type(torch.FloatTensor)
        for i in range(3):
            I[i] = (I[i]/255.-c_mean[i])/c_std[i]
            image[i] = (image[i]/255.-c_mean[i])/c_std[i]
        if torch.cuda.is_available():
            image = image.cuda()
            I = I.cuda()
        return {"low":image, "high":I}

def training_loop(model, dataloader, step=0, epoch=10, lr1=1e-3, lr2=1e-4, summary=None):
    criteria = nn.MSELoss()
    if torch.cuda.is_available():
        model.cuda()
        criteria.cuda()
    optimizer = optim.Adam([{'params': model.encoder.parameters(), 'lr':lr1},
                            {'params': model.mapping.parameters(), 'lr':lr1},
                            {'params': model.decoder.parameters(), 'lr':lr2}])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 5)
    lr_scheduler.step()
    lr_scheduler.step()
    lr_scheduler.step()
    lr_scheduler.step()
    for ep in range(epoch):
        avg_loss = 0.
        for sample in dataloader:
            inp = sample["low"]
            gt = sample["high"]
            outp = model(inp)
            mse_loss = criteria(gt, outp)
            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()
            avg_loss += mse_loss
            step += 1
            if step % 100 == 0:
                avg_loss /= 100
                print(f"Ep.{ep+1} - step.{step} - loss: {avg_loss:.6f}")
                if summary is not None:
                    summary.add_scalar("loss", avg_loss, global_step=step)
                avg_loss = 0.
            if step % 1000 == 0:
                torch.save(model.state_dict(), f"checkpoints/fsrcnn_{step}.pt")
        lr_scheduler.step()

dataset = SuperResolutionDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
fsrcnn = model.FSRCNN()

fsrcnn.load_state_dict(torch.load("checkpoints/fsrcnn_16000.pt"))

if torch.cuda.is_available():
    fsrcnn.cuda()
# summary = SummaryWriter(log_dir="logdir")
training_loop(fsrcnn, dataloader, step=16000, epoch=20, lr1=1e-5, lr2=1e-6)
# summary.close()
torch.save(model.state_dict(), "checkpoints/fsrcnn_final.pt")