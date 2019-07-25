import os, cv2
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
        self.n_samples = len(list(os.walk("../data"))[0][2])
        self.preprocess = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.RandomCrop(320),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomVerticalFlip()
        ])
        self.resize = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.Resize(160)
        ])
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        I = cv2.imread(f"../data/img_{idx}.jpg")
        I = cv2.split(cv2.cvtColor(I, cv2.BGR2YUV))[0]
        I = np.array(self.preprocess(I))
        image = np.array(self.resize(I))
        I = torch.from_numpy(I).type(torch.FloatTensor)/255.
        image = torch.from_numpy(image).type(torch.FloatTensor)/255.
        if torch.cuda.is_available():
            image = image.cuda()
            I = I.cuda()
        return {"low":image, "high":I}

def training_loop(model, dataloader, step=0, epoch=10, lr1=1e-3, lr2=1e-4, summary=None):
    criteria = nn.SmoothL1Loss()
    if torch.cuda.is_available():
        model.cuda()
        criteria.cuda()
    optimizer = optim.Adam([{'params': model.encoder.parameters(), 'lr':lr1},
                            {'params': model.mapping.parameters(), 'lr':lr1},
                            {'params': model.decoder.parameters(), 'lr':lr2}])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 5)
    for ep in range(epoch):
        avg_loss = 0.
        for sample in dataloader:
            inp = sample["low"]
            gt = sample["high"]
            outp = model(inp)
            outp = torch.clamp(outp, 0, 1)
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
                torch.save(model.state_dict(), f"../checkpoints/fsrcnn_{step}.pt")
        torch.save(model.state_dict(), f"../checkpoints/fsrcnn_ep{ep+1}.pt")
        lr_scheduler.step()

dataset = SuperResolutionDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
fsrcnn = model.FSRCNN()
if torch.cuda.is_available():
    fsrcnn.cuda()
# summary = SummaryWriter(log_dir="logdir")
training_loop(fsrcnn, dataloader, epoch=20)
# summary.close()
torch.save(fsrcnn.state_dict(), "../checkpoints/fsrcnn_final.pt")