import os, random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from PIL import ImageFilter

from fsrcnn import FSRCNN
from tools import load_ckpt, save_ckpt

class SuperResolutionDataset(Dataset):
    def __init__(self, path, blur=False):
        self.path = path
        self.n_samples = len(list(os.walk(path))[0][2])
        self.preprocess = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.RandomCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomVerticalFlip()
        ])
        self.downscale = tv.transforms.Resize(112)
        self.totensor = tv.transforms.ToTensor()
        self.blur = blur
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        I = cv2.imread(os.path.join(self.path, f'img_{idx}.png'))
        y = cv2.split(cv2.cvtColor(I, cv2.COLOR_BGR2YCrCb))[0]
        I = self.preprocess(y)
        image = self.downscale(I)
        if self.blur and random.random() > .5:
            image = image.filter(ImageFilter.GaussianBlur(radius=1))
        I = self.totensor(I)
        image = self.totensor(image)
        return {"low":image.cuda(), "high":I.cuda()}

def training_loop(model, dataloader, path, name, lr1=1e-3, lr2=1e-4, epoch=100, resume=-1):
    model.cuda()
    mse = nn.MSELoss().cuda()
    optimizer = optim.Adam([
        {'params': model.first_part.parameters(), 'lr': lr1},
        {'params': model.mid_part.parameters(), 'lr': lr1},
        {'params': model.last_part.parameters(), 'lr': lr2},
    ])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=.1)
    epo, batch = 0, 0
    last_epoch_batch = 0
    if resume >= 0:
        epo, batch = load_ckpt(path, name, model, optimizer, lr_scheduler, epoch=resume)
        print(f'restart after epoch {epo}')
    print(f'model is training: {model.training}')
    for ep in range(epo, epoch):
        epoch_loss = 0.
        lrs = [params['lr'] for params in optimizer.state_dict()['param_groups']]
        for sample in dataloader:
            inp = sample["low"]
            gt = sample["high"]
            model.train()
            outp = model(inp)
            outp = torch.clamp(outp, 0., 1.)
            loss = mse(gt, outp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            batch += 1
        lr_scheduler.step()
        print(f"Ep.{ep+1} - loss: {epoch_loss/(batch-last_epoch_batch):.6f} - lr: {lrs}")
        last_epoch_batch = batch
        save_ckpt(path, name, model, optimizer, lr_scheduler, ep+1, batch)
        print(f'Ep.{ep+1} - model saved')
        torch.cuda.empty_cache()

if __name__ == '__main__':
    dataset = SuperResolutionDataset('data')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    fsrcnn = FSRCNN(1, 1)
    training_loop(fsrcnn, dataloader, 'checkpoints', 'fsrcnn')