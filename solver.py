import argparse
import cv2
import numpy as np
import torch
import pathlib

from srcnn.srcnn import SRCNN

# c_mean = [0.513, 0.476, 0.471]
# c_std = [0.367, 0.355, 0.348]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', dest='input', type=str, required=True, help='source image to be amplified')
    parser.add_argument('--output', '-o', dest='output', type=str, default='./output.png', help='image output path')
    parser.add_argument('--gpu', '-g', action='store_true', dest='gpu', help='enable GPU')
    parser.add_argument('--logdir', '-l', dest='logdir', type=str, default='./checkpoints', help='checkpoint directory')
    parser.add_argument('--compare', '-c', action='store_true', dest='cmp', help='print out PSNR and compare different methods')
    parser.add_argument('--bigjpg', '-j', dest='bigjpg', type=str, default=None, help='image amplified by Bigjpg.com')
    parser.add_argument('--origin', '-O', dest='origin', type=str, default=None, help='original high-resolution image')
    parser.set_defaults(gpu=False)
    parser.set_defaults(cmp=False)
    return parser, parser.parse_args()

def check_log(logdir):
    logpath = logdir / 'model.log'
    if logpath.exists():
        return
    epoch, step = 0, 0
    for ckpt in logdir.glob('srcnn_*_*.pt'):
        _, ep, st = str(ckpt)[:-3].split('_')
        ep, st = int(ep), int(st)
        if ep > epoch:
            epoch = ep
            step = st
        elif ep == epoch and st > step:
            step = st
    if epoch == 0:
        raise FileNotFoundError('No checkpoint found')
    with open(logpath, 'w') as f:
        f.write(f'srcnn_{epoch}_{step}.pt')

def load_ckpt(model, logdir):
    logdir = pathlib.Path(logdir)
    check_log(logdir)
    with open(logdir / 'model.log', 'r') as l:
        ckpt_file = l.read(256).strip()
    ckpt = logdir / ckpt_file
    if not ckpt.exists():
        raise FileNotFoundError('No checkpoint found')
    state = torch.load(ckpt)
    model.load_state_dict(state['model_state_dict'])

def solve(model, img, gpu):
    y, u, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YUV))
    y = y / 255.
    h, w = y.shape
    y = cv2.resize(y, (2*w,2*h), cv2.INTER_CUBIC)
    u = cv2.resize(u, (2*w,2*h), cv2.INTER_LANCZOS4)
    v = cv2.resize(v, (2*w,2*h), cv2.INTER_LANCZOS4)
    y = torch.from_numpy(y[np.newaxis,np.newaxis,:,:]).type(torch.FloatTensor)
    if gpu:
        y = y.cuda()
    y = model(y)
    y = torch.clamp(y, 0, 1) * 255.
    y = y.detach()
    if gpu:
        y = y.cpu()
    y = y.numpy().squeeze()
    img = np.array([y, u, v]).astype(np.uint8).transpose(1,2,0)
    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    return img

def PSNR(inp, tar):
    peak = inp.max()
    mse = np.sum((inp-tar)**2)/inp.size
    return 10 * (2 * np.log10(peak) - np.log10(mse))

parser, args = parse_args()
if args.cmp and (args.origin is None or args.bigjpg is None):
    parser.error('--compare requires --origin and --bigjpg')
srcnn = SRCNN(1, 1)
load_ckpt(srcnn, args.logdir)
low_img = cv2.imread(args.input)
x2_img = solve(srcnn, low_img, args.gpu)
cv2.imwrite(args.output, x2_img)
if args.cmp:
    origin = cv2.imread(args.origin, 0)
    bigjpg = cv2.imread(args.bigjpg, 0)
    print(f'model  PSNR: {PSNR(cv2.cvtColor(x2_img, cv2.COLOR_BGR2GRAY), origin)}')
    print(f'bigjpg PSNR: {PSNR(bigjpg, origin)}')
    print(f'cubic  PSNR: {PSNR(cv2.cvtColor(cv2.resize(low_img, cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY), origin)}')