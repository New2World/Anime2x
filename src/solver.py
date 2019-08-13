import os, argparse
import cv2
import torch
import numpy as np
from PIL import Image

from fsrcnn import FSRCNN
from tools import load_ckpt

class SuperImage:
    def __run(self, model, image):
        model.eval()
        image = image / 255.
        if torch.cuda.is_available():
            model.cuda()
            image = image.cuda()
        outp = model(image)
        outp = torch.clamp(outp, 0., 1.)
        outp = outp.detach().cpu().numpy().squeeze()
        return (outp * 255.).astype(np.uint8)
    
    def __load_image(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
            y, cr, cb = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb))
        elif isinstance(image, Image):
            image = np.array(image)
            y, cr, cb = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb))
        return y, cr, cb
    
    def scale2x(self, model, image, median_blur=True):
        y, cr, cb = self.__load_image(image)
        h, w = y.shape
        cr = cv2.resize(cr, (2*w,2*h), cv2.INTER_LANCZOS4)
        cb = cv2.resize(cb, (2*w,2*h), cv2.INTER_LANCZOS4)
        y = torch.from_numpy(y[np.newaxis,np.newaxis,:,:]).type(torch.FloatTensor)
        outp = self.__run(model, y)
        image2x = np.stack((outp, cr, cb), axis=2)
        image2x = cv2.cvtColor(image2x, cv2.COLOR_YCrCb2BGR)
        if median_blur:
            image2x = cv2.medianBlur(image2x, 3)
        return image2x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', dest='input', type=str, required=True, help='source image to be amplified')
    parser.add_argument('--output', '-o', dest='output', type=str, default='./output.png', help='image output path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    solver = SuperImage()
    fsrcnn = FSRCNN(1, 1)
    load_ckpt('checkpoints', 'fsrcnn', fsrcnn)
    image = solver.scale2x(fsrcnn, args.input)
    cv2.imwrite(args.output, image)