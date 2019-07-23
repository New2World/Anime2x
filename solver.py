import cv2
import numpy as np
import torch

import model

c_mean = [0.396, 0.504, 0.570]
c_std = [0.220, 0.219, 0.215]

def read_zero_mean_image(image_name):
    image = cv2.imread(image_name).transpose(2,0,1)/255.
    for i in range(3):
        image[i] = (image[i]-c_mean[i])/c_std[i]
    image = image[np.newaxis,:,:,:]
    return image

fsrcnn = model.FSRCNN()
fsrcnn.load_state_dict(torch.load("checkpoints/fsrcnn_200.pt"))
image = torch.from_numpy(read_zero_mean_image("fantasy.jpg")).type(torch.FloatTensor)

if torch.cuda.is_available():
    fsrcnn.cuda()
    image = image.cuda()

outp = fsrcnn(image)

outp = outp.detach().cpu().numpy().squeeze()
for i in range(3):
    outp[i] = (outp[i]*c_std[i]+c_mean[i])*255
outp = outp.transpose(1,2,0)
outp[outp<0] = 0
outp[outp>255] = 255

cv2.imwrite("output.png", outp.astype(np.uint8))