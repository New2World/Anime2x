import sys, cv2
import numpy as np
import torch

import model

c_mean = [0.513, 0.476, 0.471]
c_std = [0.367, 0.355, 0.348]

gpu = False

def read_zero_mean_image(image_name):
    y, u, v = cv2.split(cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2YUV))
    image = y[np.newaxis,np.newaxis,:,:] / 255.
    return image, u, v

fsrcnn = model.FSRCNN()
fsrcnn.load_state_dict(torch.load(f"../checkpoints/fsrcnn_{sys.argv[1]}.pt"))
fsrcnn.eval()
inp, u, v = read_zero_mean_image("../sakura_half.png")
image = torch.from_numpy(inp).type(torch.FloatTensor)

if gpu:
    fsrcnn.cuda()
    image = image.cuda()

outp = fsrcnn(image)
outp = torch.clamp(outp, 0, 1)*255.
outp = outp.detach().cpu().numpy().squeeze()
h, w = outp.shape
u = cv2.resize(u, (w, h), cv2.INTER_LANCZOS4)
v = cv2.resize(v, (w, h), cv2.INTER_LANCZOS4)
outp = np.array([outp, u, v]).astype(np.uint8).transpose(1,2,0)
outp = cv2.cvtColor(outp, cv2.COLOR_YUV2BGR)

cv2.imwrite("../output.png", outp.astype(np.uint8))