import skimage.io as si
import numpy as np
import torch

import model

c_mean = [0.513, 0.476, 0.471]
c_std = [0.367, 0.355, 0.348]

def read_zero_mean_image(image_name):
    image = si.imread(image_name).transpose(2,0,1)/255.
    # for i in range(3):
    #     image[i] = (image[i]-c_mean[i])/c_std[i]
    image = image[np.newaxis,:,:,:]
    return image

fsrcnn = model.FSRCNN().eval()
fsrcnn.load_state_dict(torch.load(f"checkpoints/fsrcnn_ep10.pt"))
image = torch.from_numpy(read_zero_mean_image("sakura_half.png")).type(torch.FloatTensor)

if torch.cuda.is_available():
    fsrcnn.cuda()
    image = image.cuda()

outp = fsrcnn(image)
outp = torch.clamp(outp)*255.
outp = outp.detach().cpu().numpy().squeeze()
# for i in range(3):
#     outp[i] = (outp[i]*c_std[i]+c_mean[i])*255
outp = outp.transpose(1,2,0)

si.imsave("output.png", outp.astype(np.uint8))