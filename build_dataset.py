import os, cv2
import numpy as np
import torch
import torchvision as tv

image_list = sorted(list(os.walk("General-100"))[0][2])
data_count = 0

image_crop = tv.transforms.Compose([
    tv.transforms.ToPILImage(),
    tv.transforms.FiveCrop(256)
])
image_resize = tv.transforms.Resize(128)
image_vflip = tv.transforms.RandomVerticalFlip(p=1)
image_hflip = tv.transforms.RandomHorizontalFlip(p=1)

for each_image in image_list:
    each_image = os.path.join("General-100", each_image)
    image = cv2.imread(each_image)
    if image.shape[0] < 256 or image.shape[1] < 256:
        os.remove(each_image)
        continue
    image_part = image_crop(image)
    del image
    os.remove(each_image)
    for img in image_part:
        cv2.imwrite(f"General-100/im_{data_count}.png", np.array(img))
        cv2.imwrite(f"data/im_{data_count}.png", np.array(image_resize(img)))
        data_count += 1
        vimg = image_vflip(img)
        cv2.imwrite(f"General-100/im_{data_count}.png", np.array(vimg))
        cv2.imwrite(f"data/im_{data_count}.png", np.array(image_resize(vimg)))
        data_count += 1
        himg = image_hflip(img)
        cv2.imwrite(f"General-100/im_{data_count}.png", np.array(himg))
        cv2.imwrite(f"data/im_{data_count}.png", np.array(image_resize(himg)))
        data_count += 1
    print(f"Image {int(data_count/15):3d}/100 transformed")

print(f"{data_count} training images")