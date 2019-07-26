import os
import skimage.io as si
import numpy as np

image_list = [os.path.join('../data', image) for image in list(os.walk('../data'))[0][2]]
l = len(image_list)
image_mean = np.zeros((l,3))
image_std = np.zeros((l,3))

for i, image_name in enumerate(image_list):
    image = si.imread(image_name)/255.
    image = np.reshape(image, (-1,3))
    image_mean[i] = np.mean(image, axis=0)
    image_std[i] = np.std(image, axis=0)
    print(f"{i+1}/{l} - {image_name}")

print(f"mean: {np.mean(image_mean, axis=0)}")
print(f"std: {np.mean(image_std, axis=0)}")