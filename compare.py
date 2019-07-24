import numpy as np
import cv2

origin = cv2.imread("sakura.png", 0)
test = cv2.imread("sakura_half.png")
output = cv2.imread("output.png", 0)
target = cv2.imread("bigjpg.png", 0)

x2_origin = cv2.resize(test, (636,1000), interpolation=cv2.INTER_CUBIC)
x2_origin = cv2.cvtColor(x2_origin, cv2.COLOR_BGR2GRAY)

def PSNR(inp, tar):
    peak = inp.max()
    mse = np.sum((inp-tar)**2)/inp.size
    return 10 * (2 * np.log10(peak) - np.log10(mse))

print(f"model output: {PSNR(origin, output)}")
print(f"cubic output: {PSNR(origin, x2_origin)}")
print(f"bigjpg output: {PSNR(target, origin)}")