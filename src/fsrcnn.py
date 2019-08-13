import torch
import torch.nn as nn

class FSRCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FSRCNN, self).__init__()
        self.first_part = self.__first()
        self.mid_part = self.__mid()
        self.last_part = self.__last()
    
    def __first(self):
        first = nn.Sequential()
        first.add_module('first_conv1', nn.Conv2d(1, 32, kernel_size=3, padding=1))
        first.add_module('first_prelu1', nn.PReLU())
        first.add_module('first_conv2', nn.Conv2d(32, 32, kernel_size=3, padding=1))
        first.add_module('first_prelu2', nn.PReLU())
        first.add_module('first_conv3', nn.Conv2d(32, 64, kernel_size=3, padding=1))
        first.add_module('first_prelu3', nn.PReLU())
        first.add_module('first_conv4', nn.Conv2d(64, 64, kernel_size=3, padding=1))
        first.add_module('first_prelu4', nn.PReLU())
        for m in first.modules():
            if type(m) is nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)
        return first
    
    def __mid(self):
        mid = nn.Sequential()
        mid.add_module('mid_conv1', nn.Conv2d(64, 16, kernel_size=1))
        mid.add_module('mid_prelu1', nn.PReLU())
        for i in range(4):
            mid.add_module(f'mid_conv{i+2}', nn.Conv2d(16, 16, kernel_size=3, padding=1))
        mid.add_module('mid_conv6', nn.Conv2d(16, 64, kernel_size=1))
        mid.add_module('mid_prelu2', nn.PReLU())
        for m in mid.modules():
            if type(m) is nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)
        return mid
    
    def __last(self):
        last = nn.ConvTranspose2d(64, 1, kernel_size=9, padding=4, stride=2, output_padding=1)
        nn.init.kaiming_normal_(last.weight)
        return last
    
    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x