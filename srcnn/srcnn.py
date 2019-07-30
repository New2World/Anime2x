import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SRCNN, self).__init__()
        self.layer1 = self._make_layers(in_channels, 32)
        self.layer2 = self._make_layers(32, 64)
        self.layer3 = self._make_layers(64, 128)
        self.deconv = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.deconv.weight)
    
    def _make_layers(self, in_channels, out_channels):
        layer = nn.Sequential()
        layer.add_module('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layer.add_module('bn1', nn.BatchNorm2d(out_channels))
        layer.add_module('prelu1', nn.PReLU(init=.1))
        layer.add_module('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        layer.add_module('bn2', nn.BatchNorm2d(out_channels))
        layer.add_module('prelu2', nn.PReLU(init=.1))
        for m in layer.modules():
            if type(m) is nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)
        return layer
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.deconv(x)
        return x