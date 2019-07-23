import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class _encoder(nn.Module):
    def __init__(self):
        super(_encoder, self).__init__()
        self.conv_5x5_56 = nn.Conv2d(3, 56, kernel_size=5, padding=2)
        self.prelu = nn.PReLU()
        nn.init.kaiming_normal_(self.conv_5x5_56.weight)
    
    def forward(self, x):
        x = self.conv_5x5_56(x)
        x = self.prelu(x)
        return x

class _mapping(nn.Module):
    def __init__(self):
        super(_mapping, self).__init__()
        self.conv_1x1_12 = nn.Conv2d(56, 12, kernel_size=1, padding=0)
        self.conv_3x3_12 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.conv_1x1_56 = nn.Conv2d(12, 56, kernel_size=1, padding=0)
        self.prelu_1 = nn.PReLU()
        self.prelu_2 = nn.PReLU()
        self.prelu_3 = nn.PReLU()
        nn.init.kaiming_normal_(self.conv_1x1_12.weight)
        nn.init.kaiming_normal_(self.conv_3x3_12.weight)
        nn.init.kaiming_normal_(self.conv_1x1_56.weight)
    
    def forward(self, x):
        x = self.conv_1x1_12(x)
        x = self.prelu_1(x)
        for _ in range(4):
            x = self.conv_3x3_12(x)
        x = self.prelu_2(x)
        x = self.conv_1x1_56(x)
        x = self.prelu_3(x)
        return x

class _decoder(nn.Module):
    def __init__(self):
        super(_decoder, self).__init__()
        self.deconv_9x9_1 = nn.ConvTranspose2d(56, 3, kernel_size=9, stride=2, padding=4, output_padding=1)
        nn.init.normal_(self.deconv_9x9_1.weight, std=1e-3)
    
    def forward(self, x):
        x = self.deconv_9x9_1(x)
        return x

class FSRCNN(nn.Module):
    def __init__(self):
        super(FSRCNN, self).__init__()
        self.encoder = _encoder()
        self.mapping = _mapping()
        self.decoder = _decoder()
        if torch.cuda.is_available():
            self.encoder.cuda()
            self.mapping.cuda()
            self.decoder.cuda()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.mapping(x)
        x = self.decoder(x)
        return x