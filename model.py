import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class feature_encoder(nn.Module):
    def __init__(self):
        super(feature_encoder, self).__init__()
        self.conv_3x3_3_32 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv_3x3_32_32 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv_3x3_32_64 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv_3x3_64_64 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.prelu_1 = nn.PReLU()
        self.prelu_2 = nn.PReLU()
        self.prelu_3 = nn.PReLU()
        self.prelu_4 = nn.PReLU()
        nn.init.kaiming_normal_(self.conv_3x3_3_32.weight)
        nn.init.kaiming_normal_(self.conv_3x3_32_32.weight)
        nn.init.kaiming_normal_(self.conv_3x3_32_64.weight)
        nn.init.kaiming_normal_(self.conv_3x3_64_64.weight)
    
    def forward(self, x):
        x = self.conv_3x3_3_32(x)
        x = self.prelu_1(x)
        x = self.conv_3x3_32_32(x)
        x = self.prelu_2(x)
        x = self.conv_3x3_32_64(x)
        x = self.prelu_3(x)
        x = self.conv_3x3_64_64(x)
        x = self.prelu_4(x)
        return x

class feature_mapping(nn.Module):
    def __init__(self):
        super(feature_mapping, self).__init__()
        self.conv_1x1_64_24 = nn.Conv2d(64, 24, kernel_size=1, padding=0)
        self.conv_3x3_24_24 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.conv_1x1_24_64 = nn.Conv2d(24, 64, kernel_size=1, padding=0)
        self.prelu_1 = nn.PReLU()
        self.prelu_2 = nn.PReLU()
        self.prelu_3 = nn.PReLU()
        nn.init.kaiming_normal_(self.conv_1x1_64_24.weight)
        nn.init.kaiming_normal_(self.conv_3x3_24_24.weight)
        nn.init.kaiming_normal_(self.conv_1x1_24_64.weight)
    
    def forward(self, x):
        x = self.conv_1x1_64_24(x)
        x = self.prelu_1(x)
        for _ in range(3):
            x = self.conv_3x3_24_24(x)
        x = self.prelu_2(x)
        x = self.conv_1x1_24_64(x)
        x = self.prelu_3(x)
        return x

class x2_reconstructor(nn.Module):
    def __init__(self):
        super(x2_reconstructor, self).__init__()
        self.deconv_9x9_64_3 = nn.ConvTranspose2d(64, 3, kernel_size=9, stride=2, padding=4, output_padding=1)
        nn.init.normal_(self.deconv_9x9_64_3.weight, std=1e-3)
    
    def forward(self, x):
        x = self.deconv_9x9_64_3(x)
        return x

class FSRCNN(nn.Module):
    def __init__(self):
        super(FSRCNN, self).__init__()
        self.encoder = feature_encoder()
        self.mapping = feature_mapping()
        self.decoder = x2_reconstructor()
        if torch.cuda.is_available():
            self.encoder.cuda()
            self.mapping.cuda()
            self.decoder.cuda()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.mapping(x)
        x = self.decoder(x)
        return x