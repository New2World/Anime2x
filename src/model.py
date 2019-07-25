import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class resnet_block_basic(nn.Module):
    def __init__(self, n_features):
        super(resnet_block_basic, self).__init__()
        self.conv1 = nn.Conv2d(n_features, n_features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=n_features)
        self.prelu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(n_features,n_features, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=n_features)
        self.prelu2 = nn.LeakyReLU()
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
    
    def forward(self, x):
        outp = self.conv1(x)
        outp = self.bn1(outp)
        outp = self.prelu1(outp)
        outp = self.conv2(outp)
        outp = self.bn2(outp)
        outp += x
        outp = self.prelu2(outp)
        return outp

class feature_encoder(nn.Module):
    def __init__(self):
        super(feature_encoder, self).__init__()
        self.resblock = self._make_block(56, in_features=1)
    
    def _make_block(self, features, layers=2, in_features=None):
        layer = []
        if in_features is None:
            in_features = features
        else:
            layer.append(nn.Conv2d(in_features, features, kernel_size=1))
            nn.init.kaiming_normal_(layer[0].weight)
        for _ in range(layers):
            layer.append(resnet_block_basic(features))
        return nn.Sequential(*layer)
    
    def forward(self, x):
        x = self.resblock(x)
        return x

class feature_mapping(nn.Module):
    def __init__(self):
        super(feature_mapping, self).__init__()
        self.conv1 = nn.Conv2d(56, 12, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(12, 56, kernel_size=1, padding=0)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        for _ in range(4):
            x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        return x

class x2_reconstructor(nn.Module):
    def __init__(self):
        super(x2_reconstructor, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(56, 1, kernel_size=9, stride=2, padding=4, output_padding=1)
        nn.init.normal_(self.deconv1.weight, std=1e-3)
    
    def forward(self, x):
        x = self.deconv1(x)
        return x

class FSRCNN(nn.Module):
    def __init__(self, gpu=True):
        super(FSRCNN, self).__init__()
        self.encoder = feature_encoder()
        self.mapping = feature_mapping()
        self.decoder = x2_reconstructor()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.mapping(x)
        x = self.decoder(x)
        return x