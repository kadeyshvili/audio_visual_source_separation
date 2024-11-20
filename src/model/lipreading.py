import torch.nn as nn
import torch

"""
the adapted code for BasicBlock, ResNet, Lipreading is taken from
https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
"""
    
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

def downsample_basic_block( inplanes, outplanes, stride ):
    return  nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    

    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation1 = Swish() # or ReLU
        self.activation2 = Swish()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation2(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        self.downsample_block = downsample_basic_block

        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, blocks, stride=1):


        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block( inplanes = self.inplanes,
                                                 outplanes = planes * block.expansion,
                                                 stride = stride )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1) 
        return x


class Lipreading(nn.Module):

    def __init__( self, in_channels=1, out_channels=64):
        super(Lipreading, self).__init__()

        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2])
        self.frontend3D = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                    nn.BatchNorm3d(out_channels),
                    Swish(),
                    nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))


    def threeD_to_2D_tensor(self, x):
        n_batch, n_channels, s_time, sx, sy = x.shape
        x = x.transpose(1, 2)
        return x.reshape(n_batch*s_time, n_channels, sx, sy)

    def forward(self, x):

        B, C, T, H, W = x.size()
        x = self.frontend3D(x)

        Tnew = x.shape[2] 
        x = self.threeD_to_2D_tensor(x)

        x = self.trunk(x)
        return x, (B, Tnew)
    

class VideoEmbedding(nn.Module):
    """
    Extracting features from lip videos.

    in_channels: conv3D in_channels in frontend3D
    out_channels: conv3D out_channels in frontend3D
    embed_dim: video embedding dimention (fc layer after resnet)
    """

    def __init__(self, in_channels=1, out_channels=64,
                 embed_dim=256):
        super(VideoEmbedding, self).__init__()

        self.feature_extractor = Lipreading(in_channels, out_channels )

        self.fc = nn.Linear(512 * BasicBlock.expansion, embed_dim)
        self.bnfc = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        x, (B, Tnew) = self.feature_extractor(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bnfc(x)

        x = x.view(B, Tnew, x.size(1))
        x = x.transpose(1, 2).contiguous()
        return x