import torch.nn as nn
import torch

from src.model.tdavss.gln import GlobalLayerNorm
    
class AudioEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(AudioEncoder, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels,
                           kernel_size, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.relu(self.conv(x))
        return x


class AudioConv1D(nn.Module):
    """
       in_channels: audio encoder output channels
       out_channels: Conv1D output channels
       res_conv: channels in the residual paths conv blocks
       skipcon_conv: channels in the skip-connection paths conv blocks
       kernel_size: the depthwise conv kernel size
       dilation: the depthwise conv dilation
       skip_connection: (bool) to use skip connection
    """

    def __init__(self,
                 in_channels=256,
                 out_channels=512,
                 res_conv=256,
                 skipcon_conv=256,
                 kernel_size=3,
                 dilation=1,
                 skip_connection=False):
        
        super(AudioConv1D, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1, 1)
        self.relu1 = nn.ReLU()
        self.norm1 = GlobalLayerNorm(out_channels)
        self.pad = (dilation*(kernel_size - 1))//2 
        self.dilatedconv = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               padding=self.pad, dilation=dilation, groups=out_channels)
        self.relu2 = nn.ReLU()
        self.norm2 = GlobalLayerNorm(out_channels)
        self.res_conv = nn.Conv1d(out_channels, res_conv, 1)
        self.skipcon_conv = nn.Conv1d(out_channels, skipcon_conv, 1)
        self.skip_connection = skip_connection

    def forward(self, x):

        out = self.conv1x1(x)
        out = self.relu1(out)
        out = self.norm1(out)
        out = self.dilatedconv(out)
        out = self.relu2(self.norm2(out))

        if self.skip_connection:
            skip = self.skipcon_conv(out)
            res = self.res_conv(out)
            return skip, res+x
        else:
            res = self.res_conv(out)
            return res+x

        
class AudioSeparationBlock(nn.Module):
    """
        repeats: number of repeats of 1D dilated conv block
        blocks: number of blocks in each repeat
        in_channels: audio encoder output channels
        out_channels: Conv1D output channels
        res_conv_channels: channels in the residual paths conv blocks
        skipcon_conv_channels: channels in the skip-connection paths conv blocks
        kernel_size: the depthwise conv kernel size
        skip_connection: (bool) to use skip connection
    """
     
    def __init__(self, repeats, blocks,
                 in_channels=256,
                 out_channels=512,
                 res_conv_channels=256,
                 skipcon_conv_channels=256,
                 kernel_size=3,
                 skip_connection=False):
        super(AudioSeparationBlock, self).__init__()

        self.lists = nn.ModuleList([])
        self.skip_connection = skip_connection
        for _ in range(repeats):
            for d in range(blocks):
                self.lists.append(AudioConv1D(
                 in_channels=in_channels,
                 out_channels=out_channels,
                 res_conv=res_conv_channels,
                 skipcon_conv=skipcon_conv_channels,
                 kernel_size=kernel_size,
                 dilation=(2**d),
                 skip_connection=skip_connection))
                
    def forward(self, x):

        if self.skip_connection:
            out_skip_connection = 0
            for conv_subblock in self.lists:
                skip, out = conv_subblock(x)
                x = out
                out_skip_connection += skip
            return out_skip_connection
        else:
            for conv_subblock in self.lists:
                x = conv_subblock(x)
            return x
        

class Decoder(nn.ConvTranspose1d):

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = super().forward(x)
        x = torch.squeeze(x, dim=1)
        return x