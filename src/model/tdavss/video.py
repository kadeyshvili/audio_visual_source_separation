import torch.nn as nn
import torch

from src.model.lipreading import Lipreading, BasicBlock


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
    

    
class VideoConv1D(nn.Module):
    """
    in_channels: video Encoder output channels
    out_channels: depthwise conv channels
    kernel_size: the depthwise conv kernel size
    dilation: the depthwise conv dilation
    residual: Whether to use residual connection
    skip_connection: Whether to use skip connection
    first_block: whether this is the first block
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 residual_connection=True,
                 skip_connection=True,
                 first_block=True
                 ):
        super(VideoConv1D, self).__init__()

        self.first_block = first_block
        self.skip_connection = skip_connection
        self.residual_connection = residual_connection and not first_block
        self.bn = nn.BatchNorm1d(in_channels) if not first_block else None
        self.relu = nn.ReLU() if not first_block else None

        self.depthwise_conv1d = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            dilation=dilation,
            padding=(dilation * (kernel_size - 1)) // 2,
            bias=True)
        
        self.skip_conv1d = nn.Conv1d(in_channels, out_channels, 1)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):

        if not self.first_block:
            y = self.bn(self.relu(x))
            y = self.depthwise_conv1d(y)
        else:
            y = self.depthwise_conv1d(x)

        if self.skip_connection:
            skip = self.skip_conv1d(y)
            if self.residual_connection:
                y = y + x
            return skip, y
        else:
            y = self.conv1d(y)
            if self.residual_connection:
                y = y + x
            return y



class VideoEncoder(nn.Module):
    """
    Video sequential conv1D blocks before fusion with audio.

    in_channels: in_channels in the first Conv1d block
    out_channels: out_channels and in_channels after the first Conv1d block
    kernel_size: kernel size in Conv1d block
    repeat: number of repeats of Conv1d block
    skip_connection: whether to use skip connection
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 repeat=5,
                 skip_connection=True):
        super(VideoEncoder, self).__init__()

        self.conv1d_list = nn.ModuleList([])
        self.skip_connection = skip_connection

        self.conv1d_list.append(
                    VideoConv1D(
                        in_channels,
                        out_channels,
                        kernel_size,
                        skip_connection=self.skip_connection,
                        residual_connection=True,
                        first_block=False))
        
        for _ in range(1, repeat):
            self.conv1d_list.append(
                    VideoConv1D(
                        out_channels,
                        out_channels,
                        kernel_size,
                        skip_connection=self.skip_connection,
                        residual_connection=True,
                        first_block=False))

    def forward(self, x):

        if self.skip_connection:
            out_sc = 0
            for conv_block in self.conv1d_list:
                skip, out = conv_block(x)
                x = out
                out_sc += skip
            return out_sc
        else:
            for conv_block in self.conv1d_list:
                x = conv_block(x)
            return x




