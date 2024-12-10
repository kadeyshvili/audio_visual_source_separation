import torch
from torch import nn
from torch.nn import functional as F

from .utils import ConvBlock, GlobalLayerNorm


class GeneralSubnetwork(nn.Module):
    def __init__(self, input_channels=128, hidden_channels=512, 
                 upsampling_depth=4, kernel_size=5, norm_type=GlobalLayerNorm, activation_type=nn.PReLU,
                 needs_residual=True):
        super().__init__()
        self.compressors = nn.ModuleList([])
        self.downsamplers = nn.ModuleList([])
        self.fusions = nn.ModuleList([])
        self.depth = upsampling_depth
        self.mapper = ConvBlock(input_channels, hidden_channels, 1, norm_type=norm_type)
        self.compressors.append(ConvBlock(hidden_channels,
                                          hidden_channels,
                                          kernel_size,
                                          1,
                                          hidden_channels,
                                          padding=((kernel_size - 1) // 2),
                                          activation_type=None,
                                          norm_type=norm_type))
        for _ in range(1, self.depth):
            self.compressors.append(ConvBlock(hidden_channels,
                                                hidden_channels,
                                                kernel_size,
                                                2,
                                                hidden_channels,
                                                padding=((kernel_size - 1) // 2),
                                                activation_type=None,
                                                norm_type=norm_type))
        for _ in range(self.depth):
            self.downsamplers.append(ConvBlock(hidden_channels,
                                               hidden_channels,
                                               kernel_size,
                                               2,
                                               hidden_channels,
                                               padding=((kernel_size - 1) // 2),
                                               activation_type=None,
                                               norm_type=norm_type))
        for i in range(self.depth):
            if i == 0 or i == self.depth - 1:
                self.fusions.append(ConvBlock(2 * hidden_channels, hidden_channels,
                                              1, 1, norm_type=norm_type, activation_type=activation_type))
            else:
                self.fusions.append(ConvBlock(3 * hidden_channels, hidden_channels,
                                              1, 1, norm_type=norm_type, activation_type=activation_type))
        self.needs_residual = needs_residual
        self.concat_head = ConvBlock(hidden_channels * self.depth, hidden_channels, 1, 1,
                                     norm_type=norm_type, activation_type=activation_type)
        self.head = nn.Conv1d(hidden_channels, input_channels, 1)

    def forward(self, x):
        residual = x
        mapped_x = self.mapper(x)
        compressed_x = [mapped_x]
        for i in range(self.depth):
            compressed_x.append(self.compressors[i](compressed_x[-1]))
        compressed_x = compressed_x[1:]

        x_fused = []
        for i in range(self.depth):
            x_i_len = compressed_x[i].shape[-1]
            tensors_to_cat = []
            if i - 1 >= 0:
                tensors_to_cat.append(self.downsamplers[i](compressed_x[i - 1]))
            tensors_to_cat.append(compressed_x[i])
            if i + 1 < self.depth:
                tensors_to_cat.append(F.interpolate(compressed_x[i + 1], size=x_i_len, mode="nearest"))
            y = torch.cat(tensors_to_cat, dim=1)
            x_fused.append(self.fusions[i](y))

        total_len = compressed_x[0].shape[-1]
        concat = self.concat_head(torch.cat(list(map(lambda a: F.interpolate(a, size=total_len, mode="nearest"), 
                                                     x_fused)), dim=1))
        result = self.head(concat)
        if self.needs_residual:
            return result + residual
        else:
            return result


class ThalamicNetwork(nn.Module):
    def __init__(self, audio_channels=128, video_channels=128):
        super().__init__()
        self.fc_audio = ConvBlock(audio_channels + video_channels, audio_channels, 1, 1)
        self.fc_video = ConvBlock(audio_channels + video_channels, video_channels, 1, 1)

    def forward(self, audio, video):
        audio_interpolated = F.interpolate(audio, size=video.shape[-1], mode='nearest')
        video_interpolated = F.interpolate(video, size=audio.shape[-1], mode='nearest')
        audio_joint = torch.cat([audio, video_interpolated], dim=1)
        video_joint = torch.cat([audio_interpolated, video], dim=1)
        audio_result = self.fc_audio(audio_joint)
        video_result = self.fc_video(video_joint)
        return audio_result, video_result


class FusionNetwork(nn.Module):
    def __init__(self,
        audio_input_channels,
        video_input_channels,
        num_speakers,
        num_joint_repeats,
        num_audio_repeats,
        audio_block_input_channels,
        audio_block_hidden_channels,
        audio_block_upsampling_depth, 
        audio_block_kernel_size, 
        audio_block_norm_type, 
        audio_block_activation_type,
        video_block_input_channels,
        video_block_hidden_channels,
        video_block_upsampling_depth, 
        video_block_kernel_size, 
        video_block_norm_type, 
        video_block_activation_type,
        mask_activation):
        super().__init__()
        self.num_speakers = num_speakers
        self.num_joint_repeats = num_joint_repeats
        self.num_audio_repeats = num_audio_repeats
        self.audio_input_channels = audio_input_channels
        self.audio_preprocessing = nn.Sequential(nn.PReLU(audio_input_channels), 
                                                 nn.Conv1d(audio_input_channels, audio_block_input_channels, 1, 1))
        self.video_preprocessing = nn.Conv1d(video_input_channels, video_block_input_channels, kernel_size=3, padding=1)
        self.mask_estimator = nn.Sequential(nn.PReLU(), 
                                            nn.Conv1d(audio_block_input_channels, 
                                                      num_speakers * audio_input_channels, 1, 1),
                                            mask_activation())
        self.video_blocks = nn.ModuleList([GeneralSubnetwork(video_block_input_channels, video_block_hidden_channels, 
                                                             video_block_upsampling_depth, video_block_kernel_size, 
                                                             video_block_norm_type, video_block_activation_type, 
                                                             False) for _ in range(num_joint_repeats)])
        self.video_block_concat_nonlinearities = nn.ModuleList([nn.Sequential(
            nn.Conv1d(video_block_input_channels, video_block_input_channels, 1, 1, groups=video_block_input_channels),
            nn.PReLU()
        ) for _ in range(num_joint_repeats)])
        self.audio_block = GeneralSubnetwork(
            audio_block_input_channels, audio_block_hidden_channels, 
            audio_block_upsampling_depth, audio_block_kernel_size, 
            audio_block_norm_type, audio_block_activation_type, True
        )
        self.audio_block_concat_nonlinearities = nn.Sequential(
            nn.Conv1d(audio_block_input_channels, audio_block_input_channels, 1, 1, 
                      groups=audio_block_input_channels), 
            nn.PReLU()
        )
        self.thalamic_networks = nn.ModuleList([
            ThalamicNetwork(audio_block_input_channels, video_block_input_channels) for _ in range(num_joint_repeats)
        ])

    def forward(self, audio, video):
        B, _, T = audio.shape
        audio_preprocessed = self.audio_preprocessing(audio)
        video_preprocessed = self.video_preprocessing(video)

        audio_residual = audio_preprocessed
        video_residual = video_preprocessed
        cur_audio = self.audio_block(audio_preprocessed)
        cur_video = self.video_blocks[0](video_preprocessed)
        cur_audio, cur_video = self.thalamic_networks[0](audio_preprocessed, video_preprocessed)

        for i in range(1, self.num_joint_repeats):
            # print(cur_audio.shape, cur_video.shape)
            cur_audio = self.audio_block(self.audio_block_concat_nonlinearities(audio_residual + cur_audio))
            cur_video = self.video_blocks[i](self.video_block_concat_nonlinearities[i](video_residual + cur_video))
            cur_audio, cur_video = self.thalamic_networks[i](cur_audio, cur_video)

        for _ in range(self.num_audio_repeats):
            cur_audio = self.audio_block(self.audio_block_concat_nonlinearities(audio_residual + cur_audio))

        mask_estimation = self.mask_estimator(cur_audio)
        return mask_estimation.view(B, self.num_speakers, self.audio_input_channels, T)
    