import torch.nn as nn
import torch

from src.model.tdavss.video import VideoEmbedding, VideoEncoder
from src.model.tdavss.audio import AudioEncoder, AudioSeparationBlock, Decoder
from src.model.tdavss.gln import GlobalLayerNorm


class Concat(nn.Module):
    """
    Audio and Visual parts fusion
    """

    def __init__(self, audio_channels, video_channels, out_channels):
        super(Concat, self).__init__()
        self.audio_channels = audio_channels
        self.video_channels = video_channels
        self.conv1d = nn.Conv1d(audio_channels + video_channels, out_channels, 1)

    def forward(self, a, v):

        if a.shape[1] != self.audio_channels or v.shape[1] != self.video_channels:
            raise RuntimeError(f"Dimention 1 mismatch for audio and video features, \
                               expected {self.audio_channels}, {self.video_channels}, \
                                but got  {a.shape[1]}, {v.shape[1]} ")
                                   
        # up-sample video features
        v = torch.nn.functional.interpolate(v, size=a.size(-1))
        y = torch.cat([a, v], dim=1)
        # # position-wise projection
        return self.conv1d(y)
    
    

class Separation(nn.Module):
    """
    Separation network: sequential dilated conv blocks, 
                        concatenation of audio and video features, 
                        sequential dilated conv blocks
    """
    def __init__(
        self,
        audio_encoder_out_channels=256,
        res_conv_channels=256,
        skipcon_conv_channels=256,
        separation_conv_out_channels=512,
        separation_conv_kernel_size=3,
        audio_subblocks=8,
        video_encoder_out_channels=256,
        fused_out_channels=256,
        audio_sep_repeats=1,
        fused_sep_repeats=3,
        skip_connection=False):
        super(Separation, self).__init__()

        self.audio_conv = AudioSeparationBlock(
            repeats=audio_sep_repeats,
            blocks=audio_subblocks,
            in_channels=res_conv_channels,
            out_channels=separation_conv_out_channels,
            res_conv_channels=res_conv_channels,
            skipcon_conv_channels=skipcon_conv_channels,
            kernel_size=separation_conv_kernel_size,
            skip_connection=skip_connection)
        self.concat = Concat(res_conv_channels, video_encoder_out_channels, fused_out_channels)
        self.feats_conv = AudioSeparationBlock(
            fused_sep_repeats,
            audio_subblocks,
            in_channels=res_conv_channels,
            out_channels=separation_conv_out_channels,
            res_conv_channels=res_conv_channels,
            skipcon_conv_channels=skipcon_conv_channels,
            kernel_size=separation_conv_kernel_size,
            skip_connection=skip_connection)
                
        self.mask = nn.Conv1d(fused_out_channels, audio_encoder_out_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, a, v):

        a = self.audio_conv(a)
        f = self.concat(a, v)
        f = self.feats_conv(f)
        m = self.relu(self.mask(f))

        return m


class TDAVSS(nn.Module):
    """
    Time Domain Audio Visual Speech Separation
    Video feature extraction
        video_frontend3d_in_channels: 
        video_frontend3d_out_channels:
        video_embed_dim: 
    Video Encoder
        (video_encoder_in_channels = video_embed_dim)
        video_encoder_out_channels: Number of channels in convolutional blocks
        video_encoder_kernel_size: Kernel size in convolutional blocks
        video_encoder_Drepeats:  Number of repeats
    Audio Encoder
        audio_encoder_out_channels: Number of ﬁlters 
        audio_encoder_kernel_size: Kernel size
    Separation network
        separation_res_conv_channels: Number of channels in the residual paths conv blocks
        separation_skipcon_conv_channels: Number of channels in the skip-connection paths conv blocks
        separation_conv_out_channels: Number of channels in convolutional blocks
        separation_conv_kernel_size: Kernel size in convolutional blocks
        separation_audio_subblocks: Number of convolutional blocks in each repeat   
        fused_out_channels: Number of channels after fusion process
        audio_sep_repeats: Number of repeats in audio part
        fused_sep_repeats: Number of repeats after fusion
    """
    def __init__(
            self,
            video_frontend3d_in_channels=1, 
            video_frontend3d_out_channels=64, 
            video_embed_dim=256, 
            video_encoder_out_channels=256,
            video_encoder_kernel_size=3,
            video_encoder_Drepeats=5,
            audio_encoder_out_channels=256,
            audio_encoder_kernel_size=40,
            separation_res_conv_channels=256,
            separation_skipcon_conv_channels=256,
            separation_conv_out_channels=512,
            separation_conv_kernel_size=3,
            separation_audio_subblocks=8,
            fused_out_channels=256,
            audio_sep_repeats=1, 
            fused_sep_repeats=3,
            skip_connection=False):
        super(TDAVSS, self).__init__()

        self.skip_connection = skip_connection

        self.video_emb_extractor = VideoEmbedding(video_frontend3d_in_channels, video_frontend3d_out_channels,
                 video_embed_dim)

        self.video_encoder = VideoEncoder(video_embed_dim, video_encoder_out_channels, 
                                          video_encoder_kernel_size, video_encoder_Drepeats, 
                                          skip_connection)

        self.audio_encoder = AudioEncoder(1, audio_encoder_out_channels, audio_encoder_kernel_size, 
                                          stride=audio_encoder_kernel_size // 2)
        
        self.cln = GlobalLayerNorm(audio_encoder_out_channels)
        self.conv1x1 = nn.Conv1d(audio_encoder_out_channels, separation_res_conv_channels, 1)

        self.separation = Separation(audio_encoder_out_channels, separation_res_conv_channels, 
                                     separation_skipcon_conv_channels, separation_conv_out_channels,
                                     separation_conv_kernel_size, separation_audio_subblocks, 
                                     video_encoder_out_channels, fused_out_channels, 
                                     audio_sep_repeats, fused_sep_repeats, skip_connection)

        self.decoder = Decoder(
            audio_encoder_out_channels, 1, kernel_size=audio_encoder_kernel_size, 
            stride=audio_encoder_kernel_size // 2, bias=True)

    def forward(self, mix, mouth, **batch):
        """
        mix: wav mixes
        mouth: lip video tensors
        """

        v = self.video_emb_extractor(mouth.transpose(1, 2))
        
        # only one utterance during inference
        if mix.dim() == 1:
            mix = torch.unsqueeze(mix, 0)
            v = torch.unsqueeze(v, 0)

        v = self.video_encoder(v)
        w = self.audio_encoder(mix)
        a = self.conv1x1(self.cln(w))
        m = self.separation(a, v)
        est_source = self.decoder(w * m)

        return {"estimated" : est_source}
    
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info