import math
import torch
from torch import nn
from torch.nn import functional as F


from .encoder_decoder import AudioEncoder, AudioDecoder
from .fusion import FusionNetwork
from .utils import GlobalLayerNorm
from ..lipreading import VideoEmbedding


class CTCNet(nn.Module):
    def __init__(self, 
                 input_window_size=21,
                 latent_channels=512,
                 mouth_input_channels=1,
                 mouth_3d_output_channels=64,
                 mouth_latent_channels=512,
                 num_joint_repeats=3,
                 num_audio_repeats=13,
                 audio_block_input_channels=512,
                 audio_block_hidden_channels=512,
                 audio_block_upsampling_depth=4,
                 audio_block_kernel_size=5,
                 audio_block_norm_type=GlobalLayerNorm,
                 audio_block_activation_type=nn.PReLU,
                 video_block_input_channels=64,
                 video_block_hidden_channels=64,
                 video_block_upsampling_depth=4,
                 video_block_kernel_size=3,
                 video_block_norm_type=nn.BatchNorm1d,
                 video_block_activation_type=nn.PReLU,
                 mask_activation=nn.ReLU,
                 num_speakers=1
                 ) -> None:
        super().__init__()
        self.audio_encoder = AudioEncoder(input_window_size, latent_channels)
        self.video_emb_extractor = VideoEmbedding(mouth_input_channels, mouth_3d_output_channels, mouth_latent_channels)
        self.fusion_module = FusionNetwork(
            latent_channels,
            mouth_latent_channels,
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
            mask_activation
        )
        self.decoder = AudioDecoder(input_window_size, latent_channels)

        sizes_product = latent_channels // 2 * 2 ** audio_block_upsampling_depth
        sizes_gcd = math.gcd(input_window_size // 2, 2 ** audio_block_upsampling_depth)

        self.padding_const = sizes_product // sizes_gcd
        self.encoder_dim = latent_channels
        self.num_speakers = num_speakers

    
    def pad(self, mix):
        B, L = mix.shape
        residual = L % self.padding_const
        if residual > 0:
            residual_pad = torch.zeros(B, residual, requires_grad=True).type(mix.type())
            return torch.cat([mix, residual_pad], -1), True
        return mix, False


    def forward(self, mix, mouth, **batch):
        assert len(mix.shape) in [1, 2], "Either single audio or batch of audios is supported"
        if len(mix.shape) == 1:
            mix = mix.unsqueeze(0)
            mouth = mouth.unsqueeze(0)
        B, L = mix.shape
        padded_mix, was_padded = self.pad(mix)
        audio_embed = self.audio_encoder(padded_mix.unsqueeze(1))
        with torch.no_grad():
            mouth_embedding = self.video_emb_extractor(mouth.transpose(1, 2))
        mask = self.fusion_module(audio_embed, mouth_embedding)
        masked_embed = audio_embed.unsqueeze(1) * mask
        masked_output = self.decoder(masked_embed.view(B * self.num_speakers, self.encoder_dim, -1))
        if was_padded:
            masked_output = masked_output[:, :, :L].contiguous()
        total_output = masked_output.view(B, self.num_speakers, -1)
        if self.num_speakers == 1:
            total_output = total_output.squeeze(1)
        return {"estimated": total_output}


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
