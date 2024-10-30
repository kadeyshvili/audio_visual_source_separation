import torch
from torch import nn
from torch.nn import functional as F


class Conv1dSeparationBlock(nn.Module):
    def __init__(self, embed_channel_size, block_channel_size, skip_connection_channel_size, 
                 kernel_size, dilation, padding) -> None:
        super().__init__()
        embed_conv = nn.Conv1d(embed_channel_size, block_channel_size, 1)
        first_nonlinearity = nn.PReLU()
        first_norm = nn.GroupNorm(1, block_channel_size, eps=1e-10)
        autoregressive_conv = nn.Conv1d(block_channel_size, block_channel_size, kernel_size, 
                                        groups=block_channel_size, dilation=dilation, padding=padding)
        second_nonlinearity = nn.PReLU()
        second_norm = nn.GroupNorm(1, block_channel_size, eps=1e-10)
        self.skip_connection_head = nn.Conv1d(block_channel_size, skip_connection_channel_size, 1)
        self.output_head = nn.Conv1d(block_channel_size, embed_channel_size, 1)
        self.block_body = nn.Sequential(
            embed_conv,
            first_nonlinearity,
            first_norm,
            autoregressive_conv,
            second_nonlinearity,
            second_norm
        )
    
    def forward(self, x):
        block_embed = self.block_body(x)
        skip_connection = self.skip_connection_head(block_embed)
        residual = self.output_head(block_embed) + x
        return residual, skip_connection


class TCN(nn.Module):
    def __init__(self, input_channel_size, embed_channel_size, block_channel_size, 
                 skip_connection_channel_size, output_channel_size,
                 n_blocks, n_repeats, kernel_size) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_channel_size)
        self.input_conv = nn.Conv1d(input_channel_size, embed_channel_size, 1)
        self.repeats = self._init_repeats(n_repeats=n_repeats, n_blocks=n_blocks, embed_channel_size=embed_channel_size, 
                                          block_channel_size=block_channel_size, 
                                          skip_connection_channel_size=skip_connection_channel_size, kernel_size=kernel_size)
        self.activation = nn.PReLU()
        self.output_conv = nn.Conv1d(skip_connection_channel_size, output_channel_size, 1)

    @staticmethod
    def _init_blocks(n_blocks, **block_params):
        return [Conv1dSeparationBlock(**block_params, dilation=(2 ** i), padding=(2 ** i)) 
                for i in range(n_blocks)]
    
    @staticmethod
    def _init_repeats(n_repeats, **blocks_params):
        cur_layers_arr = []
        for _ in range(n_repeats):
            cur_layers_arr.extend(TCN._init_blocks(**blocks_params))
        return nn.ModuleList(cur_layers_arr)
    
    def forward(self, x):
        self.repeats_input = self.input_conv(self.layer_norm(x.transpose(1, 2)).transpose(1, 2))
        cur_input = self.repeats_input
        skip_connection = torch.zeros(self.repeats_input.shape)
        for i in range(len(self.repeats)):
            cur_input, skip_connection_cur = self.repeats[i](cur_input)
            skip_connection = skip_connection + skip_connection_cur
        return self.output_conv(self.activation(skip_connection))

        

class ConvTasNet(nn.Module):
    def __init__(self, input_window_size, latent_channel_size, embed_channel_size, block_channel_size, 
                 skip_connection_channel_size, n_blocks, n_repeats, kernel_size, sample_rate, num_speakers=2) -> None:
        super().__init__()
        MILLISECONDS_IN_SECOND = 1000
        self.input_window_size = input_window_size * sample_rate // MILLISECONDS_IN_SECOND

        self.encoder = nn.Conv1d(1, latent_channel_size, self.input_window_size, stride=(self.input_window_size // 2), bias=False)
        self.tcn = TCN(latent_channel_size, embed_channel_size, block_channel_size, 
                       skip_connection_channel_size, num_speakers * latent_channel_size,
                       n_blocks, n_repeats, kernel_size)
        self.decoder = nn.ConvTranspose1d(latent_channel_size, 1, self.input_window_size, stride=(self.input_window_size // 2), bias=False)
        self.num_speakers = num_speakers
        self.encoder_dim = latent_channel_size
        self.padding_const = (self.input_window_size // 2)


    def get_residual_size(self, mix):
        assert len(mix.shape) in [1, 2], "Either single audio or batch of audios is supported"
        if len(mix.shape) == 1:
            mix = mix.unsqueeze(0)
        _, L = mix.shape
        return self.input_window_size - (self.padding_const + L % self.input_window_size) % self.input_window_size


    def pad_sequence(self, mix):
        """
        Padding audios for correct encoding procedure

        Args:
            mix audio (Tensor): (B x L)
        Returns:
            padded mix audio (Tensor): (B x L)
        """
        assert len(mix.shape) in [1, 2], "Either single audio or batch of audios is supported"
        if len(mix.shape) == 1:
            mix = mix.unsqueeze(0)
        B, _ = mix.shape
        residual = self.get_residual_size(mix)
        padding_amount = self.padding_const
        if residual > 0:
            residual_pad = torch.zeros(B, residual, requires_grad=True).type(mix.type())
            mix = torch.cat([mix, residual_pad], -1)
        padding_1 = torch.zeros(B, padding_amount, requires_grad=True).type(mix.type())
        padding_2 = torch.zeros(B, padding_amount, requires_grad=True).type(mix.type())
        return torch.cat([padding_1, mix, padding_2], -1)


    def forward(self, mix, **batch):
        """
        Model forward method.

        Args:
            mix audio (Tensor): (B x L)
        Returns:
            output (dict): output dict containing estimated sources. ("estimated" : Tensor B x 2 x L)
        """
        assert len(mix.shape) in [1, 2], "Either single audio or batch of audio is supported"
        if len(mix.shape) == 1:
            mix = mix.unsqueeze(0)
        padded_mix = self.pad_sequence(mix)
        residual_size = self.get_residual_size(mix)
        audio_embed = F.relu(self.encoder(padded_mix.unsqueeze(1)))
        B = audio_embed.size(0)
        masks = torch.sigmoid(self.tcn(audio_embed)).view(B, self.num_speakers, self.encoder_dim, -1)
        masked_embed = audio_embed.unsqueeze(1) * masks
        masked_output = self.decoder(masked_embed.view(B * self.num_speakers, self.encoder_dim, -1))
        unpadded_masked_output = masked_output[:, :, self.padding_const:-(residual_size + self.padding_const)].contiguous()
        total_output = unpadded_masked_output.view(B, self.num_speakers, -1)
        return {"estimated": total_output}
