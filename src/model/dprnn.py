import torch
from torch import nn
import torch.nn.functional as F
    

class LayerNormalization(nn.Module):
    def __init__(self, channel_size, eps = 1e-9):
        super().__init__()
        self.channel_size = channel_size
        self.eps = eps
        self.z = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.r = nn.Parameter(torch.ones(channel_size), requires_grad=True)
    
    def forward(self, x):
        dims = torch.arange(1, len(x.shape)).tolist()
        tuple_dims = tuple(dims)
        x_mean = torch.mean(x, dim=tuple_dims, keepdim=True)
        x_var_squared = torch.var(x, dim=tuple_dims, keepdim=True, correction=0)
        normalized_x = (x - x_mean) / torch.sqrt((x_var_squared + self.eps))
        return (normalized_x.transpose(1, -1) * self.z + self.r).transpose(1, -1)


class DPRNNBlock(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout=0, bidirectional=True):
        super().__init__()
        self.intra_rnn = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, dropout=dropout, batch_first=True,bidirectional=bidirectional)
        linear_input_size = hidden_size if not bidirectional else 2 * hidden_size
        self.intra_linear = nn.Linear(linear_input_size, feature_size)
        self.inter_rnn = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, dropout=dropout, batch_first=True,bidirectional=bidirectional)
        self.inter_linear = nn.Linear(linear_input_size, feature_size)
        self.intra_norm = LayerNormalization(feature_size)
        self.inter_norm = LayerNormalization(feature_size)

    def forward(self, x):
        B, N, K, S = x.shape
        saved_for_residual = x
        x = x.permute(0, 3, 2, 1).reshape(B * S, K, N)
        intra_rnn_out_x, _ = self.intra_rnn(x)
        intra_linear_out_x= self.intra_linear(intra_rnn_out_x).reshape(B, S, K, N).permute(0, 3, 2, 1)
        intra_norm_out_x = self.intra_norm(intra_linear_out_x)
        after_residual_connection = saved_for_residual + intra_norm_out_x
        inter_x = after_residual_connection.permute(0, 2, 1, 3).permute(0, 1, 3, 2).reshape(B * K, S, N)
        inter_rnn_out_x, _ = self.inter_rnn(inter_x)
        inter_linear_out_x = self.inter_linear(inter_rnn_out_x).reshape(B, K, S, N).permute(0, 3, 2, 1).permute(0, 1, 3, 2).contiguous()
        inter_norm_out_x = self.inter_norm(inter_linear_out_x)
        output = after_residual_connection + inter_norm_out_x
        return output


class DPRNN(nn.Module):
    def __init__(self, input_size, feature_size=32, hidden_size=32, chunk_length=200, n_repeats=1, bidirectional=True, dropout=0):
        super().__init__()
        
        self.input_size = input_size
        self.feature_size = feature_size
        self.chunk_length = chunk_length
        self.hop_length = chunk_length // 2
        linear_norm = LayerNormalization(input_size)
        start_conv1d = nn.Conv1d(input_size, feature_size, 1)
        self.bottleneck = nn.Sequential(linear_norm, start_conv1d)

        dprnn_blocks = []
        for _ in range(n_repeats):
            dprnn_blocks += [DPRNNBlock(feature_size=feature_size, hidden_size=hidden_size, dropout=dropout, bidirectional=bidirectional)]
        self.dprnn_blocks = nn.Sequential(*dprnn_blocks)

        self.prelu = nn.PReLU()
        self.conv2d = nn.Conv2d(feature_size, feature_size * 2, kernel_size=1)
        self.output_conv = nn.Conv1d(feature_size, input_size, 1)

    def forward(self, x):
        B, N, L = x.shape
        bottleneck_x = self.bottleneck(x)
        kernel_size = (self.chunk_length, 1)
        padding = (self.chunk_length, 0)
        stride = (self.hop_length, 1)
        unfolded_x = F.unfold(bottleneck_x.unsqueeze(-1), kernel_size=kernel_size, padding=padding, stride=stride)
        n_chunks = unfolded_x.shape[-1]
        unfolded_x = unfolded_x.reshape(B, self.feature_size, self.chunk_length, n_chunks)
        drnn_x = self.dprnn_blocks(unfolded_x)
        prelu_x = self.prelu(drnn_x)
        masked_x = self.conv2d(prelu_x)
        masked_x = masked_x.reshape(B * 2, self.feature_size, self.chunk_length, n_chunks)
        twice_batch = masked_x.shape[0]
        to_unfold = self.feature_size * self.chunk_length
        folded_x = F.fold(masked_x.reshape(twice_batch, to_unfold, n_chunks), (L, 1), kernel_size=kernel_size, padding=padding, stride=stride)
        folded_x = folded_x.reshape(twice_batch, self.feature_size, -1)
        out_x = self.output_conv(folded_x)
        out_x = F.relu(out_x)
        out_x = out_x.reshape(B, 2, self.input_size, L)
        return out_x



class DPRNNTasNet(nn.Module):
    def __init__(self, input_size=512, feature_size=256, hidden_size=128, chunk_length=200, kernel_size=2,\
                n_repeats=6, bidirectional=True, dropout=0, num_speakers=2):
        super().__init__()
        self.num_speakers = num_speakers
        self.stride = kernel_size // 2
        self.encoder = nn.Conv1d(in_channels=1, out_channels=input_size, kernel_size=kernel_size, stride=self.stride,groups=1,bias=False)
        self.separation = DPRNN(
            input_size = input_size,
            feature_size = feature_size,
            hidden_size = hidden_size,
            chunk_length = chunk_length,
            n_repeats = n_repeats,
            bidirectional = bidirectional,
            dropout = dropout,
        )
        self.decoder = nn.ConvTranspose1d(in_channels=input_size, out_channels=1, kernel_size=kernel_size, stride=self.stride, bias=False)

    def forward(self, **batch):
        x = batch['mix']
        x = torch.unsqueeze(x, 1)
        encoders = self.encoder(x)
        output = F.relu(encoders)
        masks = self.separation(encoders)
        output = masks * encoders.unsqueeze(1)
        result_mixtures = []
        for i in range(self.num_speakers):
            result = self.decoder(output[:, i, :, :])
            if torch.squeeze(result).dim() == 1:
                result = torch.squeeze(result, dim=1)
            else:
                result = torch.squeeze(result)
            result_mixtures.append(result)
        mixtures = torch.stack(result_mixtures, dim=1)
        return {"estimated": mixtures}
    

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        return result_info
