import torch
from torch import nn


class AudioEncoder(nn.Module):
    def __init__(self, input_window_size, hidden_channels):
        super().__init__()
        self.encoder = nn.Conv1d(1, hidden_channels, input_window_size, stride=(input_window_size // 2), bias=False)

    def forward(self, x):
        return self.encoder(x)
    

class AudioDecoder(nn.Module):
    def __init__(self, input_window_size, input_channels) -> None:
        super().__init__()
        self.decoder = nn.ConvTranspose1d(input_channels, 1, input_window_size, stride=(input_window_size // 2), bias=False)

    def forward(self, x):
        return self.decoder(x)
