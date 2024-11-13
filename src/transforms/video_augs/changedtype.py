from torchvision.transforms.v2 import ToDtype
from torch import Tensor, nn
import torch

class ChangeDtypeToFloat(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dtype = torch.float32
        self._aug = ToDtype(self.dtype, scale=True, *args, **kwargs)

    def __call__(self, data: Tensor):
        return self._aug(data)
