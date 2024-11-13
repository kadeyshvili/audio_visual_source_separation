import torch.nn as nn
import torch


class GlobalLayerNorm(nn.Module):
    """
       Global Layer Normalization learnable per-element affine parameters 
       dim: input shape of a dimention along which to normalize
       eps: small value to add for numerical stability
    """

    def __init__(self, dim, eps=1e-05):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(self.dim, 1))
        self.bias = nn.Parameter(torch.zeros(self.dim, 1))


    def forward(self, x):

        if x.dim() != 3:
            raise RuntimeError("GlobalLayerNorm accepts 3D tensor as input")

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
        x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias

        return x

