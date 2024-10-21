import torch
from torch import Tensor
from torch.nn import MSELoss 


class MSE(MSELoss):
    def __init__(self):
        super(self, MSELoss).__init__()
        self.mse_loss = MSELoss(reduction='none')

    def forward(self, predicted, **batch) -> Tensor:
        s1 = batch['s1']
        s2 = batch['s2']
        predicted_s1 = predicted[:, 0, :]
        predicted_s2 = predicted[:, 1, :]
        elem_loss_1 = torch.sum((self.mse_loss(predicted_s1, s1) + self.mse_loss(predicted_s2, s2)) / 2, axis=-1)
        elem_loss_2 = torch.sum((self.mse_loss(predicted_s1, s2) + self.mse_loss(predicted_s2, s1)) / 2, axis=-1)
        elem_loss = torch.minimum(elem_loss_1, elem_loss_2)
        total_loss = torch.mean(elem_loss)
        return {"loss" : total_loss}
    