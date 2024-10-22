import torch
from torch import Tensor
from torch.nn import MSELoss 


class MSE(MSELoss):
    def __init__(self):
        super().__init__()
        self.mse_loss = MSELoss(reduction='none')
        self.permute = []

    def forward(self, estimated, **batch) -> Tensor:
        """
        PIT MSE loss. 
        Calculate the loss for each object in a batch, 
        save the best permutation and return the average loss over batch.
        """
        self.permute = []
        total_loss = []
        for est, s1, s2 in zip(estimated, batch["s1"], batch["s2"]):
            loss, permute = self._calc_pit_mse(est, s1, s2)
            total_loss.append(loss)
            self.permute.append(permute)

        return {"loss" : torch.mean(torch.stack(total_loss))}
    
    def _calc_pit_mse(self, estimated, s1, s2):
        """
        Calculate PIT MSE for one object from a batch.
            predicted.shape = (2, F, L)
        """
        estimated_s1 = estimated[0, :]
        estimated_s2 = estimated[1, :]

        loss_1 = torch.mean((self.mse_loss(estimated_s1, s1) + self.mse_loss(estimated_s2, s2)) / 2)
        loss_2 = torch.mean((self.mse_loss(estimated_s1, s2) + self.mse_loss(estimated_s2, s1)) / 2)

        if loss_1.item() < loss_2.item():
            return loss_1, False
        
        return loss_2, True
        