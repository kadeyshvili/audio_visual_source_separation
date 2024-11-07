import torch
from torch import nn, Tensor
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class SISNR(nn.Module):
    def __init__(self, need_pit=True):
        super().__init__()
        self.need_pit = need_pit
        self.si_snr_loss = ScaleInvariantSignalNoiseRatio()
        self.permute = []

    def forward(self, estimated, **batch) -> Tensor:
        """
        PIT SI-SNR loss. 
        Calculate the loss for each object in a batch, 
        save the best permutation and return the average loss over batch.
        """

        if self.need_pit:
            self.permute = []
            total_loss = []
            for est, s1, s2 in zip(estimated, batch["s1"], batch["s2"]):
                loss, permute = self._calc_pit_si_snr(est, s1, s2)
                total_loss.append(loss)
                self.permute.append(permute)
        else:
            total_loss = []
            for est, target in zip(estimated, batch["target"]):
                loss = self.si_snr_loss(est, target)
                total_loss.append(loss)

        return {"loss" : -torch.mean(torch.stack(total_loss))}  # maximizing sisnr -> minimizing -sisnr

    def _calc_pit_si_snr(self, estimated, s1, s2):
        """
        Calculate PIT SI-SNR for one object from a batch.
            predicted.shape = (2, F, L)
        """
        estimated_s1 = estimated[0, :]
        estimated_s2 = estimated[1, :]

        loss_1 = torch.mean((self.si_snr_loss(estimated_s1, s1) + self.si_snr_loss(estimated_s2, s2)) / 2)
        loss_2 = torch.mean((self.si_snr_loss(estimated_s1, s2) + self.si_snr_loss(estimated_s2, s1)) / 2)

        if loss_1.item() > loss_2.item():
            return loss_1, False
        
        return loss_2, True
        