from torch import nn
from torch.nn import Sequential


class BaselineModel(nn.Module):
    """
    Simple MLP
    """

    def __init__(self, n_feats, fc_hidden=512):
        """
        Args:
            n_feats (int): number of input features.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.net1 = Sequential(
            # people say it can approximate any function...
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_feats),
        )

        self.net2 = Sequential(
            nn.Linear(in_features=1, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=2)
        )

    def forward(self, mix, **batch):
        """
        Model forward method.

        Args:
            mix audio (Tensor): (B x L)
        Returns:
            output (dict): output dict containing estimated sources. ("estimated" : Tensor B x 2 x L)
        """
        output = self.net1(mix)
        output = output.unsqueeze(1)
        output = self.net2(output.transpose(1, -1)).transpose(1, -1)
        return {"estimated": output}

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
