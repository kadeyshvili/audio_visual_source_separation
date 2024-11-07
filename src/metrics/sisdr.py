import torch
from src.metrics.base_metric import BaseMetric

from src.metrics.utils import calc_si_sdr

class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, estimated, **batch):

        if batch.get('s1') is not None and batch.get('s2') is not None:
            s1 = batch['s1']
            s2 = batch['s2']
            sisdrs = []
            for est, target_s1, target_s2 in zip(estimated, s1, s2):
                sisdrs.append(calc_si_sdr(est[0], target_s1))
                sisdrs.append(calc_si_sdr(est[1], target_s2))
            return sum(sisdrs) / len(sisdrs)
        
        elif batch.get('target') is not None:
            targets = batch['target']
            sisdrs = []
            for est, target in zip(estimated, targets):
                sisdrs.append(calc_si_sdr(est, target))
            return sum(sisdrs) / len(sisdrs)
        
        else:
            return -torch.inf
