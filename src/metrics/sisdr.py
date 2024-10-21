import torch
from src.metrics.base_metric import BaseMetric

from src.metrics.utils import calc_si_sdr

class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, estimated, targets, **kwargs):
        sisdrs = []
        for est, target in zip(estimated, targets):
            sisdrs.append(calc_si_sdr(est, target))
        return sum(sisdrs) / len(sisdrs)
