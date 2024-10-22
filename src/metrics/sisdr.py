import torch
from src.metrics.base_metric import BaseMetric

from src.metrics.utils import calc_si_sdr

class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, estimated, s1, s2, **kwargs):
        #TODO: change logic to track the metric separately for s1 and s2
        sisdrs = []
        for est, target_s1, target_s2 in zip(estimated, s1, s2):
            sisdrs.append(calc_si_sdr(est[0], target_s1))
            sisdrs.append(calc_si_sdr(est[1], target_s2))
        return sum(sisdrs) / len(sisdrs)
