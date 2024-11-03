import torch
from src.metrics.base_metric import BaseMetric
from torchmetrics.functional.audio.snr import scale_invariant_signal_noise_ratio as sisnr

class SISNRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, estimated, s1, s2, **kwargs):
        #TODO: change logic to track the metric separately for s1 and s2
        sisnrs = []
        for est, target_s1, target_s2 in zip(estimated, s1, s2):
            sisnrs.append(sisnr(est[0], target_s1))
            sisnrs.append(sisnr(est[1], target_s2))
        return sum(sisnrs) / len(sisnrs)
