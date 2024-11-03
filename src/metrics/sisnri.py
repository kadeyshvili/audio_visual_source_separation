import torch
from src.metrics.base_metric import BaseMetric
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class SISNRiMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_snr = ScaleInvariantSignalNoiseRatio()

    def __call__(self, estimated, s1, s2, mix, **kwargs):
        #TODO: change logic to track the metric separately for s1 and s2
        sisnri_s = []
        for est, target_s1, target_s2, mix in zip(estimated, s1, s2, mix):

            sisnr1 = self.si_snr(est[0], target_s1)
            sisnr2 = self.si_snr(est[1], target_s2)

            sisnr1m = self.si_snr(est[0], mix)
            sisnr2m = self.si_snr(est[1], mix)

            sisnri_s.append(((sisnr1 - sisnr1m) + (sisnr2 - sisnr2m)) / 2)
        return sum(sisnri_s) / len(sisnri_s)



