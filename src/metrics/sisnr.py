import torch
from src.metrics.base_metric import BaseMetric
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class SISNRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = ScaleInvariantSignalNoiseRatio()

    def __call__(self, estimated, s1, s2, **kwargs):
        #TODO: change logic to track the metric separately for s1 and s2
        sisnrs = []
        for est, target_s1, target_s2 in zip(estimated, s1, s2):
            sisnrs.append(self.metric(est[0], target_s1))
            sisnrs.append(self.metric(est[1], target_s2))
        return sum(sisnrs) / len(sisnrs)
