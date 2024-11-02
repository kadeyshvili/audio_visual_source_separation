import torch
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from src.metrics.base_metric import BaseMetric

class PESQMetric(BaseMetric):
    def __init__(self, target_sr=16000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = PerceptualEvaluationSpeechQuality(target_sr, mode='wb')

    def __call__(self, estimated, s1, s2, **kwargs):
        #TODO: change logic to track the metric separately for s1 and s2
        pesqs = []
        for est, target_s1, target_s2 in zip(estimated, s1, s2):
            pesqs.append(self.metric(est[0], target_s1))
            pesqs.append(self.metric(est[1], target_s2))
        return sum(pesqs) / len(pesqs)
