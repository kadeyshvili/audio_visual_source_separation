import torch
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from src.metrics.base_metric import BaseMetric

class PESQMetric(BaseMetric):
    def __init__(self, target_sr=16000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = PerceptualEvaluationSpeechQuality(target_sr, mode='wb')

    def __call__(self, estimated, **batch):

        if batch.get('s1') is not None and batch.get('s2') is not None:
            s1 = batch['s1']
            s2 = batch['s2']
            pesqs = []
            for est, target_s1, target_s2 in zip(estimated, s1, s2):
                pesqs.append(self.metric(est[0], target_s1))
                pesqs.append(self.metric(est[1], target_s2))
            return sum(pesqs) / len(pesqs)
        
        elif batch.get('target') is not None:
            targets = batch['target']
            pesqs = []
            for est, target in zip(estimated, targets):
                pesqs.append(self.metric(est, target))
            return sum(pesqs) / len(pesqs)
        
