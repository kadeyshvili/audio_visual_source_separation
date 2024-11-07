import torch
from src.metrics.base_metric import BaseMetric
from torchmetrics.functional.audio.snr import scale_invariant_signal_noise_ratio as sisnr


class SISNRiMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, estimated, mix, **batch):
        if batch.get('s1') is not None and batch.get('s2') is not None:
            s1 = batch['s1']
            s2 = batch['s2']
            sisnri_s = []
            for est, target_s1, target_s2, target_mix in zip(estimated, s1, s2, mix):

                sisnr1 = sisnr(est[0], target_s1)
                sisnr2 = sisnr(est[1], target_s2)

                sisnr1m = sisnr(target_s1, target_mix)
                sisnr2m = sisnr(target_s2, target_mix)

                sisnri_s.append(((sisnr1 - sisnr1m) + (sisnr2 - sisnr2m)) / 2)
            return sum(sisnri_s) / len(sisnri_s)
        
        elif batch.get('target') is not None:
            targets = batch['target']
            sisnri_s = []
            for est, target, target_mix in zip(estimated, targets, mix):

                sisnr_est = sisnr(est, target)
                sisnr_mix = sisnr(target, target_mix)
                sisnri_s.append(sisnr_est - sisnr_mix)

            return sum(sisnri_s) / len(sisnri_s)
        
        else:
            return -torch.inf

