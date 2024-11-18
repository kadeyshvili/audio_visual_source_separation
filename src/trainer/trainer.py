from pathlib import Path

import pandas as pd
import torch
import pyloudnorm as pyln

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.metrics.utils import calc_si_sdr


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker, batch_num: int = 0):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)
        
        if self.criterion.need_pit:
            batch = self._update_predictions(**batch) # permute speakers

        if self.is_train:
            if self.gradient_accumulation == 1:
                batch["loss"].backward()  # sum of all losses is always called loss
            else:
                (batch["loss"] / self.gradient_accumulation).backward()
            self._clip_grad_norm()
            if (batch_num + 1) % self.gradient_accumulation == 0:
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step()
            
                    
        if not self.is_train:
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(batch['loss'])

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        if self.loudness_norm:
            batch = self._loudness_norm(**batch) # normalize loudness

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _update_predictions(self, **batch):
        """
        Permute the speakers for each object in the batch according to criterion.
        """
        preds = batch["estimated"].clone()
        preds[self.criterion.permute] = torch.flip(batch["estimated"][self.criterion.permute], dims=[1])
        batch["estimated"] = preds
        return batch
    
    def _loudness_norm(self, **batch):
        mix = batch['mix'].clone()
        estimated = batch["estimated"].clone()
        
        if self.dataset_type == "full_target":
            meter = pyln.Meter(self.target_sr)
            for i, (target, est1, est2) in enumerate(zip(mix, estimated[:, 0], estimated[:, 1])):
                target = target.detach().cpu().numpy()
                est1 = est1.detach().cpu().numpy()
                est2 = est2.detach().cpu().numpy()
                loudness = meter.integrated_loudness(target)
                loudness_est1 = meter.integrated_loudness(est1)
                loudness_est2 = meter.integrated_loudness(est2)

                loudness_normalized_est1 = pyln.normalize.loudness(est1, loudness_est1, loudness)
                loudness_normalized_est2 = pyln.normalize.loudness(est2, loudness_est2, loudness)
                estimated[i] = torch.Tensor([loudness_normalized_est1, loudness_normalized_est2])

            batch["estimated"] = estimated.to(self.device)
            return batch
        else:
            meter = pyln.Meter(self.target_sr)
            for i, (target, est) in enumerate(zip(mix, estimated)):
                target = target.detach().cpu().numpy()
                est = est.detach().cpu().numpy()
                loudness = meter.integrated_loudness(target)
                loudness_est = meter.integrated_loudness(est)

                loudness_normalized_est = pyln.normalize.loudness(est, loudness_est, loudness)
                estimated[i] = torch.from_numpy(loudness_normalized_est)

            batch["estimated"] = estimated.to(self.device)
            return batch   
        

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_predictions(**batch)
        else:
            # Log Stuff
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_predictions(
        self, estimated, mix_path, examples_to_log=10, **batch
    ):
        if self.dataset_type == "full_target":
            s1_all = batch['s1']
            s2_all = batch['s2']
            tuples = list(zip(mix_path, estimated, s1_all, s2_all))
            rows = {}
            for path, est, s1, s2 in tuples[:examples_to_log]:
                est_s1 = est[0, :]
                est_s2 = est[1, :]
                sisdr1 = calc_si_sdr(est_s1, s1)
                sisdr2 = calc_si_sdr(est_s2, s2)
                
                rows[Path(path).name] = {
                    "SI-SDR-s1" : sisdr1,
                    "SI-SDR-s2" : sisdr2,
                    "estimated_s1": self.writer.add_audio("estimated_s1", est_s1, 16000),
                    "estimated_s2": self.writer.add_audio("estimated_s2", est_s2, 16000),
                    "target_s1": self.writer.add_audio("target_s1", s1, 16000),
                    "target_s2": self.writer.add_audio("target_s2", s2, 16000)
                }
                self.writer.add_table(
                    "predictions", pd.DataFrame.from_dict(rows, orient="index")
                )
        else:
            
            target_all = batch['target']
            tuples = list(zip(mix_path, estimated, target_all))
            rows = {}
            for path, est, target in tuples[:examples_to_log]:
                
                sisdr = calc_si_sdr(est, target)
                
                rows[Path(path).name] = {
                    "SI-SDR" : sisdr,
                    "estimated": self.writer.add_audio("estimated", est, 16000),
                    "target": self.writer.add_audio("target", target, 16000),
                }
                self.writer.add_table(
                    "predictions", pd.DataFrame.from_dict(rows, orient="index")
                )
