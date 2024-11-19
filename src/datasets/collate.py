import torch
from torch.nn.utils.rnn import pad_sequence

class Collate:
    def __init__(self, type="full_target"):
        self.full_target = True # full target

        if type != "full_target":
            self.full_target = False # single target

    def __call__(self, dataset_items: list[dict]):
        """
        Collate and pad fields in the dataset items.
        Converts individual items into a batch.

        Args:
            dataset_items (list[dict]): list of objects from
                dataset.__getitem__.
        Returns:
            result_batch (dict[Tensor]): dict, containing batch-version
                of the tensors.
        """
        result_batch = {}
        if self.full_target:
        
            mix = []
            mix_path = []
            s1 = []
            s2 = []
            mouth_s1 = []
            mouth_s2 = []
            
            for item in dataset_items:
                mix.append(item['mix'].squeeze(0))
                mix_path.append(item['mix_path'])
                s1.append(item.get('s1', torch.Tensor([])).squeeze(0))
                mouth_s1.append(item.get('mouths1', torch.Tensor([])))
                s2.append(item.get('s2', torch.Tensor([])).squeeze(0))
                mouth_s2.append(item.get('mouths2', torch.Tensor([])))

            result_batch['mix'] = pad_sequence(mix, batch_first = True)
            result_batch['s1'] = pad_sequence(s1, batch_first = True)
            result_batch['s2'] = pad_sequence(s2, batch_first = True)
            
            result_batch['mouths1'] = torch.stack(mouth_s1, dim=0)
            result_batch['mouths2'] = torch.stack(mouth_s2, dim=0)
            result_batch['mix_path'] = mix_path

        else:
            mix = []
            mix_path = []
            target = []
            mouth = []
            
            for item in dataset_items:
                mix.append(item['mix'].squeeze(0))
                mix_path.append(item['mix_path'])
                target.append(item.get(['target'], torch.Tensor([])).squeeze(0))
                mouth.append(item.get('mouth', torch.Tensor([])))

            result_batch['mix'] = pad_sequence(mix, batch_first = True)
            result_batch['target'] = pad_sequence(target, batch_first = True)
            result_batch['mouth'] = torch.stack(mouth, dim=0)
            result_batch['mix_path'] = mix_path

        return result_batch