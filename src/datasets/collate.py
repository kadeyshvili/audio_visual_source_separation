import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(dataset_items: list[dict]):
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
    mix = []
    spectrogram = []
    s1 = []
    s2 = []
    mix_path = []
    spectrogram_length = []
    for item in dataset_items:
        mix.append(item['mix'].squeeze(0))
        spectrogram_length.append(item['spectrogram'].shape[2])
        spectrogram.append(item['spectrogram'].squeeze(0).permute(1, 0))
        s1.append(item['s1']).squeeze(0)
        s2.append(item['s2']).squeeze(0)
        mix_path.append(item['mix_path'])

    result_batch['mix'] = pad_sequence(mix, batch_first = True)
    result_batch['s1'] = pad_sequence(s1, batch_first = True)
    result_batch['s2'] = pad_sequence(s2, batch_first = True)
    result_batch['spectrogram'] = pad_sequence(spectrogram, batch_first = True).permute(0, 2, 1)
    result_batch['mix_path'] = mix_path
    result_batch['spectrogram_length'] = torch.tensor(spectrogram_length)
    return result_batch