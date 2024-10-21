from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    def __init__(self, audio_dir, mouths_dir=None, *args, **kwargs):
        data = []
        for path in Path(audio_dir + "/mix").iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["mix_path"] = str(path)
                mix_info = torchaudio.info(str(path))
                entry["audio_len"] = mix_info.num_frames / mix_info.sample_rate

                entry["s1_path"] = ""
                entry["s2_path"] = ""
                entry["mouth_s1_path"] = ""
                entry["mouth_s2_path"] = ""

                if (
                    Path(audio_dir + "/s1").exists()
                    and Path(audio_dir + "/s2").exists()
                ):
                    entry["s1_path"] = audio_dir + "/s1" + str(path.stem()) + ".wav"
                    entry["s2_path"] = audio_dir + "/s2" + str(path.stem()) + ".wav"

                if mouths_dir and Path(mouths_dir).exists():
                    user1, user2 = path.stem().split("_")
                    entry["mouth_s1_path"] = Path(mouths_dir) / (user1 + ".npz")
                    entry["mouth_s2_path"] = Path(mouths_dir) / (user2 + ".npz")

            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
