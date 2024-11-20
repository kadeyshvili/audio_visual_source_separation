from pathlib import Path
import torchaudio
from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    def __init__(self, audio_dir, mouths_dir=None, type="full_target", *args, **kwargs):
        self.type = type
        data = []
        for path in Path(audio_dir + "/mix").iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["mix_path"] = str(path)
                mix_info = torchaudio.info(str(path))
                entry["audio_len"] = mix_info.num_frames / mix_info.sample_rate

                entry["s1_path"] = ""
                entry["s2_path"] = ""
                entry["mouths1_path"] = ""
                entry["mouths2_path"] = ""

                if (
                    Path(audio_dir + "/s1").exists()
                    and Path(audio_dir + "/s2").exists()
                ):

                    entry["s1_path"] = audio_dir + "/s1/" + str(path.stem) + ".wav"
                    entry["s2_path"] = audio_dir + "/s2/" + str(path.stem) + ".wav"

                if mouths_dir and Path(mouths_dir).exists():
                    user1, user2 = Path(path).stem.split("_")
                    entry["mouths1_path"] = mouths_dir + "/" + user1 + ".npz"
                    entry["mouths2_path"] = mouths_dir + "/" + user2 + ".npz"

            if len(entry) > 0:

                if self.type != "full_target":
                    entry1 = {'mix_path': entry['mix_path'], 'audio_len': entry['audio_len']}
                    entry1['target_path'] = entry['s1_path']
                    entry1['mouth_path'] = entry['mouths1_path']
                    entry1['speaker_folder'] = 's1'

                    entry2 = {'mix_path': entry['mix_path'], 'audio_len': entry['audio_len']}
                    entry2['target_path'] = entry['s2_path']
                    entry2['mouth_path'] = entry['mouths2_path']
                    entry2['speaker_folder'] = 's2'

                    data.append(entry1)
                    data.append(entry2)

                else:
                    data.append(entry)

        super().__init__(data, *args, **kwargs)