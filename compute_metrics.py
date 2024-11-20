import argparse
from pathlib import Path
import torch
import torchaudio

from src.metrics import PESQMetric, SISDRMetric, SISNRMetric, SISNRiMetric


def load_audio(path):
    if path == "" or not Path(path).exists():
        return None
    audio_tensor, sr = torchaudio.load(path)
    audio_tensor = audio_tensor[0:1, :]
    return audio_tensor


def calc_metrics(pred_dir, gt_dir):
    total_pesq = 0
    total_sisdr = 0
    total_sisnr = 0 
    total_sisnri = 0

    pesq_metric = PESQMetric(target_sr=16000)
    sisdr_metric = SISDRMetric()
    sisnr_metric = SISNRMetric()
    sisnri_metric = SISNRiMetric()

    file_count = 0
    for pred_file in Path(pred_dir).glob("*.pth"):
        pred = torch.load(pred_file, map_location=torch.device('cpu'))

        if "predicted_s1" in pred and "predicted_s2" in pred:
            predicted_s1, predicted_s2 = pred["predicted_s1"], pred["predicted_s2"]
            predicted_s1 = predicted_s1.unsqueeze(0)
            predicted_s2 = predicted_s2.unsqueeze(0)
            estimated = torch.stack([predicted_s1, predicted_s2], dim=1)
            
            base_name = pred_file.stem
            gt_s1_path = Path(gt_dir, "s1", f"{base_name}.wav")
            gt_s2_path = Path(gt_dir, "s2", f"{base_name}.wav")
            mix_path = Path(gt_dir, "mix", f"{base_name}.wav")
            
            if not (gt_s1_path.exists() and gt_s2_path.exists()):
                print(f"Ground truth files missing for {base_name}")
                continue
            
            gt_s1 = load_audio(gt_s1_path)
            gt_s2 = load_audio(gt_s2_path)
            mix = load_audio(mix_path) if mix_path.exists() else None

            batch = {"s1": gt_s1, "s2": gt_s2}

            pesq = pesq_metric(estimated, **batch)
            sisdr = sisdr_metric(estimated, **batch)
            sisnr = sisnr_metric(estimated, **batch)
            sisnri = sisnri_metric(estimated, mix=mix, **batch) if mix is not None else None
            total_pesq += pesq.item()
            total_sisdr += sisdr.item()
            total_sisnr += sisnr.item()
            if sisnri is not None:
                total_sisnri += sisnri.item()

            
        elif "estimated" in pred and "speaker_folder" in pred:
            estimated, speaker_folder = pred["estimated"], pred["speaker_folder"]
            estimated = estimated.unsqueeze(0)
            
            base_name = pred_file.stem.split('?')[0]
            gt_path = Path(gt_dir, speaker_folder, f"{base_name}.wav")
            mix_path = Path(gt_dir, "mix", f"{base_name}.wav")
            
            if not gt_path.exists():
                print(f"Ground truth files missing for {base_name}")
                continue

            gt = load_audio(gt_path)
            mix = load_audio(mix_path) if mix_path.exists() else None

            batch = {"target": gt}

            pesq = pesq_metric(estimated, **batch)
            sisdr = sisdr_metric(estimated, **batch)
            sisnr = sisnr_metric(estimated, **batch)
            sisnri = sisnri_metric(estimated, mix=mix, **batch) if mix is not None else None
            total_pesq += pesq.item()
            total_sisdr += sisdr.item()
            total_sisnr += sisnr.item()
            if sisnri is not None:
                total_sisnri += sisnri.item()

        else:
            print("Predicted audio not found.")
            return
        
        file_count += 1

    if file_count == 0:
        print("No files processed.")
        return None
    

    avg_pesq = total_pesq / file_count
    avg_sisdr = total_sisdr / file_count
    avg_sisnr = total_sisnr / file_count
    avg_sisnri = total_sisnri / file_count

    return {
        "Average PESQ": avg_pesq,
        "Average SI-SDR": avg_sisdr,
        "Average SI-SNR": avg_sisnr,
        "Average SI-SNRi": avg_sisnri,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate audio metrics.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory with prediction files")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory with ground truth files and mixture")
    return parser.parse_args()

def main():
    args = parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    average_metrics = calc_metrics(pred_dir, gt_dir)

    if average_metrics:
        for key, value in average_metrics.items():
            print(f'{key} :   {value}')

if __name__ == "__main__":
    main()
