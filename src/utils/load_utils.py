import pathlib
import gdown
import os

from src.utils.io_utils import ROOT_PATH

URL_LINKS = {
    "pretrained_resnet": "179NgMsHo9TeZCLLtNWFVgRehDvzteMZE",
    "tdavss": "https://drive.google.com/file/d/1gEA-AwsHwPeOq3kT10JAwiSsd7EM1FWX/view?usp=sharing"
}


def load_pretrained_weights(model, pretrained_dict):
    model_dict = model.state_dict()
    update_dict = {}
    for k, v in pretrained_dict.items():
        part = k.split('.')[0]
        if part == "frontend3D" or part == "trunk":
            k_ = 'feature_extractor.' + k
            update_dict[k_] = v

    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    return model


def download_pretrained_video(video_model_pretrained_path):
    path = ""
    if os.path.isabs(video_model_pretrained_path):
        if os.path.exists(video_model_pretrained_path):
            return
    else:
        absolute_path = os.path.abspath(video_model_pretrained_path)
        if os.path.exists(absolute_path):
            return
        else:
            path = absolute_path

    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    print("Downloading pretrained resnet18...")
    gdown.download(id=URL_LINKS["pretrained_resnet"], output=path)
    print("\nResnet18 downloaded!")
    return path


def download_best_model(pretrained_path=None):
    path = ""
    if os.path.isabs(pretrained_path):
        if os.path.exists(pretrained_path):
            return
    else:
        absolute_path = os.path.abspath(pretrained_path)
        if os.path.exists(absolute_path):
            return
        else:
            path = absolute_path

    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    gdown.download(url=URL_LINKS[pathlib.Path(pretrained_path).stem], output=path, fuzzy=True)
    
