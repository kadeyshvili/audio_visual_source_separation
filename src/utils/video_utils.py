import pathlib
import gdown
import os

from src.utils.io_utils import ROOT_PATH

URL_LINKS = {
    "pretrained_resnet": "179NgMsHo9TeZCLLtNWFVgRehDvzteMZE",
    "best_model": "",
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


def download_best_model(path=None):
    if path is None:
        data_dir = ROOT_PATH / "data" / "models"
        data_dir.mkdir(exist_ok=True, parents=True)
        path = str(data_dir) + "best_model.pth"
    else:
        dir = path[: -path[::-1].find("/")]
        pathlib.Path(dir).mkdir(exist_ok=True, parents=True)

    print("Downloading best model...")
    gdown.download(id=URL_LINKS["best_model"], output=path)
    print("\nBest model downloaded!")
    return path