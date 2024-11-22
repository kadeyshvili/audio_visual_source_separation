# Audio-visual source separation (AVSS) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/project_avss).
Authors: Vsevolod Kuybida, Polina Kadeyshvili, Anna Markovich

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use
There are implementations for four models: Conv-TasNet, Dual-path RNN, TDAVSS, CTCNet.

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

### How to reproduce the results of the best model (train)

- specify the directories to audio and/or video files in *datasets.train.audio_dir="" datasets.train.mouths_dir="" datasets.val.audio_dir="" datasets.val.mouths_dir=""*

run the following command (fill in the paths to a data set):

```bash
python3 train.py -cn=tdavss  HYDRA_CONFIG_ARGUMENTS
```

## How to run inference

- To evaluate audio-only model specify *-cn "inference"* (default) and for audio-visual models use *-cn "inference_av"* as the first argument of the command
- To evaluate your own pretrained model defined in the project, specify model config as *model=* and the path to it's weights in *from_pretrained=""* (path to the **best model** by default, the weights are loaded from gdrive)
- To evaluate a model on your data set specify the directories to audio and/or video files in *datasets.train.audio_dir="" datasets.train.mouths_dir="" datasets.val.audio_dir="" datasets.val.mouths_dir=""*
- If you want to evaluate the model on less than 8 utterances, specify *dataloader.batch_size=*
- Predicted utterances are saved in *inferencer.save_path / test*

Command for inferencing the best model:

```bash
python3 inference.py -cn "inference_av" HYDRA_CONFIG_ARGUMENTS
```

## How to calculate metrics on predicted utterances

- **pred_dir** is a directory, where there must be two folders: *predicted_s1* and *predicted_s2* with predictions (running inference in this project creates such directory)
- **gt_dir** is a directory of the data set folders "mix", "s1", "s2" (see part [testing](https://github.com/markovka17/dla/tree/2024/project_avss))

```bash
python3 compute_metrics.py --pred_dir <dir> --gt_dir <dir>
```

## Metrics of the best model (TDAVSS 79 epochs)

     train_loss     : -11.48
     
     train_SI-SDR   : 9.95
     
     train_SI-SNR   : 11.48
     
     train_SI-SNRi  : 11.35
     
     train_PESQ     : 1.98

     val_loss       : -9.98
     
     val_SI-SDR     : 8.78
     
     val_SI-SNR     : 9.98
     
     val_SI-SNRi    : 9.98
     
     val_PESQ       : 1.86

     
## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
