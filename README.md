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
python3 train.py -cn="ctc_net" trainer.n_epochs=100  HYDRA_CONFIG_ARGUMENTS
```

## How to run inference

- To evaluate audio-only model specify *-cn "inference"* (default) and for audio-visual models use *-cn "inference_av"* as the first argument of the command
- To evaluate your own pretrained model defined in the project, specify model config as *model=* and the path to it's weights in *from_pretrained=""* (path to the **best model** by default, the weights are loaded from gdrive)
- To evaluate a model on your data set specify the directories to audio and/or video files in *datasets.test.audio_dir="" datasets.test.mouths_dir="" *
- If you want to evaluate the model on less than 8 utterances, specify *dataloader.batch_size=*
- Predicted utterances are saved in *inferencer.save_path / test*

Command for inferencing the best model:

```bash
python3 inference.py -cn "inference_av" model=ctc_net HYDRA_CONFIG_ARGUMENTS
```

## How to calculate metrics on predicted utterances

- **pred_dir** is a directory, where there must be two folders: *predicted_s1* and *predicted_s2* with predictions (running inference in this project creates such directory)
- **gt_dir** is a directory of the data set folders "mix", "s1", "s2" (see part [testing](https://github.com/markovka17/dla/tree/2024/project_avss))

```bash
python3 compute_metrics.py --pred_dir <dir> --gt_dir <dir>
```
## Link to pretrained CTC-NET model 
[link](https://drive.google.com/file/d/1iCQUvTOF3UPaMj6hdJUmjh3yTs8SoRf0/view?usp=sharing)

## Metrics of the best model (CTCnet 100 epochs)

    loss           : -14.003961563110352
    SI-SDR         : 13.99708366394043
    SI-SNR         : 14.004009246826172
    SI-SNRi        : 10.568355560302734
    PESQ           : 2.3285529613494873
    val_loss       : -11.430553519903723
    val_SI-SDR     : 11.427801132202148
    val_SI-SNR     : 11.430594444274902
    val_SI-SNRi    : 11.440431594848633
    val_PESQ       : 2.0325489044189453

     
## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
