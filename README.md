# Linguistic Calibration of Long-Form Generations

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/linguistic_calibration/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/linguistic_calibration/blob/main/DATA_LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This repo contains a reference implementation for linguistic calibration of long-form generations (LC), a new alignment objective that naturally encourages LMs to express more calibrated verbal statements of confidence.

Specifically, we provide
* [A training framework to train linguistically calibrated models](#training-framework)
* [An evaluation framework to benchmark the calibration and accuracy of long-form natural language generations](#evaluation-framework)

Check out our paper [Linguistic Calibration of Long-Form Generations](https://arxiv.org/abs/2404.00474) for our research findings.

The data needed to run our code is hosted on HuggingFace (<https://huggingface.co/datasets/tatsu-lab/linguistic_calibration>) and model checkpoints can be found at <https://huggingface.co/tatsu-lab> with format `tatsu-lab/linguistic-calibration-{model}`.

**Usage and License Notices**: 
This codebase is based on [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm).
It is intended and licensed for research use only.
Our datasets are CC BY NC 4.0 (allowing only non-commercial use) because they include generations from API-based LLMs.
Models trained using the datasets should not be used outside of research purposes.
The weight diffs are also CC BY NC 4.0 (allowing only non-commercial use).

<p align="center" width="100%">
<img src="assets/linguistic_calibration_banner.png" alt="LC" style="width: 50%; min-width: 300px; display: block; margin: auto;">
</p>

## Installation

```bash
conda create -n lc python=3.10
conda activate lc

# Install PyTorch Nightly -- example for CUDA 12.1 below
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia

# Install other requirements
python setup.py install
```

You can install the Flash Attention 2 and Apex packages, which we require for PPO with Llama 2 7B, as follows:
```bash
# Flash Attention 2 installation
# For detailed instructions, see https://github.com/Dao-AILab/flash-attention
pip install packaging ninja
pip install flash-attn --no-build-isolation

# Apex installation
# For detailed instructions, see https://github.com/NVIDIA/apex
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Lastly, you should set constants in [linguistic_calibration/constants.py](src/linguistic_calibration/constants.py) to point to the correct paths for your cache directories, checkpoints, etc.

## Training Framework

The LC training framework involves three steps:
1. **Supervised Finetuning (SFT)**: To obtain an LM policy with some ability to express confidence statements, we apply the **summary distillation** algorithm. Summary distillation samples many long-form paragraph generations from a base model (Llama 2 7B), summarizes them into a single consensus paragraph with statements of confidence, and finetunes a model on these summaries.
2. **Reward Modeling**: We train an LM-based **surrogate reader** which, given a long-form generation and a related question, provides a distribution over possible answers. This surrogate reader is used in the reward function during decision-based RL, analogous to how a human preference reward model is used in the RL step of [RLHF](https://arxiv.org/abs/2203.02155). In our implementation, the surrogate reader is composed of two separate functions, each parameterized by a separate LM: **ExtractAnswers** and **ForecastProbs**.
3. **Decision-Based RL**: We finetune the SFT policy using [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347). Our reward function is based on the log loss of the surrogate reader's answer distribution.

We currently support linguistic calibration of Llama 2 7B but it is straightforward to extend our framework to any causal HuggingFace models.

## Running Linguistic Calibration and Baselines

We provide scripts to replicate supervised finetuning and RL for all finetuned confidence and non-confidence baselines.
Example bash scripts for these methods can be found in the `examples/scripts` directory. They include:
* The LC RL pipeline
    * [LC SFT](examples/scripts/lc_sft.sh): the supervised finetuning (SFT) model trained with summary distillation.
    * [ExtractAnswers](examples/scripts/extract_answers.sh): the ExtractAnswers model trained using Claude 2.0 answer extractions (Algorithm 1).
    * [ForecastProbs](examples/scripts/forecast_probs.sh): the ForecastProbs model trained using Claude 2.0 probability forecasts (Algorithm 1).
    * [LC RL](examples/scripts/lc_ppo.sh): the model obtained by training LC SFT with decision-based RL.
* The Factuality RL pipeline
    * [Factuality SFT](examples/scripts/factuality_sft.sh): the SFT model trained on outputs generated with the Llama 2 7B ICL baseline.
    * [Factuality Reward Modeling](examples/scripts/factuality_reward_modeling.sh): the reward model trained on Claude 2.0 binary correctness labels.
    * [Factuality RL](examples/scripts/factuality_ppo.sh): the model obtained by training Factuality SFT with PPO using the Factuality Reward Model.
* [Claude Distill](examples/scripts/claude_distill.sh): the SFT model trained on Claude-generated long-form paragraph generations.

Below we give example commands to reproduce model artifacts. Notes:

- All supervised finetuning and reward modeling scripts were tested without Flash Attention on a machine with 4 80GB A100 GPUs.
- PPO requires at least 8 80GB GPUs and Flash Attention 2 and Apex.
- Before running the code below, follow the instructions in [Downloading Checkpoints](#downloading-checkpoints) to download the necessary checkpoints.
- All scripts below make use of cached datasets from <https://huggingface.co/datasets/tatsu-lab/linguistic_calibration> (e.g., summaries for LC SFT, or API-based LLM forecasts from Claude for LC reward modeling). If you want to use a custom dataset, refer to the [Generating SFT and Reward Modeling Datasets](#generating-sft-and-reward-modeling-datasets) section. 

### Supervised Finetuning (SFT)

To replicate the LC SFT model finetuned from Llama 2 7B using the summary distillation algorithm, run
```bash
bash examples/scripts/lc_sft.sh \
  <your_output_dir_for_lc_sft> \
  <your_wandb_run_name> \
  <your_path_to_llama_2_7b_ckpt_and_tokenizer>
```

The LC SFT model will be saved at `<your_output_dir_for_lc_sft>`, and the name of the wandb run will be `<your_wandb_run_name>`.

The scripts for other SFT baselines (Factuality SFT and Claude Distill) can be used analogously.

### Reward Modeling: ExtractAnswers

To replicate the ExtractAnswers model trained using Claude 2.0 answer extractions, run
```bash
bash examples/scripts/extract_answers.sh \
  <your_output_dir_for_extract_answers> \
  <your_wandb_run_name>
```

### Reward Modeling: ForecastProbs

To replicate the ForecastProbs model trained using Claude 2.0 probability forecasts, run
```bash
bash examples/scripts/forecast_probs.sh \
  <your_output_dir_for_forecast_probs> \
  <your_wandb_run_name>
  <your_path_to_lc_sft_ckpt_and_tokenizer>
```
The script requires the LC SFT model checkpoint and tokenizer to be stored at `<your_path_to_lc_sft_ckpt_and_tokenizer>`, since the ForecastProbs model is initialized from the LC SFT model.

Similarly, you can train the Factuality Reward Model using the Factuality SFT model as the initialization checkpoint, and the script here: [Factuality Reward Modeling](examples/scripts/factuality_reward_modeling.sh).

### Decision-Based RL

To replicate the LC RL model trained with PPO, run
```bash
bash examples/scripts/lc_ppo.sh \
  <your_output_dir_for_lc_ppo> \
  <your_wandb_run_name> \
  <your_path_to_forecast_probs_ckpt_and_tokenizer> \
  <your_path_to_lc_sft_ckpt_and_tokenizer> \
  <your_path_to_extract_answers_ckpt_and_tokenizer>
```

We have observed performance to steadily improve for >1000 steps. The default hyperparameters run 1500 steps of PPO.

### Factuality RL

To replicate the Factuality RL model trained with PPO, run
```bash
bash examples/scripts/factuality_ppo.sh \
  <your_output_dir_for_factuality_ppo> \
  <your_wandb_run_name> \
  <your_path_to_factuality_reward_model_ckpt_and_tokenizer> \
  <your_path_to_factuality_sft_ckpt_and_tokenizer>
```

## Downloading Checkpoints

Our checkpoints (available [here](https://huggingface.co/tatsu-lab), with format `tatsu-lab/linguistic-calibration-{model}`) enable quick replication of reward modeling and PPO. For example, to replicate
* Reward modeling: you can download the LC SFT checkpoint and use `examples/scripts/forecast_probs.sh` to train the `ForecastProbs` function.
* Decision-based RL: you can download the LC SFT, `ExtractAnswers`, and `ForecastProbs` checkpoints and use `examples/scripts/lc_ppo.sh` to train your own LC RL model.

Use the following steps to download checkpoints.

First, install the pretrained Llama 2 7B weights from Huggingface (skip if you have already installed the weights with transformers>=4.31.0).
For example, you can sign up for access to the model weights [here](https://huggingface.co/meta-llama/Llama-2-7b-hf) and then follow the instructions [here](https://huggingface.co/docs/hub/en/models-downloading) to install the weights, or run the following commands:
```bash
git lfs install
git clone git@hf.co:meta-llama/Llama-2-7b-hf
```

If you intend to benchmark Llama 2 7B Chat, you should also download it (`meta-llama/Llama-2-7b-chat-hf`). 

Next, you can either download all checkpoints or a specific one. To download all checkpoints, run
```bash
python pretrained_models/recover_model_weights.py \
  --llama-2-7b-hf-dir=<your_path_to_llama_2_7b_ckpt_and_tokenizer> \
  --linguistic-calibration-model-name=all \
  --models-save-dir=<dir_to_save_all_models>
```

Then, you should set CHECKPOINT_CACHE_DIR in [linguistic_calibration/constants.py](src/linguistic_calibration/constants.py) to `<dir_to_save_all_models>`.

Or, to download a specific model checkpoint, select a model name from the list
* `lc-sft`
* `factuality-sft`
* `claude-distill`
* `extract-answers`
* `lc-rl`
* `factuality-rl`
* `reward-model-forecastprobs`
* `reward-model-factuality`

and then run this command:
```bash
python pretrained_models/recover_model_weights.py \
  --llama-2-7b-hf-dir=<your_path_to_llama_2_7b_ckpt_and_tokenizer> \
  --linguistic-calibration-model-name=<one_of_the_model_names_from_above> \
  --models-save-dir=<dir_to_save_all_models>
```

If you are downloading the `reward-model-forecastprobs` or `reward-model-factuality` checkpoints, you will need to have the `lc-sft` or `factuality-sft` checkpoint, respectively, downloaded already to `<dir_to_save_all_models>`.

## Evaluation Framework

We provide an evaluation framework to benchmark the calibration of long-form natural language generations, supporting all methods from the paper (including baselines using GPT-4) and evaluation using either off-the-shelf question-answering datasets or per-claim level evaluation based on [FactScore](https://arxiv.org/abs/2305.14251).

Demo notebook example: [![Using](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsu-lab/linguistic_calibration/blob/main/examples/auto_eval_demo.ipynb)


## Generating SFT and Reward Modeling Datasets

By default, our SFT, reward modeling, and PPO scripts use cached datasets from <https://huggingface.co/datasets/tatsu-lab/linguistic_calibration>. If you want to use a custom dataset or replicate this part of the pipeline for LC RL or Factuality RL, you can generate the datasets following the Colab walkthrough here:
[![Using](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsu-lab/linguistic_calibration/blob/main/examples/generate_sft_and_reward_model_datasets.ipynb)

## Citation

Please consider citing our work if you use the code, models, or datasets from this repo.
```
@inproceedings{band2024linguistic,
      title={Linguistic Calibration of Long-Form Generations}, 
      author={Neil Band and Xuechen Li and Tengyu Ma and Tatsunori Hashimoto},
      booktitle={Forty-first International Conference on Machine Learning},
      year={2024},
      url={https://openreview.net/forum?id=rJVjQSQ8ye}
}
```

If you use our code, you should also cite AlpacaFarm since this codebase is based on it:
```
@misc{dubois2023alpacafarm,
      title={AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback}, 
      author={Yann Dubois and Xuechen Li and Rohan Taori and Tianyi Zhang and Ishaan Gulrajani and Jimmy Ba and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto},
      year={2023},
      eprint={2305.14387},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

Lastly, if you use the FactScore-based evaluation, please cite the FactScore paper:
```
@inproceedings{ factscore,
    title={ {FActScore}: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation },
    author={ Min, Sewon and Krishna, Kalpesh and Lyu, Xinxi and Lewis, Mike and Yih, Wen-tau and Koh, Pang Wei and Iyyer, Mohit and Zettlemoyer, Luke and Hajishirzi, Hannaneh },
    year={ 2023 },
    booktitle = { EMNLP },
    url={ https://arxiv.org/abs/2305.14251 }
}
```
