# Copyright 2024 Neil Band
# Copyright 2023 The Alpaca Team
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os

import numpy as np
import torch
import transformers
from huggingface_hub import HfApi, hf_hub_download

from linguistic_calibration.models.reward_model import RewardConfig, RewardModel


min_transformers_version = "4.31.0"


def get_linguistic_calibration_model_names():
    api = HfApi()
    models = api.list_models(author="tatsu-lab", search="linguistic-calibration")
    model_names = [model.modelId for model in models]
    model_names = [name.replace("tatsu-lab/linguistic-calibration-", "").replace("-wdiff", "") for name in model_names]
    return model_names


def build_argparse(model_names):
    parser = argparse.ArgumentParser("Download Linguistic Calibration models")
    parser.add_argument("--llama-2-7b-hf-dir", type=str, required=True)
    parser.add_argument(
        "--linguistic-calibration-model-name",
        choices=model_names + ["all"],
        default="all",
        required=True)
    parser.add_argument("--models-save-dir", default="./pretrained_models", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--path-to-lc-sft", type=str, help="Necessary for reconstructing ForecastProbs model.")
    parser.add_argument(
        "--path-to-factuality-sft",
        type=str,
        help="Necessary for reconstructing binary correctness reward model used in Factuality RL.")
    parser.add_argument("--run-integrity-check", action="store_true", help="Run integrity check on the model weights.")
    args = parser.parse_args()
    if args.path_to_lc_sft is None:
        args.path_to_lc_sft = os.path.join(args.models_save_dir, "lc-sft")

    if args.path_to_factuality_sft is None:
        args.path_to_factuality_sft = os.path.join(args.models_save_dir, "factuality-sft")

    return args


def load_weight_diff(
    hf_hub_name,
    is_reward_model=False,
    device="cpu",
    path_to_lc_sft=None,
    path_to_factuality_sft=None
):
    if is_reward_model:
        if "factuality" in hf_hub_name:
            path_to_sft = path_to_factuality_sft
        else:
            path_to_sft = path_to_lc_sft

        model_tuned = RewardModel.from_pretrained(
            hf_hub_name,
            device_map={"": torch.device(device)},
            torch_dtype=torch.float32,
            flash_attn=False,
            config=RewardConfig(backbone_model_name_or_path=path_to_sft),
        )
    else:
        model_tuned = transformers.AutoModelForCausalLM.from_pretrained(
            hf_hub_name, device_map={"": torch.device(device)}, torch_dtype=torch.float32
        )
    tokenizer_tuned = transformers.AutoTokenizer.from_pretrained(hf_hub_name)
    return model_tuned.eval(), tokenizer_tuned


def load_raw_model(model_dir, device="cpu"):
    config_path = os.path.join(model_dir, "config.json")
    config = json.load(open(config_path, "r"))
    transformers_version = config["transformers_version"]
    if transformers_version < min_transformers_version:
        logging.warning(
            f"Your base Llama 2 checkpoint was loaded with version {transformers_version}. "
            f"Please ensure that the HF version is at least {min_transformers_version} to avoid unexpected behavior."
        )

    model_raw = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, device_map={"": torch.device(device)}, torch_dtype=torch.float32
    )
    tokenizer_raw = transformers.AutoTokenizer.from_pretrained(model_dir)
    return model_raw.eval(), tokenizer_raw


def reconstruct_tuned_model(model_tuned, model_raw, is_reward_model=False):
    # modifies model_tuned in-place
    state_dict_diff = model_tuned.state_dict()
    state_dict_raw = model_raw.state_dict()
    if is_reward_model:
        # reward model adds nesting to main transformer
        state_dict_raw = {f"backbone_model.{k}": v for k, v in state_dict_raw.items()}
    for key in state_dict_raw:
        if state_dict_raw[key].size() != state_dict_diff[key].size():
            # weights with a size mismatch are not diff'd in the upload
            print(f"skipping {key} because of size mismatch")
            continue
        state_dict_diff[key].add_(state_dict_raw[key])


def integrity_check(model_tuned, hf_hub_name):
    model_sum = sum(param.sum() for param in model_tuned.state_dict().values()).item()
    model_sum_file = hf_hub_download(repo_id=hf_hub_name, filename="model_sum.txt")
    with open(model_sum_file, "r") as f:
        model_sum_hf_hub = float(f.read())

    return np.isclose(model_sum_hf_hub, model_sum)


if __name__ == "__main__":
    model_names = get_linguistic_calibration_model_names()
    print('All available models:', model_names)
    args = build_argparse(model_names)
    model_names = (
        model_names if args.linguistic_calibration_model_name == "all"
        else [args.linguistic_calibration_model_name])

    # Can be run after models have been downloaded to validate their checksums (sum of all parameters)
    if args.run_integrity_check:
        for model_name in model_names:
            if model_name == 'extract-answers':
                hf_hub_name = f"tatsu-lab/linguistic-calibration-{model_name}"
            else:
                hf_hub_name = f"tatsu-lab/linguistic-calibration-{model_name}-wdiff"

            save_dir = os.path.join(args.models_save_dir, model_name)

            model_tuned, _ = load_weight_diff(
                save_dir, "reward-model" in model_name, args.device, args.path_to_lc_sft, args.path_to_factuality_sft)

            if integrity_check(model_tuned, hf_hub_name):
                print(f"Model {model_name} passed integrity check.")
            else:
                print(f"Model {model_name} failed integrity check. "
                      f"Did you use the latest HF weights at meta-llama/Llama-2-7b-hf?")

            print('----' * 40)
    else:
        for model_name in model_names:
            print("Downloading", model_name)

            if model_name == 'extract-answers':
                # We simply saved the finetuned RedPajama3B model
                hf_hub_name = f"tatsu-lab/linguistic-calibration-{model_name}"
                is_reward_model = False
                save_dir = os.path.join(args.models_save_dir, model_name)

                # Just loads the model and tokenizer
                model_tuned, tokenizer_tuned = load_weight_diff(hf_hub_name, is_reward_model, args.device)

                if not integrity_check(model_tuned, hf_hub_name):
                    print("Model weights integrity check failed for RedPajama3B-based ExtractAnswers model.")
            else:
                hf_hub_name = f"tatsu-lab/linguistic-calibration-{model_name}-wdiff"
                is_reward_model = "reward-model" in model_name
                save_dir = os.path.join(args.models_save_dir, model_name)

                model_tuned, tokenizer_tuned = load_weight_diff(
                    hf_hub_name, is_reward_model, args.device, args.path_to_lc_sft, args.path_to_factuality_sft)
                model_raw, tokenizer_raw = load_raw_model(args.llama_2_7b_hf_dir, args.device)
                reconstruct_tuned_model(model_tuned, model_raw, is_reward_model)

                if not integrity_check(model_tuned, hf_hub_name):
                    print("Model weights integrity check failed. "
                          "Did you use the latest HF weights at meta-llama/Llama-2-7b-hf?")

            # For reward models, set `backbone_model_name_or_path` attribute in config to point to SFT models
            if model_name == 'reward-model-factuality':
                model_tuned.config.update({"backbone_model_name_or_path": args.path_to_factuality_sft})
                print("Updated reward-model-factuality config to point to Factuality SFT model at ",
                      args.path_to_factuality_sft)
            elif model_name == 'reward-model-forecastprobs':
                model_tuned.config.update({"backbone_model_name_or_path": args.path_to_lc_sft})
                print("Updated reward-model-forecastprobs config to point to LC SFT model at ", args.path_to_lc_sft)

            model_tuned.save_pretrained(save_dir)
            tokenizer_tuned.save_pretrained(save_dir)

            print("Downloaded to", save_dir)
            print('----' * 40)
