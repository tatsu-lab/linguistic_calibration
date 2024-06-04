# Copyright 2024 Neil Band
# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

# TODO(@user): * You should set the following paths to the correct locations on your system. *

# This is where you download the Llama 2 7B model checkpoints
PATH_TO_LLAMA_2_7B_BASE_CHECKPOINT = None  # Download from https://huggingface.co/meta-llama/Llama-2-7b-hf
PATH_TO_LLAMA_2_7B_CHAT_CHECKPOINT = None  # Download from https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

# This is where you download the checkpoints: `<dir_to_save_all_models>` in the README
CHECKPOINT_CACHE_DIR = None

# For HuggingFace Transformers caching
DEFAULT_CACHE_DIR = None

# For saving output results from the evaluation scripts
DEFAULT_OUTPUT_DIR = None

# If you use FactScore biography generation evaluation, you should set this
# Will store the FactScore Wikipedia data and database
FACTSCORE_CACHE_PATH = None

# If you want to evaluate on BioASQ:
#   BioASQ is not available on Hugging Face Datasets.
#   You need to manually install BioASQ from here: http://participants-area.bioasq.org/datasets/
#   Specifically, you should install Task B dataset: Training 12b
#   Set the following constant BIOASQ_JSON_PATH to /path/to/your/file/training12b_new.json
BIOASQ_JSON_PATH = None

# TODO(@user): * End of constants you need to set. *

# This is where we download cached datasets from
HF_DATASETS_PATH = "tatsu-lab/linguistic_calibration"

WANDB_PROJECT = "linguistic_calibration"
BASE_MODELS = {
    "llama-2-7b-hf"
}
LLAMA_2_CHAT_MODELS = {
    'llama-2-7b-chat-hf',
}
MODEL_NAME_TO_FAMILY = {
    "llama-2-7b-hf": "llama",
    "meta-llama/Llama-2-7b-hf": 'llama',
    "tatsu-lab/linguistic-calibration-lc-sft-wdiff": "llama",
    "tatsu-lab/linguistic-calibration-factuality-sft-wdiff": "llama",
    "tatsu-lab/linguistic-calibration-extract-answers": "redpajama",
    "togethercomputer/RedPajama-INCITE-Base-3B-v1": 'redpajama',
}
GPTNEOX_MODEL_CLASSES = {
    "redpajama"
}
OPENAI_MODELS = {
    "gpt-4-1106-preview"
}
ANTHROPIC_MODELS = {
    "claude-2.0",
    "claude-3-opus-20240229"
}

SHORT_NAME_TO_MODEL_PATH = {
    # Base model
    'llama-2-7b-hf': PATH_TO_LLAMA_2_7B_BASE_CHECKPOINT,

    # Chat model
    'llama-2-7b-chat-hf': PATH_TO_LLAMA_2_7B_CHAT_CHECKPOINT,

    # Finetuned Llama 2 models
    "lc_sft": os.path.join(CHECKPOINT_CACHE_DIR, "lc-sft"),
    "factuality_sft": os.path.join(CHECKPOINT_CACHE_DIR, "factuality-sft"),
    "claude_distill": os.path.join(CHECKPOINT_CACHE_DIR, "claude-distill"),
    "factuality_rl": os.path.join(CHECKPOINT_CACHE_DIR, "factuality-rl"),
    "lc_rl": os.path.join(CHECKPOINT_CACHE_DIR, "lc-rl"),

    # GPT-4
    'gpt-4-1106-preview': 'gpt-4-1106-preview',
}

# Huggingface model naming convention.
WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF2_WEIGHTS_INDEX_NAME = "tf_model.h5.index.json"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
FLAX_WEIGHTS_INDEX_NAME = "flax_model.msgpack.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME
GENERATION_CONFIG_NAME = "generation_config.json"
MODEL_CARD_NAME = "modelcard.json"

TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
