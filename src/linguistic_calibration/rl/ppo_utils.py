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

import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import transformers

from linguistic_calibration import distributed_utils
from linguistic_calibration import constants

from .. import constants, logging

logger = logging.get_logger(__name__)


@dataclass
class DataArguments:
    dataset_path: str = field(default=constants.HF_DATASETS_PATH)
    dataset_name: str = field(default="trivia_qa_paragraph_generation")
    train_splits: List[str] = field(default_factory=lambda: ["ppo"])
    eval_splits: List[str] = field(default_factory=lambda: ["ppo_validation"])
    policy_prompt_path: str = field(
        default=None,
        metadata={"help": "Path to the prompt to format examples for the policy."},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    wandb_project: str = field(default=constants.WANDB_PROJECT)
    cache_dir: Optional[str] = field(default=constants.DEFAULT_CACHE_DIR)
    flash_attn: bool = field(default=False)
    optim: str = field(default="paged_adamw_8bit")
    truncate_tokens: Optional[List[str]] = field(
        default_factory=lambda: None,
        metadata={
            "help": "Tokens in strings to truncate at first occurrence. "
            "This was used in original OAI summarization paper to avoid models returning incomplete sentences. "
        },
    )
    truncate_after: Optional[int] = field(
        default=None, metadata={"help": "Truncate after this number of tokens. Prevents early truncation."}
    )
    penalty_reward_value: float = field(
        default=-1.0,
        metadata={
            "help": "Reward assigned to sequences that are truncated, "
            "e.g., due to outputting incomplete sentences for given context window."
        },
    )
    total_epochs: int = field(default=10)
    total_steps: int = field(default=None)  # If set, overrides total_epochs.
    rollout_batch_size: int = field(default=512)
    step_batch_size: int = field(default=512)
    rollout_per_device_batch_size: int = field(default=16)
    step_per_device_batch_size: int = field(default=2)
    eval_rollout_per_device_batch_size: int = field(default=16)
    eval_reward_model_per_device_batch_size: int = field(default=32)
    noptepochs: int = field(default=1)
    vf_coef: float = field(default=0.1)
    cliprange: float = field(default=0.2)
    cliprange_value: float = field(default=0.2)
    gamma: float = field(default=1.0)
    lam: float = field(default=1.0)
    whiten_rewards: bool = field(default=True)
    adam_epsilon: float = field(
        default=1e-5,
        metadata={
            "help": "Epsilon for AdamW optimizer. "
            "This is the default for OAI PPO code and UW Quark code. "
            "This is not the Hugging Face default."
        },
    )
    temperature: float = field(default=0.7)
    kl_coef: float = field(default=0.1)
    target_kl: float = field(default=6.0)
    k_beta: float = field(default=0.1)
    adaptive_kl: bool = field(default=False)
    eval_batches: int = field(default=sys.maxsize, metadata={"help": "Maximum number of batches to evaluate on."})
    init_value_with_reward: bool = field(
        default=True, metadata={"help": "Initialize the value model with the reward model."}
    )
    save_after: int = field(default=0, metadata={"help": "Start saving after this many steps."})
    save_steps_extra: Optional[str] = field(
        default=None,
        metadata={
            "help": "A list of predetermined checkpoints to save, represented in the format 'no1__no2__no3'. "
            "Parse this with str.split('__')."
        },
    )
    query_len: int = field(default=128)
    response_len: int = field(default=300)
    policy_model_name_or_path: str = field(default=None)
    reward_model_name_or_path: str = field(default=None)
    answer_extractor_model_name_or_path: str = field(default=None)
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Use fast tokenizer if True. "
            "Fast LLaMA tokenizer forces protobuf downgrade to 3.20.3. "
            "Use fast tokenizer only if you can live with that."
        },
    )

    # Linguistic Calibration arguments
    rl_type: str = field(
        default="decision_based_rl",
        metadata={"help": "Type of RL to use; decision_based_rl or factuality_rl."})
    cross_entropy_epsilon: float = field(
        default=1e-4,
        metadata={"help": "Epsilon for cross entropy loss."})
    reward_offset: float = field(
        default=5.0,
        metadata={"help": "Positive constant C to add to reward."})
    answer_extractor_normalization_lambda: float = field(
        default=5.0,
        metadata={"help": "Lambda for answer extractor normalization."})
    value_model_prompt_path: str = field(
        default=None,
        metadata={"help": "Path to the prompt to format examples for the value model."},
    )
    reward_model_prompt_path: str = field(
        default=None,
        metadata={"help": "Path to the prompt to format examples for the reward model."},
    )
    answer_extractor_prompt_path: str = field(
        default=None,
        metadata={"help": "Path to the prompt to format examples for the answer extractor."},
    )
    answer_extractor_response_len: Optional[int] = field(
        default=300,
        metadata={"help": "The response length for the answer extractor."})
    answer_extractor_temperature: Optional[float] = field(
        default=0.3,
        metadata={"help": "The temperature for the answer extractor."})

    def __post_init__(self):
        # Super class' __post_init__ is very complicated; don't do super for now in case mess something up.
        # super().__post_init__()

        if self.tf32:  # super().__post_init__() actually does this.
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True  # noqa

        world_size = distributed_utils.get_world_size()

        # Checks on rollout_batch_size only matter for PPO.
        assert self.rollout_batch_size >= self.rollout_per_device_batch_size * world_size, (
            "rollout_batch_size is smaller than rollout_per_device_batch_size * world_size. "
            "Increase the former or decrease the latter to fix this."
        )
        assert (
            self.rollout_batch_size % (self.rollout_per_device_batch_size * world_size) == 0
        ), "rollout_batch_size is not a multiple of rollout_per_device_batch_size * world_size. "

        assert self.step_batch_size >= self.step_per_device_batch_size * world_size, (
            "step_batch_size is smaller than step_per_device_batch_size * world_size. "
            "Increase the former or decrease the latter to fix this."
        )
        assert (
            self.step_batch_size % (self.step_per_device_batch_size * world_size) == 0
        ), "step_batch_size is not a multiple of step_per_device_batch_size * world_size. "

        logger.warning(
            f"Rollout stats:\n"
            f"\trollout_batch_size: {self.rollout_batch_size}\n"
            f"\trollout_per_device_batch_size: {self.rollout_per_device_batch_size}\n"
            f"\tworld_size: {world_size}\n",
        )
        assert (self.rollout_batch_size // self.rollout_per_device_batch_size) % world_size == 0
        self.rollout_accumulation_steps = self.rollout_batch_size // self.rollout_per_device_batch_size // world_size

        logger.warning(
            f"Step stats:\n"
            f"\tstep_batch_size: {self.step_batch_size}\n"
            f"\tstep_per_device_batch_size: {self.step_per_device_batch_size}\n"
            f"\tworld_size: {world_size}\n",
        )
        assert (self.step_batch_size // self.step_per_device_batch_size) % world_size == 0
        self.gradient_accumulation_steps = self.step_batch_size // self.step_per_device_batch_size // world_size

        logger.warning(
            f"Accumulation steps:\n"
            f"\trollout_accumulation_steps: {self.rollout_accumulation_steps}\n"
            f"\tgradient_accumulation_steps: {self.gradient_accumulation_steps}\n"
        )

        if self.save_steps_extra is not None:
            self.save_steps_extra_list = [int(string) for string in self.save_steps_extra.split("__")]
        else:
            self.save_steps_extra_list = []

    def set_truncate_token_ids(self, tokenizer: transformers.PreTrainedTokenizer):
        """Convert truncation token to token ids.

        This is called in RLTrainer.
        """
        truncate_tokens = self.truncate_tokens
        if truncate_tokens is None:
            truncate_token_ids = None
        else:
            truncate_token_ids = tokenizer.convert_tokens_to_ids(truncate_tokens)
        self.truncate_token_ids = truncate_token_ids
