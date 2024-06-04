# Copyright 2024 Neil Band
# Copyright 2023 The Alpaca Team
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

import torch.nn.functional as F
import transformers
from datasets import load_metric
from scipy.special import expit
from transformers.trainer_utils import EvalPrediction
from typing import Dict

from linguistic_calibration import common


class Trainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # input_ids, attention_mask each of size (bsz, seq_len).
        input_ids, attention_mask, reward = common.unpack_dict(
            inputs, keys=("input_ids", "attention_mask", "reward")
        )
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.rewards

        if reward.ndim == 0:
            reward = reward.unsqueeze(0)

        # Type casting of `reward` is due to amp.autocast context manager.
        loss = F.binary_cross_entropy_with_logits(logits, reward.to(logits.dtype), reduction="mean")
        return (loss, dict(logits=logits)) if return_outputs else loss


def compute_reward_modeling_metrics(eval_prediction: EvalPrediction) -> Dict:
    predictions, labels = eval_prediction
    # Print shape
    print(predictions.shape)
    print(labels.shape)
    mse_metric = load_metric("mse")
    predictions = expit(predictions)
    mse = mse_metric.compute(predictions=predictions, references=labels)
    return dict(mse=mse)
