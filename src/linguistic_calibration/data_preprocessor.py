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

import copy
import dataclasses
from typing import Callable, Dict, List, Optional, Sequence, Union

import einops
import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset

from . import constants, logging, torch_ops, utils
from .types import Tensor
from collections import defaultdict

logger = logging.get_logger(__name__)


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> dict:
    """Tokenize a list of strings and return the tokenized content as well metadata (e.g., truncation statistics)."""
    padding = getattr(tokenizer, "padding", "max_length")
    return_overflowing_tokens = transformers.__version__ <= "4.26.1"
    # TODO(lxuechen): Until HF supports fast tokenizer for OPT, we can't make a joint call on the list of strings
    #  when `return_overflowing_tokens=True`.
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=padding,
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_overflowing_tokens=return_overflowing_tokens,
        )
        for text in strings
    ]

    if padding == "max_length":
        input_ids = labels = torch.cat([tokenized.input_ids for tokenized in tokenized_list])
    else:  # "longest"
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]

    if return_overflowing_tokens:
        input_ids_lens = labels_lens = [
            tokenizer.model_max_length + tokenized.num_truncated_tokens.item() for tokenized in tokenized_list
        ]
        # `num_truncated_tokens` can be negative, if no truncation occurred.
        num_truncated_tokens = sum(max(tokenized.num_truncated_tokens.item(), 0) for tokenized in tokenized_list)
        num_truncated_examples = sum(tokenized.num_truncated_tokens.item() > 0 for tokenized in tokenized_list)
    else:
        logger.warning(
            "You are using a `transformers` version that does not support `return_overflowing_tokens=True`. "
            "The tokenization metadata will not be recorded."
            "In order to see truncation statistics, please downgrade to `transformers<=4.26.1`."
        )
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        num_truncated_tokens = num_truncated_examples = -1

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
        tokenization_metadata=dict(
            num_examples=len(tokenized_list),
            num_truncated_tokens=num_truncated_tokens,
            num_truncated_examples=num_truncated_examples,
            input_ids_avg_len=utils.mean(input_ids_lens),
            input_ids_max_len=max(input_ids_lens),
            input_ids_min_len=min(input_ids_lens),
            labels_avg_len=utils.mean(labels_lens),
            labels_max_len=max(labels_lens),
            labels_min_len=min(labels_lens),
            model_max_length=tokenizer.model_max_length,
        ),
    )


def preprocess_for_lc_sft(
    df: pd.DataFrame,
    prompt_template: str,
    tokenizer: transformers.PreTrainedTokenizer,
    df_postprocessor=None,
    verbose=True,
) -> dict[str, Union[torch.Tensor, Sequence[torch.Tensor]]]:
    """Tokenize each example and create the labels.

    Args:
        df: DataFrame containing the data.
        prompt_template: Template for formatting prompts.
        tokenizer: Tokenizer to use. If None, use the tokenizer for the given model.
        df_postprocessor: Function to apply to the DataFrame before tokenization.
        verbose: Whether to print tokenization metadata.

    Returns:
        A dictionary mapping str to torch.Tensor.
    """
    if df_postprocessor is not None:
        df = df_postprocessor(df)
    list_dict_data = df.to_dict(orient="records")

    sources = [prompt_template.format(**dict_data) for dict_data in list_dict_data]
    targets = []
    for dict_data in list_dict_data:
        target = dict_data["target"]
        target += tokenizer.eos_token
        targets.append(target)

    examples = [s + t for s, t in utils.zip_(sources, targets)]

    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]

    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in utils.zip_(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = constants.IGNORE_INDEX  # Input context should not contribute to loss.

    packaged_data = dict(
        input_ids=input_ids,
        labels=labels,
        metadata=dict(),
        tokenization_metadata=examples_tokenized["tokenization_metadata"],
    )
    if verbose:
        logger.warning(f"Tokenization metadata:\n{utils.jdumps(packaged_data['tokenization_metadata'])}")

    return packaged_data


def postprocess_binary_correctness_reward_modeling_df(
    df: pd.DataFrame
) -> pd.DataFrame:
    # Convert the following column names:
    # factuality_sft_generated_paragraph -> generated_paragraph
    # factuality_sft_binary_correctness -> reward
    # Drop the rest of the columns
    df = df.rename(columns={
        "factuality_sft_generated_paragraph": "generated_paragraph",
        "factuality_sft_binary_correctness": "reward"
    })
    df = df[["generated_paragraph", "ground_truth_top_answer", "question", "reward"]]
    logger.warning(f"Number of rows in the reward model DF before postprocessing: {len(df)}")

    # Drop any row with NaN reward or reward not in {0, 1}
    df = df.dropna(subset=["reward"])
    df = df[df["reward"].isin({0, 1})]

    logger.warning(f"Number of rows in the reward model DF after dropping NaN "
                   f"and rewards not equal to 0 or 1: {len(df)}")
    return df


def postprocess_forecast_probs_reward_modeling_df(
    df: pd.DataFrame
) -> pd.DataFrame:
    data_dict = defaultdict(list)

    # Unroll the "lc_sft_ground_truth_and_extracted_answers" and "lc_sft_forecasted_probs" columns
    for i, row in df.iterrows():
        generated_paragraph = row["lc_sft_generated_paragraph"]
        ground_truth_and_extracted_answers = row["lc_sft_ground_truth_and_extracted_answers"]
        forecasted_probs = row["lc_sft_forecasted_probs"]

        # Continue if either of the columns is NaN
        if pd.isna(ground_truth_and_extracted_answers) or pd.isna(forecasted_probs):
            continue

        ground_truth_and_extracted_answers = eval(ground_truth_and_extracted_answers)
        forecasted_probs = eval(forecasted_probs)

        assert isinstance(ground_truth_and_extracted_answers, list) and isinstance(forecasted_probs, list)

        # If they are empty, continue
        if len(ground_truth_and_extracted_answers) == 0 or len(forecasted_probs) == 0:
            continue

        assert len(ground_truth_and_extracted_answers) == len(forecasted_probs), (
            f"Length of ground_truth_and_extracted_answers ({len(ground_truth_and_extracted_answers)}) "
            f"and forecasted_probs ({len(forecasted_probs)}) should be the same")

        for j, (answer_choice) in enumerate(ground_truth_and_extracted_answers):
            reward_to_add = forecasted_probs[j]

            if pd.isna(answer_choice) or pd.isna(reward_to_add):
                continue

            data_dict["question"].append(row["question"])
            data_dict["ground_truth_top_answer"].append(answer_choice)

            # Assert that the reward is in [0, 1]
            assert 0 <= reward_to_add <= 1, f"Reward should be in [0, 1], but got {reward_to_add}"

            data_dict["reward"].append(reward_to_add)
            data_dict["generated_paragraph"].append(generated_paragraph)

    df_to_return = pd.DataFrame(data_dict)

    logger.warning(f"Number of rows in the reward model DF after postprocessing: {len(df_to_return)}")
    return df_to_return


def postprocess_extract_answers_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    data_dict = defaultdict(list)

    # Featurize ExtractAnswers targets
    for i, row in df.iterrows():
        ground_truth_and_extracted_answers = row["lc_sft_ground_truth_and_extracted_answers"]
        forecasted_probs = row["lc_sft_forecasted_probs"]

        # Continue if either of the columns is NaN
        if pd.isna(ground_truth_and_extracted_answers) or pd.isna(forecasted_probs):
            continue

        ground_truth_and_extracted_answers = eval(ground_truth_and_extracted_answers)
        forecasted_probs = eval(forecasted_probs)

        assert isinstance(ground_truth_and_extracted_answers, list) and isinstance(forecasted_probs, list)
        assert len(ground_truth_and_extracted_answers) == len(forecasted_probs)

        # Ensure none of the values in either list are NaN
        if any(pd.isna(x) for x in ground_truth_and_extracted_answers) or any(pd.isna(x) for x in forecasted_probs):
            continue

        if len(ground_truth_and_extracted_answers) == 0:
            continue
        elif len(ground_truth_and_extracted_answers) == 1:
            # There were no extracted answers
            data_dict["target"].append("No Answer")
        else:
            extracted_answers = ground_truth_and_extracted_answers[1:]

            # Map to string
            extracted_answers = map(str, extracted_answers)

            # Remove duplicates, without changing order
            extracted_answers = list(dict.fromkeys(extracted_answers))

            # Join the answers
            extracted_answers = '; '.join(extracted_answers)

            data_dict["target"].append(extracted_answers)

        data_dict["question"].append(row["question"])
        data_dict["generated_paragraph"].append(row["lc_sft_generated_paragraph"])

    df_to_return = pd.DataFrame(data_dict)

    logger.warning(f"Number of rows in the ExtractAnswers DF after postprocessing: {len(df_to_return)}")
    return df_to_return


def lc_sft_get_target_column(row):
    row["target"] = row["claude_summary"]
    return row


def factuality_sft_get_target_column(row):
    list_of_icl_generated_paragraphs = row["icl_generated_paragraphs"]
    try:
        list_of_icl_generated_paragraphs = eval(list_of_icl_generated_paragraphs)
        row["target"] = list_of_icl_generated_paragraphs[0]
    except Exception:
        row["target"] = None

    return row


def claude_distill_get_target_column(row):
    row["target"] = row["claude_generated_paragraph"]
    return row


def postprocess_sft_df(
    df: pd.DataFrame,
    sft_type: str
) -> pd.DataFrame:
    if sft_type == "lc_sft":
        row_filter_fn = lc_sft_get_target_column
    elif sft_type == "factuality_sft":
        row_filter_fn = factuality_sft_get_target_column
    elif sft_type == "claude_distill":
        row_filter_fn = claude_distill_get_target_column
    else:
        raise ValueError(f"Unsupported SFT type: {sft_type}")

    df = df.apply(row_filter_fn, axis=1)

    # Keep only paragraph_generation_prompt, target columns
    df = df[["paragraph_generation_prompt", "target"]]

    # Drop rows with NaN in either column
    df = df.dropna(subset=["paragraph_generation_prompt", "target"])

    logger.warning(f"Number of rows in the SFT DF after postprocessing: {len(df)}")
    return df


def preprocess_for_lc_reward_modeling(
    df: pd.DataFrame,
    prompt_template: str,
    tokenizer: transformers.PreTrainedTokenizer,
    df_postprocessor: Optional[Callable] = None,
    end_sequence_with_eos: bool = False,
    verbose=True,
    max_generated_paragraph_length: int = 2000,
    truncate_to_multiple_of_64: bool = False,
) -> dict[str, torch.Tensor]:
    if df_postprocessor is not None:
        df = df_postprocessor(df)

    list_dict_data = df.to_dict(orient="records")

    if truncate_to_multiple_of_64:
        logger.warning(f"Truncating to multiple of 64. Original length: {len(df)}")

        if len(list_dict_data) % 64 != 0:
            list_dict_data = list_dict_data[:-(len(list_dict_data) % 64)]

        logger.warning(f"Truncated length: {len(list_dict_data)}")

    reward = torch.tensor([[dict_data["reward"]] for dict_data in list_dict_data])

    def _get_text(example: dict):
        format_dict = {
            "generated_paragraph": example["generated_paragraph"][:max_generated_paragraph_length],
            "question": example["question"],
            "ground_truth_top_answer": example["ground_truth_top_answer"],
        }
        prompt_str = prompt_template.format(**format_dict)

        if end_sequence_with_eos:
            prompt_str += tokenizer.eos_token

        return prompt_str

    text_list = [_get_text(dict_data) for dict_data in list_dict_data]

    logger.warning(f"Tokenizing {len(list_dict_data)} pairs...")
    tokenized_list = _tokenize_fn(text_list, tokenizer)

    # "size" (bsz, seq_len)
    input_ids = tokenized_list["input_ids"]
    labels = tokenized_list["labels"]
    tokenization_metadata = tokenized_list["tokenization_metadata"]

    packaged_data = dict(
        input_ids=input_ids,
        labels=labels,
        reward=reward,
        tokenization_metadata=tokenization_metadata,
        metadata=dict(mean_reward=reward.float().mean().item()),
    )
    if verbose:
        logger.warning(f"Tokenization metadata:\n{utils.jdumps(packaged_data['tokenization_metadata'])}")

    return packaged_data


def _get_generator(seed: int) -> torch.Generator:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return rng


def split_train_into_train_and_eval(train_dataset: Dataset, eval_size: int, seed: int) -> tuple[Dataset, Dataset]:
    assert eval_size < len(
        train_dataset  # noqa
    ), "Requested eval_size cannot be equal/larger than original train data size."
    new_train_size = len(train_dataset) - eval_size  # noqa
    train_dataset, eval_dataset = torch.utils.data.random_split(
        train_dataset, [new_train_size, eval_size], generator=_get_generator(seed)
    )
    return train_dataset, eval_dataset


class LCSFTDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        prompt_template: str,
        tokenizer: transformers.PreTrainedTokenizer,
        df_postprocessor: Optional[Callable] = None,
    ):
        super(LCSFTDataset, self).__init__()
        data_dict = preprocess_for_lc_sft(
            df=df,
            prompt_template=prompt_template,
            tokenizer=tokenizer,
            df_postprocessor=df_postprocessor
        )
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.metadata = data_dict["metadata"]
        self.tokenization_metadata = data_dict["tokenization_metadata"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclasses.dataclass
class DataCollatorForLCSFTDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=constants.IGNORE_INDEX)
        # When sequences are right padded, `attention_mask` is only useful for T5 training.
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


class LCRewardModelingDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        prompt_template: str,
        tokenizer: transformers.PreTrainedTokenizer,
        df_postprocessor: Optional[Callable] = None,
        end_sequence_with_eos: bool = False,
        truncate_to_multiple_of_64: bool = False,
    ):
        super(LCRewardModelingDataset, self).__init__()
        data_dict = preprocess_for_lc_reward_modeling(
            df=df,
            prompt_template=prompt_template,
            tokenizer=tokenizer,
            df_postprocessor=df_postprocessor,
            end_sequence_with_eos=end_sequence_with_eos,
            truncate_to_multiple_of_64=truncate_to_multiple_of_64,
        )
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.reward = data_dict["reward"]
        self.metadata = data_dict["metadata"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            reward=self.reward[i],
        )


@dataclasses.dataclass
class DataCollatorForLCRewardModelingDataset(object):
    """
    This collation assumes data preprocessing converts text into *padded* tensors of the same length.
    For autoregressive models like OPT and GPT2, `input_ids` alone is sufficient to produce the rewards.
    For enc-dec models like T5, we need `labels`.

    `input_ids` and `labels` are tensors of size (bsz, max_seq_len).
    `reward` is an int/long tensor of size (bsz,) indicating the reward for the sequence.
    """
    tokenizer: transformers.PreTrainedTokenizer

    def _left_pad_helper(self, instances: Sequence[dict], key: str):
        # TODO(lxuechen): Potentially replace with `transformers.PretrainedTokenizerBase.prepare_for_model`.
        # `instances` is a list of dicts, each dict has key whose value is a list of tensors, possibly of unequal length.
        input_ids = [instance[key] for instance in instances]  # Flatten.
        input_ids = torch_ops.pad_sequence_from_left(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Tensor]:
        reward = torch.squeeze(
            torch.stack([instance['reward'] for instance in instances]))
        input_ids = self._left_pad_helper(instances, "input_ids")
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            reward=reward,
        )


class LCQueryDataset(Dataset):
    """Dataset that emits tokenized left-padded queries."""

    def __init__(
        self,
        df: pd.DataFrame,
        prompt_template: str,
        tokenizer: transformers.PreTrainedTokenizer,
        query_len: int,
        df_postprocessor: Optional[Callable] = None,
    ):
        super(LCQueryDataset, self).__init__()

        if df_postprocessor is not None:
            df = df_postprocessor(df)

        list_dict_data = df.to_dict(orient="records")

        prompts = [
            prompt_template.format_map({
                "paragraph_generation_prompt": dict_data["paragraph_generation_prompt"]
            }) for dict_data in list_dict_data]

        queries = [tokenizer(prompt, return_tensors="pt", truncation=False).input_ids[0] for prompt in prompts]

        # queries: List[Tensor], the tokenized input prompts for the policy
        # user_decision_questions: List[str], the user decision questions x ~ p(x)
        # ground_truth_top_answers: List[str], the ground truth top answers y ~ p(y|x)
        unfiltered_user_decision_questions = [dict_data["question"] for dict_data in list_dict_data]
        unfiltered_ground_truth_top_answers = [dict_data["ground_truth_top_answer"] for dict_data in list_dict_data]

        # Filter everything according to query_len of the policy query
        filtered_queries = []
        filtered_user_decision_questions = []
        filtered_ground_truth_top_answers = []
        for i, query in enumerate(queries):
            if len(query) <= query_len:
                filtered_queries.append(query)
                filtered_user_decision_questions.append(unfiltered_user_decision_questions[i])
                filtered_ground_truth_top_answers.append(unfiltered_ground_truth_top_answers[i])

        logger.warning(
            f"Filtered out {len(queries) - len(filtered_queries)} instances out of {len(queries)} that "
            f"exceed length limit. These examples are not used for training, but will still be used in evaluation. "
        )
        queries = torch.stack(
            [
                torch_ops.left_pad(query, target_size=(query_len,), value=tokenizer.pad_token_id)
                for query in filtered_queries
            ]
        )

        self.queries = queries
        self.query_attn_masks = queries.ne(tokenizer.pad_token_id).long()
        self.user_decision_questions = filtered_user_decision_questions
        self.ground_truth_top_answers = filtered_ground_truth_top_answers

        # Auxiliary data. Used during evaluation.
        self.prompts = prompts
        self.list_dict_data = list_dict_data
        self.unfiltered_user_decision_questions = unfiltered_user_decision_questions
        self.unfiltered_ground_truth_top_answers = unfiltered_ground_truth_top_answers

    def __getitem__(self, i):
        return dict(
            queries=self.queries[i],
            query_attn_masks=self.query_attn_masks[i],
            user_decision_question=self.user_decision_questions[i],
            ground_truth_top_answer=self.ground_truth_top_answers[i]
        )

    def __len__(self):
        return len(self.queries)


@dataclasses.dataclass
class DataCollatorForStackableDataset(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Union[Tensor, List[str]]]:
        to_return = {}
        for key in instances[0].keys():
            if key in {"user_decision_question", "ground_truth_top_answer"}:
                to_return[key] = [instance[key] for instance in instances]
            else:
                to_return[key] = torch.stack([instance[key] for instance in instances])

        return to_return
