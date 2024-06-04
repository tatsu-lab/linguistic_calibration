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
from functools import partial

import datasets
import numpy as np
import pandas as pd
import transformers
from huggingface_hub import login

from . import logging, utils, constants
from .auto_annotations.factscore_retrieval_utils import download_file
from .data_preprocessor import (
    LCRewardModelingDataset,
    DataCollatorForLCRewardModelingDataset,
    postprocess_binary_correctness_reward_modeling_df,
    postprocess_forecast_probs_reward_modeling_df,
    postprocess_extract_answers_df,
    postprocess_sft_df,
    DataCollatorForStackableDataset,
    DataCollatorForLCSFTDataset,
    LCQueryDataset,
    LCSFTDataset,
    split_train_into_train_and_eval,
)

logger = logging.get_logger(__name__)


def make_linguistic_calibration_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
) -> dict:
    """We use the supervised.py script for training either SFT models (e.g., Factuality SFT and LC SFT) or the
    ExtractAnswers surrogate function.

    Args:
        tokenizer: transformers.PreTrainedTokenizer
        data_args: argparse.Namespace
        training_args: argparse.Namespace

    Returns:
        dict: train_dataset, eval_dataset, data_collator
    """
    prompt_template = utils.read(data_args.prompt_template_path)

    if training_args.sft_type == "extract_answers":
        data_args.dataset_name = "reward_model_training"
        df_postprocessor = postprocess_extract_answers_df
    elif training_args.sft_type == "lc_sft":
        data_args.dataset_name = "sft_training"
        df_postprocessor = partial(postprocess_sft_df, sft_type="lc_sft")
    elif training_args.sft_type == "factuality_sft":
        data_args.dataset_name = "sft_training"
        df_postprocessor = partial(postprocess_sft_df, sft_type="factuality_sft")
    elif training_args.sft_type == "claude_distill":
        data_args.dataset_name = "sft_training"
        df_postprocessor = partial(postprocess_sft_df, sft_type="claude_distill")
    else:
        raise ValueError(f"Unknown SFT type: {training_args.sft_type}")

    logger.info(f"Loading dataset from {data_args.dataset_path}/{data_args.dataset_name}.")
    training_dataset = datasets.load_dataset(data_args.dataset_path, data_args.dataset_name)['train']

    # Convert to df
    training_df = training_dataset.to_pandas()

    # Load TriviaQA dataset
    trivia_qa_dataset = datasets.load_dataset("trivia_qa", "unfiltered.nocontext", split="train")
    trivia_qa_dataset = trivia_qa_dataset.to_pandas()

    # Join on the question_id
    training_df = training_df.merge(trivia_qa_dataset, on="question_id", how="left")

    # Postprocess
    training_df = training_df.apply(process_tqa_row, axis=1)

    # Use stratified split on the question_id (avoid duplicating question_id across train and eval)
    transformers.set_seed(training_args.seed)
    train_df, eval_df = stratified_split_on_question_id(df=training_df, eval_size=data_args.eval_size)

    train_dataset = LCSFTDataset(
        df=train_df,
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        df_postprocessor=df_postprocessor,
    )
    eval_dataset = LCSFTDataset(
        df=eval_df,
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        df_postprocessor=df_postprocessor,
    )

    data_collator = DataCollatorForLCSFTDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def process_tqa_row(row):
    # Get answers from the list of answer
    answer_list = row['answer']['aliases']
    top_answer = row['answer']['value']
    row['ground_truth_answer_list'] = answer_list
    row['ground_truth_top_answer'] = top_answer
    return row


def process_jeopardy_row(row):
    answer_list = [row['answer']]
    top_answer = row['answer']
    row['ground_truth_answer_list'] = answer_list
    row['ground_truth_top_answer'] = top_answer
    return row


def process_sciq_row(row):
    row['distractor_list'] = [row['distractor1'], row['distractor2'], row['distractor3']]
    row['ground_truth_top_answer'] = row['correct_answer']
    return row

def process_bioasq_row(row):
    row['question'] = row['body']
    row['question_id'] = row['id']
    row['ground_truth_top_answer'] = row['exact_answer'][0]
    return row


def stratified_split_on_question_id(
    df: pd.DataFrame,
    eval_size: int,
    random_state: int = 42,
):
    # Get the question_ids
    question_ids = df['question_id'].tolist()

    # Shuffle the question_ids
    np.random.seed(random_state)
    np.random.shuffle(question_ids)

    train_question_ids = question_ids[:-eval_size]
    eval_question_ids = question_ids[-eval_size:]

    # Split the dataset
    train_df, eval_df = df[df['question_id'].isin(train_question_ids)], df[df['question_id'].isin(eval_question_ids)]

    assert len(train_df) + len(eval_df) == len(df)
    assert len(set(train_df['question_id']).intersection(set(eval_df['question_id']))) == 0

    return train_df, eval_df


def make_linguistic_calibration_reward_modeling_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
) -> dict:
    prompt_template = utils.read(data_args.prompt_template_path)

    logger.info(f"Loading dataset from {data_args.dataset_path}/{data_args.dataset_name}.")
    reward_model_training_dataset = datasets.load_dataset(data_args.dataset_path, data_args.dataset_name)['train']

    # Convert to df
    reward_model_training_df = reward_model_training_dataset.to_pandas()

    # Load TriviaQA dataset
    trivia_qa_dataset = datasets.load_dataset("trivia_qa", "unfiltered.nocontext", split="train")
    trivia_qa_dataset = trivia_qa_dataset.to_pandas()

    # Join on the question_id
    reward_model_training_df = reward_model_training_df.merge(trivia_qa_dataset, on="question_id", how="left")

    # Postprocess
    reward_model_training_df = reward_model_training_df.apply(process_tqa_row, axis=1)

    if training_args.reward_model_type == "binary_correctness":
        df_postprocessor = postprocess_binary_correctness_reward_modeling_df
    elif training_args.reward_model_type == "forecast_probs":
        df_postprocessor = postprocess_forecast_probs_reward_modeling_df
    else:
        raise ValueError(f"Unknown reward model type: {training_args.reward_model_type}")

    if data_args.use_random_split:
        logger.warning("Using random split for training and evaluation.")
        train_dataset = LCRewardModelingDataset(
            df=reward_model_training_df,
            prompt_template=prompt_template,
            tokenizer=tokenizer,
            df_postprocessor=df_postprocessor,
            end_sequence_with_eos=training_args.end_sequence_with_eos,
        )
        train_dataset, eval_dataset = split_train_into_train_and_eval(
            train_dataset=train_dataset,
            eval_size=data_args.eval_size,
            seed=training_args.seed,
        )
    else:
        # Use stratified split on the question_id (avoid duplicating question_id across train and eval)
        logger.warning("Using stratified split on the question_id.")
        transformers.set_seed(training_args.seed)
        train_df, eval_df = stratified_split_on_question_id(
            df=reward_model_training_df,
            eval_size=data_args.eval_size,
            random_state=training_args.seed)

        train_dataset = LCRewardModelingDataset(
            df=train_df,
            prompt_template=prompt_template,
            tokenizer=tokenizer,
            df_postprocessor=df_postprocessor,
            end_sequence_with_eos=training_args.end_sequence_with_eos,
        )
        eval_dataset = LCRewardModelingDataset(
            df=eval_df,
            prompt_template=prompt_template,
            tokenizer=tokenizer,
            df_postprocessor=df_postprocessor,
            end_sequence_with_eos=training_args.end_sequence_with_eos,
            truncate_to_multiple_of_64=True
        )

    logger.warning(f"Training dataset size: {len(train_dataset)}")
    logger.warning(f"Evaluation dataset size: {len(eval_dataset)}")

    data_collator = DataCollatorForLCRewardModelingDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def make_linguistic_calibration_rl_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
) -> dict:
    # Load RL dataset using .load_dataset(). By default, we use the "ppo" split for train,
    #   and "ppo_validation" split for eval.
    policy_prompt_template = utils.read(data_args.policy_prompt_path)
    prompt_dataset = datasets.load_dataset(data_args.dataset_path, data_args.dataset_name)
    train_df = pd.concat([pd.DataFrame(prompt_dataset[split]) for split in data_args.train_splits])
    eval_df = pd.concat([pd.DataFrame(prompt_dataset[split]) for split in data_args.eval_splits])

    # Load TriviaQA dataset
    trivia_qa_dataset = datasets.load_dataset("trivia_qa", "unfiltered.nocontext", split="train")
    trivia_qa_dataset = trivia_qa_dataset.to_pandas()

    # Join on the question_id
    train_df = train_df.merge(trivia_qa_dataset, on="question_id", how="left")
    eval_df = eval_df.merge(trivia_qa_dataset, on="question_id", how="left")

    # Postprocess
    train_df = train_df.apply(process_tqa_row, axis=1)
    eval_df = eval_df.apply(process_tqa_row, axis=1)

    logger.info(f"Training dataset size: {len(train_df)}")
    logger.info(f"Evaluation dataset size: {len(eval_df)}")

    train_dataset = LCQueryDataset(
        df=train_df,
        prompt_template=policy_prompt_template,
        tokenizer=tokenizer,
        query_len=training_args.query_len,
    )
    eval_dataset = LCQueryDataset(
        df=eval_df,
        prompt_template=policy_prompt_template,
        tokenizer=tokenizer,
        query_len=training_args.query_len,
    )
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=DataCollatorForStackableDataset())


def load_eval_dataset(
    dataset_name: str,
    split: str
):
    assert dataset_name in [
        "trivia_qa", "jeopardy", "factscore", "sciq", "bioasq"], f"Unknown dataset name: {dataset_name}"

    # Load the paragraph generation prompts
    if dataset_name in ["trivia_qa", "jeopardy", "sciq", "bioasq"]:
        # First load our paragraph generation prompts
        eval_dataset = datasets.load_dataset(
            constants.HF_DATASETS_PATH,
            name=f"{dataset_name}_paragraph_generation",
            split=split,
            cache_dir=constants.DEFAULT_CACHE_DIR,
        )
        eval_dataset_df = pd.DataFrame(eval_dataset)

    if dataset_name == "trivia_qa":
        assert split in ["reward_model", "validation", "test"], f"Unknown split for trivia_qa: {split}"

        if split in {'reward_model', 'validation'}:
            trivia_qa_split = 'train'
        elif split == 'test':
            trivia_qa_split = 'validation'
        else:
            raise ValueError(f"Unknown split for evaluation: {split}")

        dataset = datasets.load_dataset("trivia_qa", "unfiltered.nocontext", split=trivia_qa_split)
        dataset_df = pd.DataFrame(dataset)

        # Join on the question_id
        eval_dataset_df = eval_dataset_df.merge(dataset_df, on="question_id", how="left")
        eval_dataset_df = eval_dataset_df.apply(process_tqa_row, axis=1)
    elif dataset_name == "jeopardy":
        assert split == "test", f"Unknown split for jeopardy: {split}"

        dataset = datasets.load_dataset("jeopardy", split="train")
        dataset_df = pd.DataFrame(dataset)

        # The question_id in eval_dataset_df is the index
        indices_to_take = eval_dataset_df['question_id'].values.tolist()
        dataset_df = dataset_df.iloc[indices_to_take]

        # Move answer column from dataset_df to eval_dataset_df
        eval_dataset_df['answer'] = dataset_df['answer'].values

        eval_dataset_df = eval_dataset_df.apply(process_jeopardy_row, axis=1)
    elif dataset_name == "sciq":
        assert split == "test", f"Unknown split for sciq: {split}"

        sciq_datasets = []

        # Load all sciq splits and concatenate
        for sciq_split in ["train", "validation", "test"]:
            dataset = datasets.load_dataset("allenai/sciq", split=sciq_split)
            dataset_df = pd.DataFrame(dataset)
            sciq_datasets.append(dataset_df)

        dataset_df = pd.concat(sciq_datasets)

        # The question_id in eval_dataset_df is the index
        indices_to_take = eval_dataset_df['question_id'].values.tolist()
        dataset_df = dataset_df.iloc[indices_to_take]

        # Move columns from dataset_df to eval_dataset_df
        columns_to_transfer = [
            'question', 'correct_answer', 'distractor1', 'distractor2', 'distractor3', 'support'
        ]
        for column in columns_to_transfer:
            eval_dataset_df[column] = dataset_df[column].values

        eval_dataset_df = eval_dataset_df.apply(process_sciq_row, axis=1)
    elif dataset_name == "bioasq":
        #   BioASQ is not available on Hugging Face Datasets.
        #   You need to manually install BioASQ from here: http://participants-area.bioasq.org/datasets/
        #   Specifically, you should install Task B dataset: Training 12b
        #   Set the following constant BIOASQ_JSON_PATH to /path/to/your/file/training12b_new.json
        bioasq_json = utils.jload(constants.BIOASQ_JSON_PATH)
        bioasq_df = pd.DataFrame(bioasq_json['questions'])
        bioasq_df = bioasq_df[bioasq_df['type'] == 'factoid']
        bioasq_df = bioasq_df.apply(process_bioasq_row, axis=1)
        bioasq_df = bioasq_df[['question_id', 'question', 'ground_truth_top_answer']]

        # Join on the question_id
        eval_dataset_df = eval_dataset_df.merge(bioasq_df, on="question_id", how="left")
    elif dataset_name == "factscore":
        assert split in ["validation", "test"], f"Unknown split for factscore: {split}"
        logger.warning("Loading factscore evaluation dataset. "
                       "Please cite FactScore (Min et al. 2023) if you evaluate on it.")

        if split == "validation":
            factscore_file_path = os.path.join(
                constants.FACTSCORE_CACHE_PATH, "data", "labeled", "prompt_entities.txt")
        elif split == "test":
            factscore_file_path = os.path.join(
                constants.FACTSCORE_CACHE_PATH, "data", "unlabeled", "prompt_entities.txt")
        else:
            raise ValueError(f"Unknown split for FactScore evaluation: {split}")

        # If the dataset is not already downloaded, download it
        if not os.path.exists(factscore_file_path):
            logger.warning("FactScore dataset not found. Downloading from Google Drive.")

            # FactScore data
            data_dest = os.path.join(constants.FACTSCORE_CACHE_PATH, "data.zip")
            download_file("1enz1PxwxeMr4FRF9dtpCPXaZQCBejuVF", data_dest, constants.FACTSCORE_CACHE_PATH)

            # Database for evaluation
            # This needs to be downloaded manually for now due to a permissions issue.
            # Refer to the `auto_eval_demo.ipynb` for instructions.
            # db_dest = os.path.join(constants.FACTSCORE_CACHE_PATH, "enwiki-20230401.db")
            # download_file(
            #     "1mekls6OGOKLmt7gYtHs0WGf5oTamTNat", db_dest, constants.FACTSCORE_CACHE_PATH)

        factscore_entities = utils.readlines(factscore_file_path)

        if split == "test":
            # There is an entity in the original FactScore dataset which is ambiguous,
            # leading to failing to retrieve the corresponding paragraphs. We make it unambiguous.
            new_factscore_entities = []
            for entity in factscore_entities:
                if entity == "Francisco Urroz":
                    new_factscore_entities.append("Francisco Urroz (footballer)")
                else:
                    new_factscore_entities.append(entity)

            factscore_entities = new_factscore_entities

        eval_dataset_df = pd.DataFrame(
            {
                "entity": factscore_entities,
                "paragraph_generation_prompt": [
                    f"Write a paragraph bio about {entity}." for entity in factscore_entities]
            }
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return eval_dataset_df
