import os
import pandas as pd
import pathlib
from collections import defaultdict

from linguistic_calibration import constants, utils
from linguistic_calibration.auto_annotations.prompt_retry_utils import run_interpretation_with_model
from linguistic_calibration.data_utils import load_eval_dataset
from linguistic_calibration.inference.decode import (
    decode_prompts_with_model, get_text_from_completions, get_decoding_args)
from linguistic_calibration.logging import get_logger
from linguistic_calibration.openai_utils import requires_chatml

logger = get_logger(__name__)


def is_fewshot_icl_evaluation(model_name):
    return model_name in constants.BASE_MODELS


def postprocess_fewshot_icl_evaluation(
    paragraphs_df: pd.DataFrame
) -> pd.DataFrame:
    """Postprocess the paragraphs dataframe for few-shot ICL evaluation."""
    def postprocess_paragraphs_df_row(row):
        row['unprocessed_generated_paragraph'] = row['generated_paragraph']
        row['generated_paragraph'] = row['generated_paragraph'].split('\n\n')[0]
        return row

    return paragraphs_df.apply(postprocess_paragraphs_df_row, axis=1)


def unroll_answer_extraction_df_for_forecasting(df):
    new_df_data = defaultdict(list)
    all_answers_to_forecast = []

    keys_list = list(list(df.iterrows())[0][1].keys())
    for i, row in df.iterrows():
        try:
            # Get answer extractions
            answer_extractions = eval(row['interpretation__answer_extraction'])
        except Exception:
            answer_extractions = [None]

        # We will forecast for the GT answer as well as the extracted answers
        answers_to_forecast = [
            row['ground_truth_top_answer']] + answer_extractions

        # Add a row for each answer to forecast
        for answer_to_forecast in answers_to_forecast:
            # Add all keys to new_df_data
            for key in keys_list:
                new_df_data[key].append(row[key])

            all_answers_to_forecast.append(answer_to_forecast)

    # Move the actual GT answers to another column
    new_df_data['actual_ground_truth_top_answer'] = new_df_data[
        'ground_truth_top_answer']
    new_df_data['ground_truth_top_answer'] = all_answers_to_forecast

    return pd.DataFrame(new_df_data)


def get_deterministic_prob_forecasts_df(unrolled_answer_extractions_df):
    # We simply set the prob forecast for all entries to 1.
    unrolled_answer_extractions_df[f"cot_reasoning_path__forecast_probs"] = None
    unrolled_answer_extractions_df[f"forecast_probs_prompt"] = None
    unrolled_answer_extractions_df[f"interpretation__forecast_probs"] = 1.0
    return unrolled_answer_extractions_df


def preprocess_prob_forecast_df_for_semantic_equivalence_check(df):
    """For each question_id, we have one row for the ground-truth answer,
    and then k rows for each of the extracted answers (where k varies per question_id).

    We construct a new DataFrame with
     - ground_truth_top_answer := actual_ground_truth_top_answer
     - generated_answer := argmax extracted answer

    Then the semantic_equivalence_check can take this as input to check if the
        generated answer is semantically equivalent to the ground-truth answer.

     Argmax is decided by the confidence score. In the case of ties,
        we take the first extracted answer that appears in the generated
        paragraph.
    """
    question_id_to_extracted_answer_and_conf_score = defaultdict(list)
    question_id_to_actual_ground_truth_top_answer = {}
    question_id_to_question = {}
    question_id_to_ground_truth_answer_list = {}
    processed_question_ids = []
    for i, row in df.iterrows():
        question_id = row['question_id']

        # We are at an extracted answer row (there may be many for each question_id)
        if question_id in set(processed_question_ids):
            question_id_to_extracted_answer_and_conf_score[question_id].append(
                (row['ground_truth_top_answer'], row['interpretation__forecast_probs']))

        # We are at the ground-truth answer row (these come before the extracted answer rows)
        else:
            processed_question_ids.append(question_id)
            question_id_to_actual_ground_truth_top_answer[question_id] = (
                row['actual_ground_truth_top_answer'])
            question_id_to_question[question_id] = row['question']

            if 'ground_truth_answer_list' in row:
                question_id_to_ground_truth_answer_list[question_id] = (
                    row['ground_truth_answer_list'])

    # Construct new DataFrame
    # First get argmax answers
    argmax_answers = []
    for question_id in processed_question_ids:
        extracted_answer_and_conf_score_list = (
            question_id_to_extracted_answer_and_conf_score[question_id])

        if len(extracted_answer_and_conf_score_list) == 0:
            argmax_answers.append(None)
            logger.info(f'No extracted answers for question_id {question_id}')
            continue

        # If there are multiple extracted answers with the same confidence score,
        #   just take the first one
        index_of_top_answer = 0
        for j in range(1, len(extracted_answer_and_conf_score_list)):
            if extracted_answer_and_conf_score_list[j][1] > \
                    extracted_answer_and_conf_score_list[index_of_top_answer][1]:
                index_of_top_answer = j

        argmax_answers.append(
            extracted_answer_and_conf_score_list[index_of_top_answer][0])

    new_df_data = {
        'question_id': processed_question_ids,
        'question': [question_id_to_question[question_id]
                     for question_id in processed_question_ids],
        'ground_truth_top_answer': [question_id_to_actual_ground_truth_top_answer[question_id]
                                    for question_id in processed_question_ids],
        'generated_paragraph': argmax_answers,
    }

    if len(question_id_to_ground_truth_answer_list) > 0:
        ground_truth_answer_list = [question_id_to_ground_truth_answer_list[question_id]
                                    for question_id in processed_question_ids]
        new_df_data['ground_truth_answer_list'] = ground_truth_answer_list

    return pd.DataFrame(new_df_data)


def format_paragraph_generation_prompt(
    data_df: pd.DataFrame,
    model_name: str,
    paragraph_generation_prompt_type: str,
):
    # Load paragraph generation prompt template from directory
    prompt_template_path = os.path.join(
        pathlib.Path(__file__).parent.parent,
        "prompts",
        "paragraph_generation",
        f"{paragraph_generation_prompt_type}.txt"
    )
    prompt_template = utils.read(prompt_template_path)

    if model_name in constants.ANTHROPIC_MODELS:
        # Add leading "\n\n" to prompt
        prompt_template = "\n\n" + prompt_template
    
    prompts = []
    data_dicts = data_df.to_dict(orient='records')
    for data_dict in data_dicts:
        prompts.append(prompt_template.format(**data_dict))
        
    if requires_chatml(model_name):
        prompts = [[{
            "role": "user",
            "content": prompt_str
        }] for prompt_str in prompts]

    return prompts


def run_paragraph_generation(
    max_n_examples: int,
    paragraph_generator_model_name: str,
    paragraph_generation_prompt: str,
    dataset_name: str,
    dataset_split: str,
    per_device_batch_size: int = 8,
    temperature: float = 0.3,
    paragraph_generation_output_path: str = None,
):
    """Run paragraph generation.

    Args:
        max_n_examples: Maximum number of examples to generate paragraphs for.
        paragraph_generator_model_name: Name of the paragraph generator model.
        paragraph_generation_prompt: Prompt for paragraph generation.
        dataset_name: Name of the dataset to use.
        dataset_split: Dataset split to use.
        per_device_batch_size: Batch size.
        temperature: Temperature for sampling.
        paragraph_generation_output_path: Path to write paragraphs to.
    """
    logger.info(f"Generating paragraphs and writing to "
                f"{paragraph_generation_output_path}")

    data_df = load_eval_dataset(dataset_name=dataset_name, split=dataset_split)

    # Take first max_n_examples
    if max_n_examples is not None:
        data_df = data_df.head(max_n_examples)
        logger.warning(f"Only using the first {max_n_examples} examples.")

    # Generate paragraphs
    decoding_args = get_decoding_args(paragraph_generator_model_name)

    if temperature is not None:
        decoding_args.temperature = temperature
        logger.info(f"Setting paragraph generation temperature to {temperature}.")

    if hasattr(decoding_args, 'num_return_sequences'):
        decoding_args.num_return_sequences = 1
        logger.info(f"Setting num_return_sequences to 1.")

    paragraph_generations = decode_prompts_with_model(
        prompts=format_paragraph_generation_prompt(
            data_df, paragraph_generator_model_name, paragraph_generation_prompt),
        model_name=paragraph_generator_model_name,
        decoding_args=decoding_args,
        per_device_batch_size=per_device_batch_size,
    )
    paragraph_generations = get_text_from_completions(paragraph_generations)

    logger.info(f"Generated {len(paragraph_generations)} long-form generations.")

    # Add results of paragraph generation to data_dict
    data_df["generated_paragraph"] = paragraph_generations
    data_df["paragraph_generation_prompt"] = (
            [paragraph_generation_prompt] * len(paragraph_generations))
    data_df["dataset_name"] = [dataset_name] * len(paragraph_generations)
    data_df["dataset_split"] = [dataset_split] * len(paragraph_generations)

    # Save paragraphs
    data_df.to_csv(paragraph_generation_output_path, index=False)
    logger.info(f"Wrote paragraphs to {paragraph_generation_output_path}")

    del data_df
    data_df = pd.read_csv(paragraph_generation_output_path)
    return data_df


def run_extraction_or_forecasting(
    data_df: pd.DataFrame,
    interpretation_prompt_type: str,
    mode: str,
    interp_model_name: str,
    per_device_batch_size: int,
    temperature: float,
    extraction_or_forecasting_output_path: str,
):
    """Run extraction or forecasting.

    Args:
        data_df: DataFrame with generated paragraphs.
        interpretation_prompt_type: Prompt type for interpretation.
        mode: Mode for interpretation.
        interp_model_name: Name of the interpretation model.
        per_device_batch_size: Batch size.
        temperature: Temperature for sampling.
        extraction_or_forecasting_output_path: Path to write extraction or forecast results to.
    """
    decoding_args = get_decoding_args(interp_model_name)
    if temperature is not None:
        decoding_args.temperature = temperature
        logger.info(f"Setting {mode} temperature to {temperature}.")

    generated_paragraphs = data_df['generated_paragraph'].values  # List of generated paragraphs.
    questions = data_df['question'].values  # List of questions.

    # List of top ground truth answers. If we are performing answer extraction, this is unused.
    # If we are performing probability forecasting, this is equal to a possible answer choice
    # y' \in \mathcal{Y} for the question x.
    ground_truth_top_answers = data_df['ground_truth_top_answer'].values

    # Run interpretation
    interp_df_data = run_interpretation_with_model(
        data_dict={
            'generated_paragraph': generated_paragraphs,
            'question': questions,
            'ground_truth_top_answer': ground_truth_top_answers,
        },
        decoding_args=decoding_args,
        model_name=interp_model_name,
        interpretation_prompt_type=interpretation_prompt_type,
        per_device_batch_size=per_device_batch_size)

    logger.info(f"Generated {len(interp_df_data['interpretation'])} "
                f"interpretations for mode {mode}.")

    # Add new fields to DataFrame
    if 'cot_reasoning_path' in interp_df_data:
        data_df[f"cot_reasoning_path__{mode}"] = interp_df_data["cot_reasoning_path"]

    data_df[f"{mode}_prompt"] = interp_df_data["interpretation_prompt"]
    data_df[f"interpretation__{mode}"] = interp_df_data["interpretation"]

    # Save answers
    data_df.to_csv(extraction_or_forecasting_output_path, index=False)
    logger.info(f"Wrote answers to {extraction_or_forecasting_output_path}")

    del data_df
    data_df = pd.read_csv(extraction_or_forecasting_output_path)
    return data_df
