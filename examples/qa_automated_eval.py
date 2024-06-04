import fire
import os
import pandas as pd
from transformers import set_seed

from linguistic_calibration import constants
from linguistic_calibration.auto_annotations.qa_auto_eval_utils import (
    run_paragraph_generation,
    run_extraction_or_forecasting,
    is_fewshot_icl_evaluation,
    postprocess_fewshot_icl_evaluation,
    unroll_answer_extraction_df_for_forecasting,
    preprocess_prob_forecast_df_for_semantic_equivalence_check,
    get_deterministic_prob_forecasts_df
)
from linguistic_calibration.common import assert_models_available
from linguistic_calibration.logging import get_logger

logger = get_logger(__name__)


def main(
    paragraph_generator_model_name: str = 'llama-2-7b-hf',
    paragraph_generation_prompt: str = 'generate_paragraphs_llama_trivia_qa_icl_8shot',
    answer_extractor_model_name: str = 'claude-2.0',
    answer_extractor_prompt: str = 'eval/extract_answers_claude_10shot',
    forecast_probs_model_name: str = 'claude-2.0',
    forecast_probs_prompt: str = 'eval/forecast_probs_claude_0shot',
    semantic_equivalence_model_name: str = 'claude-2.0',
    semantic_equivalence_prompt: str = 'eval/check_semantic_equivalence_10shot_batch10',
    dataset_name: str = 'trivia_qa',
    dataset_split: str = 'test',
    max_n_examples: int = None,
    seed: int = 42,
    skip_answer_extraction: bool = False,
    skip_forecast_probs: bool = False,
    per_device_batch_size: int = 8,
    generation_temperature: float = 0.3,
    extraction_temperature: float = 0.2,
    forecast_temperature: float = 0.2,
    semantic_equivalence_temperature: float = 0.2,
    output_root_dir: str = constants.DEFAULT_OUTPUT_DIR,
):
    """Run all steps of evaluation for a given paragraph generation model.

    Args:
        paragraph_generator_model_name: Name of the paragraph generator model.
        paragraph_generation_prompt: Answer generation prompt.
        answer_extractor_model_name: Name of the answer extractor model.
        answer_extractor_prompt: Answer extractor prompt.
        forecast_probs_model_name: Name of the model to forecast the
            probabilities of the GT and extracted answers.
        forecast_probs_prompt: Forecast prompt.
        semantic_equivalence_model_name: Name of the model to check semantic
            equivalence between the argmax answer and the ground-truth answer.
        semantic_equivalence_prompt: Semantic equivalence prompt.
        dataset_name: Name of the dataset to use.
        dataset_split: Dataset split to use.
        max_n_examples: Maximum number of examples to generate paragraphs for.
        seed: Random seed.
        skip_answer_extraction: If True, skip answer extraction.
        skip_forecast_probs: If True, skip forecasting of probabilities.
        per_device_batch_size: Batch size.
        generation_temperature: Temperature for sampling during paragraph
            generation.
        extraction_temperature: Temperature for sampling during answer
            extraction.
        forecast_temperature: Temperature for sampling during
            probability forecasting.
        semantic_equivalence_temperature: Temperature for sampling during
            semantic equivalence check.
        output_root_dir: Root directory for output.
    """
    seed = int(seed)
    args = {
        "max_n_examples": max_n_examples,
        "seed": seed,
        "paragraph_generator_model_name": paragraph_generator_model_name,
        "paragraph_generation_prompt": paragraph_generation_prompt,
        "answer_extractor_model_name": answer_extractor_model_name,
        "answer_extractor_prompt": answer_extractor_prompt,
        "forecast_probs_model_name": forecast_probs_model_name,
        "forecast_probs_prompt": forecast_probs_prompt,
        "semantic_equivalence_model_name": semantic_equivalence_model_name,
        "semantic_equivalence_prompt": semantic_equivalence_prompt,
        "skip_answer_extraction": skip_answer_extraction,
        "skip_forecast_probs": skip_forecast_probs,
        "dataset_name": dataset_name,  # "trivia_qa" or "jeopardy"
        "dataset_split": dataset_split,
        "per_device_batch_size": per_device_batch_size,
        "generation_temperature": generation_temperature,
        "extraction_temperature": extraction_temperature,
        "forecast_temperature": forecast_temperature,
        "semantic_equivalence_temperature": semantic_equivalence_temperature,
        "output_root_dir": output_root_dir,
    }

    logger.info(f"Running with args: {args}")

    interpretation_models = [
        paragraph_generator_model_name,
        semantic_equivalence_model_name
    ]
    if not skip_answer_extraction:
        interpretation_models.append(answer_extractor_model_name)
    if not skip_forecast_probs:
        interpretation_models.append(forecast_probs_model_name)

    assert_models_available(interpretation_models)

    paragraph_generation_short_model_name = paragraph_generator_model_name
    paragraph_generation_model_full_path = constants.SHORT_NAME_TO_MODEL_PATH.get(paragraph_generation_short_model_name)
    del paragraph_generator_model_name

    # Set seed
    set_seed(seed)

    paragraph_generation_output_dir = os.path.join(
        output_root_dir,
        "paragraph_generation",
        dataset_name,
        dataset_split,
        paragraph_generation_short_model_name,
        f"max_ex-{max_n_examples}--"
        f"seed-{seed}",
        f"gen_prompt-{paragraph_generation_prompt.replace('/', '__')}",
        f"gen_temp-{generation_temperature}")
    os.makedirs(paragraph_generation_output_dir, exist_ok=True)

    logger.info('Generating or loading paragraphs...')

    # 1. Answer Generation

    # Try to load paragraphs from disk
    paragraph_generation_output_path = os.path.join(
        paragraph_generation_output_dir, "paragraphs.csv")
    if os.path.exists(paragraph_generation_output_path):
        logger.info(f"Loading paragraphs from {paragraph_generation_output_path}")
        data_df = pd.read_csv(paragraph_generation_output_path)
        logger.info(f"Loaded paragraphs from {paragraph_generation_output_path}")
    else:
        # Generate paragraphs
        data_df = run_paragraph_generation(
            max_n_examples=max_n_examples,
            paragraph_generator_model_name=paragraph_generation_model_full_path,
            paragraph_generation_prompt=paragraph_generation_prompt,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            per_device_batch_size=per_device_batch_size,
            temperature=generation_temperature,
            paragraph_generation_output_path=paragraph_generation_output_path)

    # 1.5. For few-shot ICL evaluation with base models, postprocess data_df.
    if is_fewshot_icl_evaluation(paragraph_generation_short_model_name):
        logger.info('Detected few-shot ICL evaluation with base model. '
                    'Postprocessing data_df...')
        data_df = postprocess_fewshot_icl_evaluation(data_df)
        logger.info('Done!')

    # 2. Answer Extraction
    # For each generated paragraph, extract answers using the answer extractor
    if skip_answer_extraction:
        logger.info('Skipping answer extraction.')

        # Just directly use the paragraphs from paragraph generation for probability forecasting
        unrolled_answer_extractions_df = data_df
    else:
        answer_extraction_output_dir = os.path.join(
            output_root_dir,
            "answer_extraction",
            dataset_name,
            dataset_split,
            paragraph_generation_short_model_name,
            answer_extractor_model_name,
            f"max_ex-{max_n_examples}--"
            f"seed-{seed}",
            f"gen_prompt-{paragraph_generation_prompt.replace('/', '__')}",
            f"extr_prompt-{answer_extractor_prompt.replace('/', '__')}",
            f"gen_temp-{generation_temperature}",
            f"ext_temp-{extraction_temperature}")
        os.makedirs(answer_extraction_output_dir, exist_ok=True)

        logger.info('Generating or loading answer extractions...')

        # Try to load answer extractions from disk
        answer_extraction_output_path = os.path.join(
            answer_extraction_output_dir, "answer_extractions.csv")
        if os.path.exists(answer_extraction_output_path):
            logger.info(f"Loading answer extractions from "
                        f"{answer_extraction_output_path}")
            answer_extractions_df = pd.read_csv(answer_extraction_output_path)
            logger.info(f"Loaded answer extractions from "
                        f"{answer_extraction_output_path}")
        else:
            # Extract answers
            answer_extractions_df = run_extraction_or_forecasting(
                data_df=data_df,
                interpretation_prompt_type=answer_extractor_prompt,
                mode='answer_extraction',
                interp_model_name=answer_extractor_model_name,
                per_device_batch_size=per_device_batch_size,
                temperature=extraction_temperature,
                extraction_or_forecasting_output_path=answer_extraction_output_path)

        # 2.5. Prepare answer_extractions_df for forecasting
        logger.info('Preparing answer extractions for forecasting...')
        unrolled_answer_extractions_df = (
            unroll_answer_extraction_df_for_forecasting(answer_extractions_df))
        logger.info('Done!')

    # 3. Forecasting
    # For the ground-truth answer and for each extracted answer,
    #   forecast the probabilities of the answer using the forecasting
    #   model.
    if skip_forecast_probs:
        # Should be used when we are using the following pipeline on a
        #  non-confidence model:
        # 1. Answer extraction
        # 2. Take first extracted answer, if any
        # 3. Run semantic equivalence check
        logger.info('Skipping forecasting. Using first extracted answer '
                    'directly for semantic equivalence check.')
        prob_forecasts_df = get_deterministic_prob_forecasts_df(
            unrolled_answer_extractions_df)
    else:
        forecast_probs_output_dir = os.path.join(
            output_root_dir,
            "forecast_probs",
            dataset_name,
            dataset_split,
            paragraph_generation_short_model_name,
            answer_extractor_model_name,
            forecast_probs_model_name,
            f"skip_answer_extraction-{skip_answer_extraction}--"
            f"max_ex-{max_n_examples}--"
            f"seed-{seed}",
            f"gen_prompt-{paragraph_generation_prompt.replace('/', '__')}",
            f"extr_prompt-{answer_extractor_prompt.replace('/', '__')}",
            f"forecast_prompt-{forecast_probs_prompt.replace('/', '__')}",
            f"gen_temp-{generation_temperature}",
            f"ext_temp-{extraction_temperature}",
            f"forecast_temp-{forecast_temperature}")
        os.makedirs(forecast_probs_output_dir, exist_ok=True)
        logger.info('Generating or loading forecasts...')

        # Try to load forecasts from disk
        forecast_probs_output_path = os.path.join(
            forecast_probs_output_dir, "probability_forecasts.csv")
        if os.path.exists(forecast_probs_output_path):
            logger.info(f"Loading prob forecasts from "
                        f"{forecast_probs_output_path}")
            prob_forecasts_df = pd.read_csv(forecast_probs_output_path)
            logger.info(f"Loaded prob forecasts from "
                        f"{forecast_probs_output_path}")
        else:
            # Forecast probabilities
            prob_forecasts_df = run_extraction_or_forecasting(
                data_df=unrolled_answer_extractions_df,
                interpretation_prompt_type=forecast_probs_prompt,
                mode='forecast_probs',
                interp_model_name=forecast_probs_model_name,
                per_device_batch_size=per_device_batch_size,
                temperature=forecast_temperature,
                extraction_or_forecasting_output_path=forecast_probs_output_path)

    # 4. Semantic Equivalence
    # If we did answer extraction, we check if the extracted answer with the
    #  highest probability is semantically equivalent to the ground-truth
    #  answer. We break ties by taking the first extracted answer that
    #  appears in the generated paragraph.
    if not skip_answer_extraction:
        semantic_equivalence_output_dir = os.path.join(
            output_root_dir,
            "semantic_equivalence",
            dataset_name,
            dataset_split,
            paragraph_generation_short_model_name,
            answer_extractor_model_name,
            forecast_probs_model_name,
            semantic_equivalence_model_name,
            f"skip_forecast_probs-{skip_forecast_probs}--"
            f"max_ex-{max_n_examples}--"
            f"seed-{seed}",
            f"gen_prompt-{paragraph_generation_prompt.replace('/', '__')}",
            f"extr_prompt-{answer_extractor_prompt.replace('/', '__')}",
            f"forecast_prompt-{forecast_probs_prompt.replace('/', '__')}",
            f"sem_eq_prompt-{semantic_equivalence_prompt.replace('/', '__')}",
            f"gen_temp-{generation_temperature}",
            f"ext_temp-{extraction_temperature}",
            f"forecast_temp-{forecast_temperature}",
            f"sem_eq_temp-{semantic_equivalence_temperature}")
        os.makedirs(semantic_equivalence_output_dir, exist_ok=True)
        logger.info('Generating or loading semantic equivalence results...')

        # Try to load semantic equivalence results from disk
        semantic_equivalence_output_path = os.path.join(
            semantic_equivalence_output_dir, "semantic_equivalence.csv")
        if os.path.exists(semantic_equivalence_output_path):
            logger.info(f"Semantic equivalence results already exist at "
                        f"{semantic_equivalence_output_path}.")
        else:
            # Preprocess probability forecasts df
            semantic_equivalence_input_df = (
                preprocess_prob_forecast_df_for_semantic_equivalence_check(
                    prob_forecasts_df))

            # Run semantic equivalence prompting
            run_extraction_or_forecasting(
                data_df=semantic_equivalence_input_df,
                interpretation_prompt_type=semantic_equivalence_prompt,
                mode='semantic_equivalence',
                interp_model_name=semantic_equivalence_model_name,
                per_device_batch_size=per_device_batch_size,
                temperature=semantic_equivalence_temperature,
                extraction_or_forecasting_output_path=semantic_equivalence_output_path)

    logger.info('Done!')


if __name__ == "__main__":
    fire.Fire(main)
