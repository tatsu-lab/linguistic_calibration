import fire
import os
import pandas as pd
import pathlib
from transformers import set_seed

from linguistic_calibration import constants
from linguistic_calibration.auto_annotations.biography_generation_auto_eval_utils import (
    run_paragraph_interpretation
)
from linguistic_calibration.auto_annotations.qa_auto_eval_utils import (
    run_paragraph_generation,
    is_fewshot_icl_evaluation,
    postprocess_fewshot_icl_evaluation
)
from linguistic_calibration.common import assert_models_available
from linguistic_calibration.logging import get_logger

logger = get_logger(__name__)


def main(
    paragraph_generator_model_name: str,
    paragraph_generation_prompt: str,
    claim_decomposition_model_name: str = "claude-2.0",
    claim_decomposition_prompt: str = "biography_generation_eval/confidence_decompose_claims_claude_8shot",
    claim_uncertainty_filter_model_name: str = "claude-2.0",
    claim_uncertainty_filter_prompt: str = "biography_generation_eval/confidence_filter_claude_1shot",
    atomic_fact_checker_model_name: str = "claude-2.0",
    dataset_split: str = "test",
    max_n_examples: int = None,
    seed: int = 42,
    per_device_batch_size: int = 8,
    generation_temperature: float = 0.3,
    decomposition_temperature: float = 0.2,
    filter_temperature: float = 0.2,
    fact_check_temperature: float = 0.2,
    output_root_dir: str = pathlib.Path(__file__).parent / "output",
):
    """Run all steps of evaluation for a given paragraph generation model on
        open-ended biography generation with per-claim evaluation.

    1. Generate paragraphs with `paragraph_generator_model_name`, and store them in a
        CSV file.
    2. Decompose paragraphs into atomic claims with `claim_decomposition_model_name`,
        and store them in a CSV file.
    3. Optionally, for uncertainty-enabled models, filter claims with
        `claim_uncertainty_filter_model_name`, and store them in a CSV file.
    4. Fact check claims with `atomic_fact_checker_model_name`, and store them
        in a CSV file.

    Args:
        max_n_examples: Maximum number of examples to generate paragraphs for.
        seed: Random seed.
        paragraph_generator_model_name: Name of the paragraph generator model.
        paragraph_generation_prompt: Paragraph generation prompt.
        claim_decomposition_model_name: Name of the claim decomposition model.
        claim_decomposition_prompt: Claim decomposition prompt.
        claim_uncertainty_filter_model_name: Name of the claim uncertainty
            filter model.
        claim_uncertainty_filter_prompt: Claim uncertainty filter prompt.
        atomic_fact_checker_model_name: Name of the atomic fact checker model.
        dataset_split: Dataset split to use.
        per_device_batch_size: Batch size.
        generation_temperature: Temperature for sampling during paragraph
            generation.
        decomposition_temperature: Temperature for sampling during
            decomposition.
        filter_temperature: Temperature for sampling during filtering.
        fact_check_temperature: Temperature for sampling during fact checking.
        output_root_dir: Root directory for output.
    """
    assert atomic_fact_checker_model_name == 'claude-2.0', (
        'For now we only support Claude models for fact checking. TODO: support other model families.')

    seed = int(seed)
    args = {
        "max_n_examples": max_n_examples,
        "seed": seed,
        "paragraph_generator_model_name": paragraph_generator_model_name,
        "paragraph_generation_prompt": paragraph_generation_prompt,
        "claim_decomposition_model_name": claim_decomposition_model_name,
        "claim_decomposition_prompt": claim_decomposition_prompt,
        "claim_uncertainty_filter_model_name": claim_uncertainty_filter_model_name,
        "claim_uncertainty_filter_prompt": claim_uncertainty_filter_prompt,
        "atomic_fact_checker_model_name": atomic_fact_checker_model_name,
        "dataset_name": "factscore",
        "dataset_split": dataset_split,
        "per_device_batch_size": per_device_batch_size,
        "generation_temperature": generation_temperature,
        "decomposition_temperature": decomposition_temperature,
        "filter_temperature": filter_temperature,
        "fact_check_temperature": fact_check_temperature,
        "output_root_dir": output_root_dir,
    }
    logger.info(f"Running with args: {args}")

    assert_models_available([
        paragraph_generator_model_name,
        claim_decomposition_model_name,
        claim_uncertainty_filter_model_name,
        atomic_fact_checker_model_name])

    paragraph_generation_short_model_name = paragraph_generator_model_name
    paragraph_generation_model_full_path = constants.SHORT_NAME_TO_MODEL_PATH.get(paragraph_generation_short_model_name)
    del paragraph_generator_model_name

    # Set seed
    set_seed(seed)

    paragraph_generation_output_dir = os.path.join(
        output_root_dir,
        "paragraph_generation",
        "factscore",
        dataset_split,
        paragraph_generation_short_model_name,
        f"max_ex-{max_n_examples}--"
        f"seed-{seed}",
        f"gen_prompt-{paragraph_generation_prompt.replace('/', '__')}",
        f"gen_temp-{generation_temperature}")
    os.makedirs(paragraph_generation_output_dir, exist_ok=True)

    logger.info('Generating or loading paragraphs...')

    # 1. Paragraph Generation

    # Try to load paragraphs from disk
    paragraph_generation_output_path = os.path.join(
        paragraph_generation_output_dir, "paragraphs.csv")
    if os.path.exists(paragraph_generation_output_path):
        logger.info(f"Loading paragraphs from {paragraph_generation_output_path}")
        paragraphs_df = pd.read_csv(paragraph_generation_output_path)
        logger.info(f"Loaded paragraphs from {paragraph_generation_output_path}")
    else:
        # Generate paragraphs
        paragraphs_df = run_paragraph_generation(
            max_n_examples=max_n_examples,
            paragraph_generator_model_name=paragraph_generation_model_full_path,
            paragraph_generation_prompt=paragraph_generation_prompt,
            dataset_name='factscore',
            dataset_split=dataset_split,
            per_device_batch_size=per_device_batch_size,
            temperature=generation_temperature,
            paragraph_generation_output_path=paragraph_generation_output_path)

    # 1.5. For few-shot ICL evaluation with base models, postprocess
    #  paragraphs_df.
    if is_fewshot_icl_evaluation(paragraph_generation_short_model_name):
        logger.info('Detected few-shot ICL evaluation with base model. '
                    'Postprocessing paragraphs_df...')
        paragraphs_df = postprocess_fewshot_icl_evaluation(paragraphs_df)
        logger.info('Done!')

    # 2. Claim Decomposition
    # For each generated paragraph, decompose it into atomic claims using the
    #   claim decomposition model.
    claim_decomposition_output_dir = os.path.join(
        output_root_dir,
        "claim_decomposition",
        "factscore",
        dataset_split,
        paragraph_generation_short_model_name,
        claim_decomposition_model_name,
        f"max_ex-{max_n_examples}--"
        f"seed-{seed}",
        f"gen_prompt-{paragraph_generation_prompt.replace('/', '__')}",
        f"decomp_prompt-{claim_decomposition_prompt.replace('/', '__')}",
        f"gen_temp-{generation_temperature}",
        f"decomp_temp-{decomposition_temperature}")
    os.makedirs(claim_decomposition_output_dir, exist_ok=True)

    logger.info('Generating or loading claim decompositions...')

    # Try to load claim decompositions from disk
    claim_decomposition_output_path = os.path.join(
        claim_decomposition_output_dir, "claim_decompositions.csv")
    if os.path.exists(claim_decomposition_output_path):
        logger.info(f"Loading claim decompositions from "
                    f"{claim_decomposition_output_path}")
        claim_decompositions_df = pd.read_csv(claim_decomposition_output_path)
        logger.info(f"Loaded claim decompositions from "
                    f"{claim_decomposition_output_path}")
    else:
        # Decompose claims
        claim_decompositions_df = run_paragraph_interpretation(
            paragraphs_df=paragraphs_df,
            interpretation_prompt_type=claim_decomposition_prompt,
            mode='claim_decomposition',
            interp_model_name=claim_decomposition_model_name,
            per_device_batch_size=per_device_batch_size,
            temperature=decomposition_temperature,
            interp_output_path=claim_decomposition_output_path)

    # 2.5. If we are using an uncertainty-enabled model, filter the claims
    #   using the claim uncertainty filter model.
    if 'nonconfidence' not in claim_decomposition_prompt:
        logger.info('Detected uncertainty-enabled model. Filtering claims...')
        claim_uncertainty_filter_output_dir = os.path.join(
            output_root_dir,
            "claim_uncertainty_filter",
            "factscore",
            dataset_split,
            paragraph_generation_short_model_name,
            claim_decomposition_model_name,
            claim_uncertainty_filter_model_name,
            f"max_ex-{max_n_examples}--"
            f"seed-{seed}",
            f"gen_prompt-{paragraph_generation_prompt.replace('/', '__')}",
            f"decomp_prompt-{claim_decomposition_prompt.replace('/', '__')}",
            f"filter_prompt-{claim_uncertainty_filter_prompt.replace('/', '__')}",
            f"gen_temp-{generation_temperature}",
            f"decomp_temp-{decomposition_temperature}",
            f"filter_temp-{filter_temperature}")
        os.makedirs(claim_uncertainty_filter_output_dir, exist_ok=True)

        logger.info('Generating or loading claim uncertainty filter results...')

        # Try to load claim uncertainty filter results from disk
        claim_uncertainty_filter_output_path = os.path.join(
            claim_uncertainty_filter_output_dir, "claim_uncertainty_filter.csv")
        if os.path.exists(claim_uncertainty_filter_output_path):
            logger.info(f"Loading claim uncertainty filter results from "
                        f"{claim_uncertainty_filter_output_path}")
            claim_uncertainty_filter_df = pd.read_csv(
                claim_uncertainty_filter_output_path)
            logger.info(f"Loaded claim uncertainty filter results from "
                        f"{claim_uncertainty_filter_output_path}")
        else:
            # Filter claims
            claim_uncertainty_filter_df = run_paragraph_interpretation(
                paragraphs_df=claim_decompositions_df,
                interpretation_prompt_type=claim_uncertainty_filter_prompt,
                mode='claim_uncertainty_filter',
                interp_model_name=claim_uncertainty_filter_model_name,
                per_device_batch_size=per_device_batch_size,
                temperature=filter_temperature,
                interp_output_path=claim_uncertainty_filter_output_path)

        fact_check_mode = 'uncertainty_fact_checker'
        input_df_for_fact_checking = claim_uncertainty_filter_df
    else:
        fact_check_mode = 'fact_checker'
        input_df_for_fact_checking = claim_decompositions_df

    # 3. Fact Checking
    # For each generated paragraph, fact check each atomic claim using the
    #   fact checker model.
    fact_checker_output_dir = os.path.join(
        output_root_dir,
        "fact_checker",
        "factscore",
        dataset_split,
        paragraph_generation_short_model_name,
        claim_decomposition_model_name,
        claim_uncertainty_filter_model_name,
        atomic_fact_checker_model_name,
        f"max_ex-{max_n_examples}--"
        f"seed-{seed}",
        f"gen_prompt-{paragraph_generation_prompt.replace('/', '__')}",
        f"decomp_prompt-{claim_decomposition_prompt.replace('/', '__')}",
        f"filter_prompt-{claim_uncertainty_filter_prompt.replace('/', '__')}",
        f"gen_temp-{generation_temperature}",
        f"decomp_temp-{decomposition_temperature}",
        f"filter_temp-{filter_temperature}",
        f"fact_check_temp-{fact_check_temperature}")
    os.makedirs(fact_checker_output_dir, exist_ok=True)

    logger.info('Generating or loading fact checker results...')

    # Try to load fact checker results from disk
    fact_checker_output_path = os.path.join(
        fact_checker_output_dir, "fact_checker.csv")
    if os.path.exists(fact_checker_output_path):
        logger.info(f"Loading fact checker results from "
                    f"{fact_checker_output_path}")
        _ = pd.read_csv(fact_checker_output_path)
        logger.info(f"Loaded fact checker results from "
                    f"{fact_checker_output_path}")
    else:
        # Fact check claims
        run_paragraph_interpretation(
            paragraphs_df=input_df_for_fact_checking,
            # We use a prompt identical to Min et al. (FactScore)
            interpretation_prompt_type='biography_generation_eval/fact_checker',
            mode=fact_check_mode,
            interp_model_name=atomic_fact_checker_model_name,
            per_device_batch_size=per_device_batch_size,
            temperature=fact_check_temperature,
            interp_output_path=fact_checker_output_path)

    logger.info('Done!')


if __name__ == "__main__":
    fire.Fire(main)