import pandas as pd
from typing import Iterable

from linguistic_calibration.auto_annotations.prompt_retry_utils import run_interpretation_with_model
from linguistic_calibration.inference.decode import (
    get_decoding_args)
from linguistic_calibration.logging import get_logger

logger = get_logger(__name__)


def preprocess_deterministic_claims_for_fact_checking(
    generated_paragraphs: Iterable[str]
):
    # Deterministic claims can be either objective or subjective
    # We factcheck all objective claims
    assert isinstance(generated_paragraphs, list)
    assert len(generated_paragraphs) > 0

    output_generated_paragraphs = []

    for generated_paragraph in generated_paragraphs:
        if isinstance(generated_paragraph, str):
            try:
                generated_paragraph = eval(generated_paragraph)
            except Exception:
                raise ValueError(f"Could not eval generated paragraph {generated_paragraph}")
        elif not isinstance(generated_paragraph, dict):
            raise ValueError(f"Expected either a string or a dict, got "
                             f"{type(generated_paragraph)}")

        output_generated_paragraph = []
        for claim, claim_type in generated_paragraph.items():
            # Note that only claims annotated with 'Objective' are fact checked.
            # This implies that if a claim is annotated with something like
            # 'Partially Objective', it will not be fact checked.
            if claim_type == 'Objective':
                output_generated_paragraph.append((claim, claim_type))

        output_generated_paragraphs.append(output_generated_paragraph)

    n_claims_per_paragraph = [len(x) for x in output_generated_paragraphs]
    return output_generated_paragraphs, n_claims_per_paragraph


def preprocess_uncertainty_claims_for_lu_filtering(
    generated_paragraphs: Iterable[str]
):
    """Each claim is annotated with Subjective, Objective, and Full Uncertainty."""
    assert isinstance(generated_paragraphs, list) and len(generated_paragraphs) > 0

    output_generated_paragraphs = []

    for generated_paragraph in generated_paragraphs:
        if isinstance(generated_paragraph, str):
            try:
                generated_paragraph = eval(generated_paragraph)
            except Exception:
                raise ValueError(
                    f"Could not eval generated paragraph {generated_paragraph}")
        elif not isinstance(generated_paragraph, list):
            raise ValueError(f"Expected either a string or a list, got "
                             f"{type(generated_paragraph)}")

        assert isinstance(generated_paragraph, list)

        output_generated_paragraph = []
        for claim_tuple in generated_paragraph:
            if isinstance(claim_tuple, tuple):
                claim = claim_tuple[0]
                claim_type = claim_tuple[1]
                if claim_type in {'Subjective', 'Full Uncertainty'}:
                    continue
                elif claim_type == 'Objective':
                    output_generated_paragraph.append(claim)
                else:
                    raise ValueError(f"Unknown claim type {claim_type}")
            else:
                for claim_ in claim_tuple:
                    assert isinstance(claim_, tuple)
                    claim = claim_[0]
                    claim_type = claim_[1]
                    if claim_type in {'Subjective', 'Full Uncertainty'}:
                        continue
                    elif claim_type == 'Objective':
                        output_generated_paragraph.append(claim)
                    else:
                        raise ValueError(f"Unknown claim type {claim_type}")

        output_generated_paragraphs.append(output_generated_paragraph)

    n_claims_per_paragraph = [len(x) for x in output_generated_paragraphs]
    return output_generated_paragraphs, n_claims_per_paragraph


def preprocess_uncertainty_claims_for_fact_checking(
    generated_paragraphs: Iterable[str]
):
    assert isinstance(generated_paragraphs, list)
    assert len(generated_paragraphs) > 0

    output_generated_paragraphs = []

    for generated_paragraph in generated_paragraphs:
        if isinstance(generated_paragraph, str):
            try:
                generated_paragraph = eval(generated_paragraph)
            except Exception:
                raise ValueError(f"Could not eval generated paragraph {generated_paragraph}")
        elif not isinstance(generated_paragraph, list):
            raise ValueError(f"Expected either a string or a list, got "
                                f"{type(generated_paragraph)}")

        assert isinstance(generated_paragraph, list)

        output_generated_paragraph = []
        for claim_dict in generated_paragraph:
            assert isinstance(claim_dict, dict)
            claim_type = claim_dict['classification']

            # We run fact checks on the following claim types:
            if claim_type in {'Direct', 'Numerical Uncertainty', 'Linguistic Uncertainty'}:
                output_generated_paragraph.append((
                    claim_dict['core_claim'], claim_type))
            else:
                raise ValueError(f"Unknown claim type {claim_type}")

        output_generated_paragraphs.append(output_generated_paragraph)

    n_claims_per_paragraph = [len(x) for x in output_generated_paragraphs]
    return output_generated_paragraphs, n_claims_per_paragraph


def run_paragraph_interpretation(
    paragraphs_df: pd.DataFrame,
    interpretation_prompt_type: str,
    mode: str,
    interp_model_name: str = 'claude-2.0',
    per_device_batch_size: int = 8,
    temperature: float = 0.2,
    interp_output_path: str = None
):
    # Claim Decomposition needs:
    # - entity
    # - generated_paragraph (the biography)
    # Uncert. Filtering needs:
    # - entity
    # - interpretation__claim_decomposition (the decomposed claims)
    # Fact Checker needs:
    # - entity
    # - interpretation__claim_decomposition for non-confidence or
    #   interpretation__uncertainty_filter for confidence baselines
    entities = paragraphs_df['entity'].values

    if mode == 'claim_decomposition':
        generated_paragraphs = paragraphs_df['generated_paragraph'].values
    elif mode == 'claim_uncertainty_filter':
        generated_paragraphs = paragraphs_df['interpretation__claim_decomposition'].tolist()
        generated_paragraphs, n_claims_per_gen_paragraph = (
            preprocess_uncertainty_claims_for_lu_filtering(generated_paragraphs))
    elif mode == 'fact_checker':
        generated_paragraphs = paragraphs_df[
            'interpretation__claim_decomposition'].tolist()
        generated_paragraphs, n_claims_per_gen_paragraph = (
            preprocess_deterministic_claims_for_fact_checking(
                generated_paragraphs))
    elif mode == 'uncertainty_fact_checker':
        generated_paragraphs = paragraphs_df[
            'interpretation__claim_uncertainty_filter'].tolist()
        generated_paragraphs, n_claims_per_gen_paragraph = (
            preprocess_uncertainty_claims_for_fact_checking(generated_paragraphs))
    else:
        raise ValueError(f'Unknown mode {mode}')

    decoding_args = get_decoding_args(interp_model_name)
    if temperature is not None:
        decoding_args.temperature = temperature
        logger.info(f"Setting {mode} temperature to {temperature}.")

    decoding_args.max_tokens = 5000
    logger.info(f"Setting {mode} max_tokens to {decoding_args.max_tokens}.")

    # Run interpretation
    interp_df_data = run_interpretation_with_model(
        data_dict={
            'entity': entities,
            'generated_paragraph': generated_paragraphs,
        },
        decoding_args=decoding_args,
        model_name=interp_model_name,
        interpretation_prompt_type=interpretation_prompt_type,
        per_device_batch_size=per_device_batch_size)

    logger.info(f"Generated {len(interp_df_data['interpretation'])} "
                 f"interpretations for mode {mode}.")

    # Add new fields to DataFrame
    if 'cot_reasoning_path' in interp_df_data:
        paragraphs_df[f"cot_reasoning_path__{mode}"] = interp_df_data["cot_reasoning_path"]

    paragraphs_df[f"{mode}_prompt"] = interp_df_data["interpretation_prompt"]
    paragraphs_df[f"interpretation__{mode}"] = interp_df_data["interpretation"]

    # Save paragraphs
    paragraphs_df.to_csv(interp_output_path, index=False)
    logger.info(f"Wrote paragraph interpretations to {interp_output_path}")

    del paragraphs_df
    paragraphs_df = pd.read_csv(interp_output_path)
    return paragraphs_df
