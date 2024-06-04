import os
from pathlib import Path

from linguistic_calibration import utils, constants
from linguistic_calibration.auto_annotations import prompt_processing_utils
from linguistic_calibration.auto_annotations.factscore_retrieval_utils import FactScorer
from linguistic_calibration.inference.decode import decode_prompts_with_model, get_text_from_completions
from linguistic_calibration.logging import get_logger
from linguistic_calibration.types import List, Callable, Optional, Mapping

logger = get_logger(__name__)

INTERPRETATION_PROMPT_TO_METADATA = {
    "train/extract_answers_claude_8shot": {
        "prompt_preprocessor": prompt_processing_utils.format_extract_answers_prompt,
        "prompt_postprocessor": prompt_processing_utils.JSONExtractAnswersAnnotatorPostprocessor(),
    },
    "train/forecast_probs_claude_0shot": {
        "prompt_preprocessor": prompt_processing_utils.format_forecast_probs_prompt,
        "prompt_postprocessor": prompt_processing_utils.JSONForecastProbsAnnotatorPostprocessor(
            extraction_key='Answer Probability, float in range [0.0, 1.0]'),
    },
    "train/score_binary_correctness_claude_0shot": {
        "prompt_preprocessor": prompt_processing_utils.format_forecast_probs_prompt,
        "prompt_postprocessor": prompt_processing_utils.JSONForecastProbsAnnotatorPostprocessor(
            extraction_key='Context is correct (1) or incorrect (0)'),
    },
    # QA Evaluation
    "eval/extract_answers_claude_10shot": {
        "prompt_preprocessor": prompt_processing_utils.format_extract_answers_prompt,
        "prompt_postprocessor": prompt_processing_utils.MultilineExtractAnswersAnnotatorPostprocessor(),
    },
    "eval/forecast_probs_claude_0shot": {
        "prompt_preprocessor": prompt_processing_utils.format_forecast_probs_prompt,
        "prompt_postprocessor": prompt_processing_utils.MultilineForecastProbsAnnotatorPostprocessor(),
    },
    "eval/check_semantic_equivalence_10shot_batch10": {
        "prompt_preprocessor": prompt_processing_utils.format_multiquery_prompt_semantic_equivalence,
        "prompt_postprocessor": prompt_processing_utils.SemanticEquivalenceAnnotatorPostprocessor(),
    },
    "eval/check_semantic_equivalence_10shot_batch10_claude_chat": {
        "prompt_preprocessor": prompt_processing_utils.format_multiquery_prompt_semantic_equivalence,
        "prompt_postprocessor": prompt_processing_utils.SemanticEquivalenceAnnotatorPostprocessor(),
    },
    # Biography Generation Evaluation
    # Decompose a generation into atomic claims, for a non-confidence model
    "biography_generation_eval/nonconfidence_decompose_claims_claude_8shot": {
        "prompt_preprocessor": prompt_processing_utils.format_factscore_decomposition_prompt,
        "prompt_postprocessor": prompt_processing_utils.JSONNonConfidenceClaimDecompositionAnnotatorPostprocessor(),
    },
    # Decompose a generation into atomic claims, for a confidence model
    "biography_generation_eval/confidence_decompose_claims_claude_8shot": {
        "prompt_preprocessor": prompt_processing_utils.format_factscore_decomposition_prompt,
        "prompt_postprocessor": prompt_processing_utils.JSONConfidenceClaimDecompositionAnnotatorPostprocessor(),
    },
    # Confidence/uncertainty filtering: do forecasting of confidences in each atomic claim
    "biography_generation_eval/confidence_filter_claude_1shot": {
        "prompt_preprocessor": prompt_processing_utils.format_factscore_confidence_filter_prompt,
        "prompt_postprocessor": prompt_processing_utils.MultilineConfidenceFilterAnnotatorPostprocessor(),
    },
    # Fact-checking: do interpretation of correctness in each atomic claim
    "biography_generation_eval/fact_checker": {
        "prompt_preprocessor": None,  # We construct prompts using the FactScore prompt format
        "prompt_postprocessor": prompt_processing_utils.FactCheckAnnotatorPostprocessor(),
    },
}


def generate_and_parse_completions(
    prompts: List[str],
    decoding_args,
    model_name: str,
    per_device_batch_size: int,
    postprocess_fn: Callable,
    is_multiquery: bool,
    num_procs: int = 7,
    n_expected_outputs_per_example: Optional[List[int]] = None,
    mixed_precision: Optional[str] = None,
    **kwargs
):
    if is_multiquery:
        assert n_expected_outputs_per_example is None
    if n_expected_outputs_per_example is not None:
        assert not is_multiquery

    logger.warning(f'Using {num_procs} processes for decoding from model {model_name}.')

    completions = decode_prompts_with_model(
        prompts=prompts,
        model_name=model_name,
        decoding_args=decoding_args,
        per_device_batch_size=per_device_batch_size,
        num_procs=num_procs,
        mixed_precision=mixed_precision,
    )

    completions_text = [response_str for response_str in get_text_from_completions(completions)]
    if is_multiquery:
        if postprocess_fn.is_cot:
            raise NotImplementedError
        else:
            interpretations = []
            for response_str in completions_text:
                try:
                    response_interpretations = postprocess_fn(response_str)
                except Exception:
                    logger.error("Error parsing response: %s", response_str)
                    response_interpretations = None

                interpretations.append(response_interpretations)
    elif n_expected_outputs_per_example is not None:
        logger.warning("Using n_expected_outputs_per_example to validate completions.")
        interpretations = []
        assert len(completions_text) == len(n_expected_outputs_per_example)
        for i, n_expected_outputs in enumerate(n_expected_outputs_per_example):
            interpretations.append(
                postprocess_fn(
                    completions_text[i],
                    n_expected_outputs=n_expected_outputs
                ))
    else:
        interpretations = [postprocess_fn(response_str) for response_str in completions_text]

    return completions_text, interpretations


def get_retry_indices_to_prompts(
    full_prompts_list: List[str],
    full_interpretations_list: List[str],
):
    # For any interpretations that are None, we will retry the prompt.
    index_in_original_list = []
    prompts_to_retry = []
    for i, (interpretation, prompt) in enumerate(
            utils.zip_(full_interpretations_list, full_prompts_list)):
        if interpretation is None:
            prompts_to_retry.append(prompt)
            index_in_original_list.append(i)

    return prompts_to_retry, index_in_original_list


def run_completion_retry_loop(
    prompts: List[str],
    decoding_args,
    model_name: str,
    per_device_batch_size: int,
    postprocess_fn: Callable,
    max_retry_loops: int,
    is_multiquery: bool,
    n_expected_outputs_per_example: Optional[List[int]] = None,
    **kwargs,
):
    current_prompts = prompts
    indices_in_original_list = list(range(len(current_prompts)))
    full_interpretations_list = [None for _ in range(len(current_prompts))]
    full_completions_text_list = [None for _ in range(len(current_prompts))]

    while len(current_prompts) > 0 and max_retry_loops > 0:
        logger.warning("Running retry loop with %d prompts", len(current_prompts))
        new_completions_text, new_interpretations = generate_and_parse_completions(
            prompts=current_prompts,
            decoding_args=decoding_args,
            model_name=model_name,
            per_device_batch_size=per_device_batch_size,
            postprocess_fn=postprocess_fn,
            is_multiquery=is_multiquery,
            n_expected_outputs_per_example=n_expected_outputs_per_example,
            **kwargs
        )

        # Update the full lists
        for i, index_in_original_list_elem in enumerate(indices_in_original_list):
            full_interpretations_list[index_in_original_list_elem] = new_interpretations[i]
            full_completions_text_list[index_in_original_list_elem] = new_completions_text[i]

        # Get the prompts to retry
        current_prompts, indices_in_original_list = get_retry_indices_to_prompts(
            full_prompts_list=prompts,
            full_interpretations_list=full_interpretations_list
        )
        max_retry_loops -= 1

    logger.warning('Terminated with %d prompts left to retry', len(current_prompts))
    return full_completions_text_list, full_interpretations_list


def run_interpretation_with_model(
    data_dict: Mapping[str, List],
    decoding_args,
    model_name: str,
    interpretation_prompt_type: Optional[str],
    per_device_batch_size: int = 8,
    mixed_precision: Optional[str] = 'bf16',
    max_retry_loops: int = 100,
    **kwargs
):
    """Given a list of generated paragraphs, questions, and ground truth answers,
        run answer extraction or other interpretation (forecasting, semantic equivalence check, etc.).

    Args:
        data_dict: Dictionary of data associated with the interpretation.
        decoding_args: Decoding arguments.
        model_name: Name of model to use for interpretation.
        interpretation_prompt_type: Type of interpretation prompt to use.
        per_device_batch_size: Batch size per device.
        mixed_precision: Whether to use mixed precision.
        max_retry_loops: Maximum number of retry loops to run.

    Returns:
        List of interpretations, and a dictionary of data
            associated with each interpretation.
    """
    logger.info(f"Running interpretation with model: {model_name}")
    logger.info(f"Interpretation prompt type: {interpretation_prompt_type}")

    # Load the prompt
    if interpretation_prompt_type == 'biography_generation_eval/fact_checker':
        logger.warning("Using FactScore fact-checking prompt.")
        # We construct prompts using the FactScore prompt format
        prompt_template_or_dict = None
    else:
        prompt_path = os.path.join(Path(__file__).parent.parent, 'prompts', f"{interpretation_prompt_type}.txt")
        prompt_template_or_dict = utils.read(prompt_path)

        if model_name in constants.ANTHROPIC_MODELS:
            # Add leading "\n\n" to prompt
            prompt_template_or_dict = "\n\n" + prompt_template_or_dict

    # Load the intermediate results
    generated_paragraphs = data_dict['generated_paragraph']

    # Use the Biography Generation eval pipeline
    if 'biography_generation_eval' == interpretation_prompt_type.split('/')[0]:
        entities = data_dict['entity']
        is_multiquery = False

        if interpretation_prompt_type == 'biography_generation_eval/fact_checker':
            factscorer = FactScorer()

            # Retrieves the top 5 articles for each atomic fact, from the entity's wiki page
            prompts, n_prompts_per_topic, data_to_return = factscorer.construct_prompts_with_retrieval(
                topics=entities,
                atomic_facts=generated_paragraphs,
                model_name=model_name,
                knowledge_source="enwiki-20230401",
            )
        else:
            prompts, data_to_return = prompt_processing_utils.construct_factscore_prompts(
                generated_paragraphs=generated_paragraphs,
                entities=entities,
                prompt_template_or_dict=prompt_template_or_dict,
                prompt_string_formatter_fn=INTERPRETATION_PROMPT_TO_METADATA[
                    interpretation_prompt_type]["prompt_preprocessor"],
            )
    else:
        questions = data_dict['question']
        ground_truth_top_answers = data_dict['ground_truth_top_answer']

        # TODO(@nband): generalize
        # We use a batched prompt to check semantic equivalence, since it's a simple operation
        # (In the future, we'll consider batching other prompts too)
        if interpretation_prompt_type in {
            'eval/check_semantic_equivalence_10shot_batch10',
            'eval/check_semantic_equivalence_10shot_batch10_claude_chat'
        }:
            extra_args = {
                'multiquery_chunk_size': 10,
            }
            prompt_constructor_fn = prompt_processing_utils.construct_multiquery_prompts
            is_multiquery = True
        else:
            extra_args = {}
            prompt_constructor_fn = prompt_processing_utils.construct_prompts
            is_multiquery = False

        prompts, data_to_return = prompt_constructor_fn(
            generated_paragraphs=generated_paragraphs,
            questions=questions,
            ground_truth_top_answers=ground_truth_top_answers,
            prompt_template_or_dict=prompt_template_or_dict,
            prompt_string_formatter_fn=INTERPRETATION_PROMPT_TO_METADATA[
                interpretation_prompt_type]["prompt_preprocessor"],
            **extra_args
        )

    logger.warning("Running completion loop with %d prompts", len(prompts))
    logger.warning("Max retry loops: %d", max_retry_loops)

    completions_text, interpretations = run_completion_retry_loop(
        prompts=prompts,
        decoding_args=decoding_args,
        model_name=model_name,
        per_device_batch_size=per_device_batch_size,
        postprocess_fn=INTERPRETATION_PROMPT_TO_METADATA[interpretation_prompt_type]["prompt_postprocessor"],
        max_retry_loops=max_retry_loops,
        mixed_precision=mixed_precision,
        is_multiquery=is_multiquery,
        **kwargs
    )

    if interpretation_prompt_type == 'biography_generation_eval/fact_checker':
        completions_per_entity = []
        interpretations_per_entity = []
        pointer = 0
        for n_prompts in n_prompts_per_topic:
            completions_per_entity.append(completions_text[pointer:pointer + n_prompts])
            interpretations_per_entity.append(interpretations[pointer:pointer + n_prompts])
            pointer += n_prompts

        completions_text = completions_per_entity
        interpretations = interpretations_per_entity

    if is_multiquery:
        completions_text_new = []
        interpretations_new = []
        for completions_text_elem, interpretations_nested_list in utils.zip_(completions_text, interpretations):
            for interpretation in interpretations_nested_list:
                completions_text_new.append(completions_text_elem)
                interpretations_new.append(interpretation)

        completions_text = completions_text_new
        interpretations = interpretations_new

    data_to_return["cot_reasoning_path"] = completions_text
    data_to_return["interpretation"] = interpretations

    return data_to_return
