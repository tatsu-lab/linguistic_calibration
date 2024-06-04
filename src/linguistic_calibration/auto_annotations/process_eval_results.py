import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import bootstrap
from scipy.stats._common import ConfidenceInterval
from scipy.stats._resampling import _bootstrap_resample, BootstrapResult

from linguistic_calibration import utils
from linguistic_calibration.logging import get_logger
from linguistic_calibration.types import Tuple, List, Dict, Union

logger = get_logger(__name__)

# QA Metrics


def get_metrics_for_non_confidence_baseline(
    results: Tuple[pd.DataFrame, pd.DataFrame],
    method_key=None,
    verbose=False
):
    logger.info(f"Getting metrics for non-confidence baseline: {method_key}.")
    extract_answers_df, semantic_equivalence_df = results
    assert len(extract_answers_df) == len(semantic_equivalence_df)
    
    correctness_arr = []
    generated_paragraph_arr = []
    question_arr = []
    ground_truth_top_answer_arr = []
    argmax_answer_arr = []
    nan_count_answer_extraction = 0
    nan_count_semantic_equivalence = 0
    for i in range(len(extract_answers_df)):
        extracted_answers_row, semantic_equivalence_row = extract_answers_df.iloc[i], semantic_equivalence_df.iloc[i]
        extracted_answers_str = extracted_answers_row['interpretation__answer_extraction']
        generated_paragraph_arr.append(extracted_answers_row['generated_paragraph'])
        question_arr.append(extracted_answers_row['question'])
        ground_truth_top_answer_arr.append(extracted_answers_row['ground_truth_top_answer'])
        argmax_answer_arr.append(extracted_answers_row['interpretation__answer_extraction'])
        if pd.isna(extracted_answers_str):
            logger.warning(f"extracted_answers_str is NaN for row {i}")
            correctness_arr.append(False)
            nan_count_answer_extraction += 1
            continue
        
        extracted_answers = eval(extracted_answers_str)
        assert isinstance(extracted_answers, list)
        
        # Consider an empty answer extraction to be incorrect
        if len(extracted_answers) == 0:
            correctness_arr.append(False)
            
        # Otherwise, check if the semantic equivalence is correct
        else:
            semantic_equivalence = semantic_equivalence_row['interpretation__semantic_equivalence']
            if pd.isna(semantic_equivalence):
                logger.warning(f"semantic_equivalence is NaN for row {i}")
                correctness_arr.append(False)
                nan_count_semantic_equivalence += 1
                continue
            
            if semantic_equivalence == 'Yes':
                correctness_arr.append(True)
            elif semantic_equivalence == 'No':
                correctness_arr.append(False)
            else:
                # Classify as incorrect if semantic equivalence is not 'Yes' or 'No'
                logger.warning(f"semantic_equivalence is not 'Yes' or 'No' for row {i}: {semantic_equivalence}")
                correctness_arr.append(False)
                nan_count_semantic_equivalence += 1

    correctness_arr = np.array(correctness_arr)
    assert correctness_arr.shape == (len(extract_answers_df),)
    accuracy = correctness_arr.mean()
    
    # Get upper and lower bootstrap confidence intervals
    ci = bootstrap(
        data=(correctness_arr,),
        statistic=np.mean,
        n_resamples=10000,
        vectorized=True,
        method='percentile',
        confidence_level=0.95
    )
    metrics_to_return = {
        'accuracy': accuracy,
        'ci_lower': ci.confidence_interval.low,
        'ci_upper': ci.confidence_interval.high,
        'ece': 1 - accuracy,  # ECE when confidence = 1 for all predictions
        'ece_upper': 1 - ci.confidence_interval.low,
        'ece_lower': 1 - ci.confidence_interval.high,
        'correctness_arr': correctness_arr,
        'generated_paragraph_arr': generated_paragraph_arr,
        'question_arr': question_arr,
        'ground_truth_top_answer_arr': ground_truth_top_answer_arr,
        'argmax_answer_arr': argmax_answer_arr
    }
    if verbose:
        logger.warning(f'Total NaNs in answer extraction: {nan_count_answer_extraction}')
        logger.warning(f'Total NaNs or incorrect phrases in semantic equivalence: {nan_count_semantic_equivalence}')
    return metrics_to_return


def compute_ece(
    confidence_arr,
    correctness_arr,
    ece_type='equal_bin_count',
    n_bins=20,
    return_ece=False,
    verbose=False
):
    # Compute the expected accuracy and confidence in each bin.
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    # Remove any nans; by construction only the confidences can have nans
    confidence_nans = np.isnan(confidence_arr)
    correctness_arr = correctness_arr[~confidence_nans]
    confidence_arr = confidence_arr[~confidence_nans]
    n_nans = confidence_nans.sum()
    if n_nans > 0 and verbose:
        logger.warning(f"Removed {n_nans} nans from confidence_arr.")

    # Bin with equal size in each bin
    n_count_per_bin = len(confidence_arr) // n_bins

    if ece_type == 'equal_bin_length':
        # Bin the confidences into uniformly spaced bins.
        bin_edges = np.linspace(0, 1, n_bins + 1)

        # Get the bin indices for each confidence
        bin_indices = np.digitize(confidence_arr, bin_edges) - 1

        # Edge case: assign all examples with confidence 1 to the last bin
        bin_indices[confidence_arr == 1] = n_bins - 1

        for bin_idx in range(n_bins):
            bin_mask = bin_indices == bin_idx
            if bin_mask.sum() == 0 and verbose:
                logger.warning(f'No examples in bin {bin_idx}. Skipping.')
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
                continue

            bin_confs = confidence_arr[bin_mask]
            bin_confidences.append(bin_confs.mean())
            bin_correctness = correctness_arr[bin_mask]
            bin_accuracies.append(bin_correctness.mean())
            bin_counts.append(bin_mask.sum())

            if verbose:
                logger.info(
                    f'Bin {bin_idx}: {len(bin_confs)} examples, '
                    f'{bin_correctness.mean()} accuracy, {bin_confs.mean()} confidence')
    elif ece_type == 'equal_bin_count':
        confidence_and_correctness = list(utils.zip_(confidence_arr, correctness_arr))
        sorted_confidence_and_correctness = sorted(confidence_and_correctness,
                                                   key=lambda x: x[0])
        for i in range(n_bins):
            bin_confidence_and_correctness = (
                sorted_confidence_and_correctness[i * n_count_per_bin: (i + 1) * n_count_per_bin])
            bin_confidence, bin_correctness = list(
                utils.zip_(*bin_confidence_and_correctness))
            bin_confidence = np.array(bin_confidence)
            bin_correctness = np.array(bin_correctness)

            bin_accuracies.append(bin_correctness.mean())
            bin_confidences.append(bin_confidence.mean())
            bin_counts.append(len(bin_confidence))

            if verbose:
                logger.info(
                    f'Bin {i}: {len(bin_confidence)} examples, '
                    f'{bin_correctness.mean()} accuracy, {bin_confidence.mean()} confidence')
    else:
        raise NotImplementedError

    # Compute ECE
    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)
    ece = np.abs(bin_accuracies - bin_confidences).dot(bin_counts) / bin_counts.sum()

    if return_ece:
        return ece

    return {
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
        'ece': ece
    }


def ece_bootstrap(
    data,
    confidence_level: float = 0.95,
    method: str = 'percentile',
    alternative: str = 'two-sided',
    n_resamples: int = 10000,
    batch: int = None,
    random_state=None
):
    """Calculate bootstrap confidence interval of the ECE.

    Based on scipy.stats.bootstrap.
    """
    theta_hat_b = []
    batch_nominal = batch or n_resamples or 1
    for k in range(0, n_resamples, batch_nominal):
        batch_actual = min(batch_nominal, n_resamples - k)

        # Generate resamples
        resampled_data = []
        for sample in data:
            resample = _bootstrap_resample(sample, n_resamples=batch_actual, random_state=random_state)
            resampled_data.append(resample)

        confidence_scores = resampled_data[0][0]
        correctness_scores = resampled_data[0][1]
        for m in range(batch_actual):
            ece = compute_ece(confidence_scores[m], correctness_scores[m], return_ece=True)
            theta_hat_b.append(ece)

    theta_hat_b = np.array(theta_hat_b)

    # Calculate percentile interval
    alpha = ((1 - confidence_level) / 2 if alternative == 'two-sided'
             else (1 - confidence_level))
    if method == 'bca':
        raise NotImplementedError
    else:
        interval = alpha, 1 - alpha

        def percentile_fun(a, q):
            return np.percentile(a=a, q=q, axis=-1)

    # Calculate confidence interval of statistic
    ci_l = percentile_fun(theta_hat_b, interval[0] * 100)
    ci_u = percentile_fun(theta_hat_b, interval[1] * 100)
    if method == 'basic':  # see [3]
        # theta_hat = statistic(*data, axis=-1)
        theta_hat = compute_ece(data[:, 0], data[:, 1], return_ece=True)
        ci_l, ci_u = 2 * theta_hat - ci_u, 2 * theta_hat - ci_l

    if alternative == 'less':
        ci_l = np.full_like(ci_l, -np.inf)
    elif alternative == 'greater':
        ci_u = np.full_like(ci_u, np.inf)

    return BootstrapResult(confidence_interval=ConfidenceInterval(ci_l, ci_u),
                           bootstrap_distribution=theta_hat_b,
                           standard_error=np.std(theta_hat_b, ddof=1, axis=-1))

    return resampled_data


def get_argmax_answers_for_forecast_probs_df(df):
    question_id_to_extracted_answer_and_conf_score = defaultdict(list)
    processed_question_ids = []
    generated_paragraphs = []
    ground_truth_top_answers = []
    questions = []
    for i, row in df.iterrows():
        question_id = row['question_id']

        # We are at an extracted answer row (there may be many for each question_id)
        if question_id in set(processed_question_ids):
            question_id_to_extracted_answer_and_conf_score[question_id].append(
                (row['ground_truth_top_answer'], row['interpretation__forecast_probs']))
        # We are at the ground-truth answer row (these come before the extracted answer rows)
        else:
            processed_question_ids.append(question_id)
            questions.append(row['question'])
            generated_paragraphs.append(row['generated_paragraph'])
            ground_truth_top_answers.append(row['ground_truth_top_answer'])

    empty_list_count = 0

    # Get argmax answers
    argmax_answers = []
    argmax_probs = []
    for question_id in processed_question_ids:
        extracted_answer_and_conf_score_list = (
            question_id_to_extracted_answer_and_conf_score[question_id])

        if len(extracted_answer_and_conf_score_list) == 0:
            argmax_answers.append(None)
            argmax_probs.append(None)
            empty_list_count += 1
            continue

        # If there are multiple extracted answers with the same confidence score,
        #   just take the first one
        index_of_top_answer = 0
        for j in range(1, len(extracted_answer_and_conf_score_list)):
            if extracted_answer_and_conf_score_list[j][1] > \
                    extracted_answer_and_conf_score_list[index_of_top_answer][1]:
                index_of_top_answer = j

        argmax_answer, argmax_prob = extracted_answer_and_conf_score_list[index_of_top_answer]

        assert argmax_answer is not None, (
            f'Argmax answer is None for question_id {question_id}')
        assert argmax_prob is not None, (
            f'Argmax prob is None for question_id {question_id}')

        argmax_answers.append(argmax_answer)
        argmax_probs.append(argmax_prob)

    logger.info(f'Found {empty_list_count} empty lists in the ForecastProbs results. '
                f'Empty lists are expected if the model did not provide a direct response to the question '
                f'in its long-form paragraph generation.')

    return argmax_answers, argmax_probs, generated_paragraphs, ground_truth_top_answers, questions


def get_metrics_for_confidence_baseline(
        results: Tuple[pd.DataFrame, pd.DataFrame],
        method_key=None
):
    logger.info(method_key)
    forecast_probs_df, semantic_equivalence_df = results
    question_ids = semantic_equivalence_df['question_id']

    # Get argmax answers
    argmax_answers, argmax_probs, generated_paragraphs, ground_truth_top_answers, questions = (
        get_argmax_answers_for_forecast_probs_df(forecast_probs_df))

    assert len(argmax_answers) == len(question_ids), (
        f'Length of argmax answers {len(argmax_answers)} does not match length of question_ids {len(question_ids)}')

    # Get semantic equivalence values
    binary_correctness = []
    for argmax_answer, semantic_equivalence in utils.zip_(
            argmax_answers, semantic_equivalence_df['interpretation__semantic_equivalence']):
        # If not None, then the semantic equivalence value is meaningful
        if argmax_answer is not None:
            if semantic_equivalence == 'Yes':
                binary_correctness.append(True)
            elif semantic_equivalence == 'No':
                binary_correctness.append(False)
            else:
                # Classify as incorrect if semantic equivalence is not 'Yes' or 'No'
                binary_correctness.append(False)
                logger.warning(f"semantic_equivalence is not 'Yes' or 'No': {semantic_equivalence}")
                logger.warning(f"argmax_answer: {argmax_answer}")
        else:
            # In this case, we used `None` as a filler for the empty list in the semantic equivalence prompt.
            # These examples have empty answer extraction lists, meaning that the long-form generation was not
            # responsive to the question.
            binary_correctness.append(None)

    # We handle empty lists as incorrect with confidence 1, which is the worst case penalty for both accuracy and ECE
    binary_correctness_for_accuracy = [False if elem is None else elem for elem in binary_correctness]
    accuracy = sum(binary_correctness_for_accuracy) / len(binary_correctness_for_accuracy)

    confidence_and_correctness = list(utils.zip_(argmax_probs, binary_correctness))

    # Confidence 1, correctness False
    confidence_and_correctness = [(1, False) if elem[0] is None or elem[1] is None else elem for elem in
                                  confidence_and_correctness]
    confidence_arr, correctness_arr = list(utils.zip_(*confidence_and_correctness))

    ci = bootstrap(
        data=(binary_correctness_for_accuracy,),
        statistic=np.mean,
        n_resamples=10000,
        vectorized=True,
        method='percentile',
        confidence_level=0.95
    )

    ece_dict = compute_ece(np.array(confidence_arr), np.array(correctness_arr))
    ece_ci = ece_bootstrap(
        (np.array([
            np.array(confidence_arr),
            np.array(correctness_arr)
        ]),))

    return {
        'accuracy': accuracy,
        'ci_lower': ci.confidence_interval.low,
        'ci_upper': ci.confidence_interval.high,
        'ece_lower': ece_ci.confidence_interval.low,
        'ece_upper': ece_ci.confidence_interval.high,
        'argmax_answers': argmax_answers,
        'argmax_probs': argmax_probs,
        'binary_correctness_arr': correctness_arr,
        'confidence_arr': confidence_arr,
        'generated_paragraph_arr': generated_paragraphs,
        'ground_truth_top_answer_arr': ground_truth_top_answers,
        'question_arr': questions,
        **ece_dict
    }

# FactScore Metrics

# Methods to check consistency of claims between the steps of our FactScore-style evaluation pipeline.


def eval_objects_and_get_nan_count(list_of_object_strs: List[str]):
    """Evaluates a list of strings into objects and returns the number of NaNs."""
    objects_to_return = []
    nan_count = 0
    for obj_str in list_of_object_strs:
        try:
            objects_to_return.append(eval(obj_str))
        except:
            objects_to_return.append(None)
            nan_count += 1

    return objects_to_return, nan_count


def check_claims_are_equal_nonconfidence(claim_decomp_dict, fact_check_list):
    if claim_decomp_dict is None or fact_check_list is None:
        return False

    assert isinstance(claim_decomp_dict, dict), f"claim_decomp_dict is not a dict: {claim_decomp_dict}"
    assert isinstance(fact_check_list, list), f"fact_check_dict is not a list: {fact_check_list}"

    # Filter to Objective claims
    claim_decomp_objective_claims = [key for key, value in claim_decomp_dict.items() if value == 'Objective']
    fact_check_claims = [claim_dict['claim'] for claim_dict in fact_check_list]

    assert len(claim_decomp_objective_claims) == len(
        fact_check_claims), (f"claim_decomp_objective_claims and fact_check_claims have different lengths: "
                             f"{len(claim_decomp_objective_claims)} vs {len(fact_check_claims)}")
    return np.array_equal(claim_decomp_objective_claims, fact_check_claims)


def get_proportion_of_matching_objects(iterable1, iterable2, match_fn):
    assert len(iterable1) == len(
        iterable2), f"iterable1 and iterable2 have different lengths: {len(iterable1)} vs {len(iterable2)}"

    n_matching = 0
    for obj1, obj2 in utils.zip_(iterable1, iterable2):
        if match_fn(obj1, obj2):
            n_matching += 1

    return n_matching / len(iterable1)


def flatten_nested_list(nested_list):
    """Flattens a 2D nested list."""
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(item)
        else:
            flattened_list.append(item)
    return flattened_list


def check_claims_are_equal_between_decomp_and_filter(claim_decomp_nested_list, lu_filter_list_of_dicts):
    """Checks if the claims match between the claim decomposition and linguistic uncertainty (LU) filter."""

    if claim_decomp_nested_list is None or lu_filter_list_of_dicts is None:
        return False

    assert isinstance(claim_decomp_nested_list,
                      list), f"claim_decomp_nested_list is not a list: {claim_decomp_nested_list}"
    assert isinstance(lu_filter_list_of_dicts,
                      list), f"lu_filter_list_of_dicts is not a list: {lu_filter_list_of_dicts}"

    claim_decomp_claims = flatten_nested_list(claim_decomp_nested_list)

    # The claim decomposition for linguistic uncertainty classifies into {Subjective, Objective, Full Uncertainty}
    # We take only the Objective claims
    claim_decomp_claims = [elem[0] for elem in claim_decomp_claims if elem[1] == 'Objective']
    lu_claims = [claim_dict['statement'] for claim_dict in lu_filter_list_of_dicts]
    assert len(claim_decomp_claims) == len(
        lu_claims), f"claim_decomp_claims and lu_claims have different lengths: {len(claim_decomp_claims)} vs {len(lu_claims)}"
    return np.array_equal(claim_decomp_claims, lu_claims)


def check_claims_are_equal_between_filter_and_fact_check(lu_filter_list_of_dicts, fact_check_list_of_dicts):
    """Checks if the claims match between the LU filter and fact check."""

    if lu_filter_list_of_dicts is None or fact_check_list_of_dicts is None:
        return False

    assert isinstance(lu_filter_list_of_dicts,
                      list), f"lu_filter_list_of_dicts is not a list: {lu_filter_list_of_dicts}"
    assert isinstance(fact_check_list_of_dicts,
                      list), f"fact_check_list_of_dicts is not a list: {fact_check_list_of_dicts}"

    lu_filter_core_claims = [claim_dict['core_claim'] for claim_dict in lu_filter_list_of_dicts]
    fact_check_claims = [claim_dict['claim'] for claim_dict in fact_check_list_of_dicts]

    assert len(lu_filter_core_claims) == len(
        fact_check_claims), (f"lu_filter_core_claims and fact_check_claims have different lengths: "
                             f"{len(lu_filter_core_claims)} vs {len(fact_check_claims)}")
    return np.array_equal(lu_filter_core_claims, fact_check_claims)


# Methods for computing the actual FactScore-style metrics (factuality, ECE, etc.)

def get_correctness_of_eval_dict(eval_dict):
    final_answer_key = 'Final Answer' if 'Final Answer' in eval_dict else 'final_answer'

    if eval_dict[final_answer_key] == 'Supported':
        return True
    elif eval_dict[final_answer_key] == 'Not Supported':
        return False
    else:
        print(f'Final Answer is not Supported or Not Supported: {eval_dict[final_answer_key]}')
        print('Assuming Not Supported.')
        return False


def get_confidence_from_lu_filter_dict(lu_filter_dict):
    if lu_filter_dict['classification'] == 'Direct':
        return 1.0
    elif lu_filter_dict['classification'] == 'Numerical Uncertainty':
        return lu_filter_dict['probability']
    elif lu_filter_dict['classification'] == 'Linguistic Uncertainty':
        return lu_filter_dict['probability']
    else:
        raise ValueError(f"Classification is not Direct, Numerical Uncertainty, or Linguistic Uncertainty: "
                         f"{lu_filter_dict['classification']}")


def get_factuality_metrics_for_example(fact_check_list):
    if len(fact_check_list) == 0:
        # Can happen if the model abstains -- outputs no atomic claims or only subjective claims
        return 0, 0, 0

    supported_count = 0
    not_supported_count = 0

    for eval_dict in fact_check_list:
        if get_correctness_of_eval_dict(eval_dict):
            supported_count += 1
        else:
            not_supported_count += 1

    proportion_supported = supported_count / (supported_count + not_supported_count)

    return supported_count, not_supported_count, proportion_supported


def get_lu_factuality_metrics_for_example(
        nested_lu_filter_list: List[Union[List[Dict], Dict]],
        nested_fact_check_list: List[Union[List[Dict], Dict]]
):
    assert len(nested_lu_filter_list) == len(
        nested_fact_check_list), (f"nested_lu_filter_list and nested_fact_check_list have different lengths: "
                                  f"{len(nested_lu_filter_list)} vs {len(nested_fact_check_list)}")
    assert len(flatten_nested_list(nested_lu_filter_list)) == len(flatten_nested_list(
        nested_fact_check_list)), (f"nested_lu_filter_list and nested_fact_check_list have different lengths: "
                                   f"{len(flatten_nested_list(nested_lu_filter_list))} vs "
                                   f"{len(flatten_nested_list(nested_fact_check_list))}")

    supported_count = 0
    not_supported_count = 0
    correctness_arr = []
    confidence_arr = []

    for lu_filter_elem, fact_check_elem in zip(nested_lu_filter_list, nested_fact_check_list):
        if isinstance(lu_filter_elem, list):
            # In this case, the model has provided multiple mutually exclusive possibilties for a claim.
            # For example,
            #   "there is a 75% chance she was educated at Princeton and a 25% chance she was educated at Yale."
            # would result in a list with two elements, each with a different probability (75% and 25%).
            assert isinstance(fact_check_elem, list), (
                f"lu_filter_elem is a list but fact_check_elem is not: {lu_filter_elem} vs {fact_check_elem}")
            assert len(lu_filter_elem) == len(fact_check_elem), (
                f"lu_filter_elem and fact_check_elem have different lengths: "
                f"{len(lu_filter_elem)} vs {len(fact_check_elem)}")

            # Get the index of the argmax of the confidence
            max_confidence_idx = np.argmax(
                [get_confidence_from_lu_filter_dict(lu_filter_dict) for lu_filter_dict in lu_filter_elem])

            # Evaluate the correctness of the argmax
            if get_correctness_of_eval_dict(fact_check_elem[max_confidence_idx]):
                supported_count += 1
            else:
                not_supported_count += 1

            correctness_arr.append(get_correctness_of_eval_dict(fact_check_elem[max_confidence_idx]))
            confidence_arr.append(get_confidence_from_lu_filter_dict(lu_filter_elem[max_confidence_idx]))
        elif isinstance(lu_filter_elem, dict):
            assert isinstance(fact_check_elem, dict), (
                f"lu_filter_elem is not a list but fact_check_elem is: {lu_filter_elem} vs {fact_check_elem}")

            if get_correctness_of_eval_dict(fact_check_elem):
                supported_count += 1
            else:
                not_supported_count += 1

            correctness_arr.append(get_correctness_of_eval_dict(fact_check_elem))
            confidence_arr.append(get_confidence_from_lu_filter_dict(lu_filter_elem))
        else:
            raise ValueError(f"lu_filter_elem is not a list or dict: {lu_filter_elem}")

    proportion_supported = supported_count / (supported_count + not_supported_count)
    return supported_count, not_supported_count, proportion_supported, correctness_arr, confidence_arr


def perform_mutually_exclusive_grouping(nested_arr, flat_arr):
    """Nest a flat_array like a nested_array."""
    if nested_arr is None or flat_arr is None:
        return None

    assert isinstance(nested_arr, list), f"nested_arr is not a list: {nested_arr}"
    assert isinstance(flat_arr, list), f"flat_arr is not a list: {flat_arr}"

    newly_nested_array = []
    pointer = 0
    for element in nested_arr:
        if isinstance(element, list):
            newly_nested_array.append(flat_arr[pointer:pointer + len(element)])
            pointer += len(element)
        else:
            newly_nested_array.append(flat_arr[pointer])
            pointer += 1

    return newly_nested_array


def filter_nested_array_with_condition(nested_arr, condition):
    filtered_nested_arr = []
    for element in nested_arr:
        if isinstance(element, list):
            arr_to_append = []
            for nested_element in element:
                if condition(nested_element):
                    arr_to_append.append(nested_element)

            # Don't add empty lists
            if len(arr_to_append) > 0:
                filtered_nested_arr.append(arr_to_append)
        else:
            if condition(element):
                filtered_nested_arr.append(element)

    return filtered_nested_arr


def is_claim_dict_factchecked(claim_dict):
    return claim_dict['classification'] in {'Direct', 'Numerical Uncertainty', 'Linguistic Uncertainty'}


def get_metrics_for_factscore_nonconfidence_results(
    results_df: pd.DataFrame,
    method_key=None,
    claim_decomposition_column_name: str = 'interpretation__claim_decomposition',
    fact_check_column_name: str = 'interpretation__fact_checker',
    verbose=False
):
    if method_key is not None and verbose:
        print(f"Method: {method_key}")

    # Check count of NaNs in claim decomposition
    claim_decomp_dicts, claim_decomp_nan_count = eval_objects_and_get_nan_count(
        results_df[claim_decomposition_column_name].tolist())

    if verbose:
        print(f"Claim decomposition NaN count: {claim_decomp_nan_count}")

    # Check count of NaNs in fact check
    fact_check_lists, fact_check_nan_count = eval_objects_and_get_nan_count(
        results_df[fact_check_column_name].tolist())

    if verbose:
        print(f"Fact check NaN count: {fact_check_nan_count}")

    # For every non-NaN fact check dict, we attempt to compute the factuality score
    supported_counts = []
    not_supported_counts = []
    proportion_supported_arr = []
    for claim_decomp_dict, fact_check_list in utils.zip_(claim_decomp_dicts, fact_check_lists):
        if fact_check_list is not None:
            supported_count, not_supported_count, proportion_supported = get_factuality_metrics_for_example(
                fact_check_list)
            supported_counts.append(supported_count)
            not_supported_counts.append(not_supported_count)
            proportion_supported_arr.append(proportion_supported)

    # Pool all claims together and compute ECE
    total_supported_count = sum(supported_counts)
    total_not_supported_count = sum(not_supported_counts)
    pooled_proportion_supported = total_supported_count / (total_supported_count + total_not_supported_count)
    pooled_ece = np.abs(1 - pooled_proportion_supported)

    is_supported_arr = np.array([True] * total_supported_count + [False] * total_not_supported_count)
    ci = bootstrap(
        data=(is_supported_arr,),
        statistic=np.mean,
        n_resamples=10000,
        vectorized=True,
        method='percentile',
        confidence_level=0.95
    )

    return {
        'claim_decomp_nan_count': claim_decomp_nan_count,
        'fact_check_nan_count': fact_check_nan_count,
        'supported_counts': supported_counts,
        'not_supported_counts': not_supported_counts,
        'proportion_supported_arr': proportion_supported_arr,
        'avg_supported_count': np.mean(supported_counts),
        'avg_not_supported_count': np.mean(not_supported_counts),
        'avg_proportion_supported': np.mean(proportion_supported_arr),
        'pooled_proportion_supported': pooled_proportion_supported,
        'pooled_proportion_supported_ci_lower': ci.confidence_interval.low,
        'pooled_proportion_supported_ci_upper': ci.confidence_interval.high,
        'pooled_ece': pooled_ece,
        'pooled_ece_ci_lower': 1 - ci.confidence_interval.high,
        'pooled_ece_ci_upper': 1 - ci.confidence_interval.low
    }


def get_metrics_for_factscore_confidence_results(
    results_df: pd.DataFrame,
    method_key=None,
    ece__n_bins_pooled=40,
    claim_decomposition_column_name: str = 'interpretation__claim_decomposition',
    lu_filter_column_name: str = 'interpretation__claim_uncertainty_filter',
    fact_check_column_name: str = 'interpretation__uncertainty_fact_checker',
    verbose=False
):
    if method_key is not None and verbose:
        print(f"Method: {method_key}")

    # Check count of NaNs in claim decomposition
    claim_decomp_dicts, claim_decomp_nan_count = eval_objects_and_get_nan_count(
        results_df[claim_decomposition_column_name].tolist())
    if verbose:
        print(f"Claim decomposition NaN count: {claim_decomp_nan_count}")

    # Check count of NaNs in LU filter
    lu_filter_lists, lu_filter_nan_count = eval_objects_and_get_nan_count(
        results_df[lu_filter_column_name].tolist())

    if verbose:
        print(f"LU filter NaN count: {lu_filter_nan_count}")

    # Check if the claims output by step 1 exactly match the claims output by step 2
    prob_identical_claims_step_1_2 = get_proportion_of_matching_objects(
        claim_decomp_dicts,
        lu_filter_lists,
        check_claims_are_equal_between_decomp_and_filter)

    if verbose:
        print(f"Proportion of examples for which the claims output from step 1 and used in step 2 are identical: "
              f"{prob_identical_claims_step_1_2}")

    # Check count of NaNs in fact check
    fact_check_lists, fact_check_nan_count = eval_objects_and_get_nan_count(
        results_df[fact_check_column_name].tolist())

    if verbose:
        print(f"Fact check NaN count: {fact_check_nan_count}")

    # We use the FactScore prompt for fact checking, so by construction, we will have a fact check for every (objective)
    #  claim in the LU filter.
    if claim_decomp_nan_count + lu_filter_nan_count + fact_check_nan_count > 0:
        raise ValueError("There are NaNs in the claim decomposition, LU filter, or fact check.")

    # Compute factuality and ECE
    supported_counts = []
    not_supported_counts = []
    proportion_supported_arr = []
    list_of_confidence_lists = []
    list_of_correctness_lists = []
    invalid_indices = []

    for i, (claim_decomp_nested_list, lu_filter_list_of_dicts, fact_check_list_of_dicts) in enumerate(
            utils.zip_(claim_decomp_dicts, lu_filter_lists, fact_check_lists)):
        # Filter the nested claim decomposition to only include objective claims
        filtered_claim_decomp_nested_list = filter_nested_array_with_condition(
            nested_arr=claim_decomp_nested_list,
            condition=lambda x: x[1] == 'Objective')

        # Unroll claim decomposition nested list
        filtered_claim_decomp_flattened_list = flatten_nested_list(filtered_claim_decomp_nested_list)

        # Assert that the number of objective claims in the claim decomposition and LU Filter list of dicts are the same
        if len(filtered_claim_decomp_flattened_list) != len(lu_filter_list_of_dicts):
            print(f"Number of claims in filtered_claim_decomp_flattened_list "
                  f"({len(filtered_claim_decomp_flattened_list)}) and lu_filter_list_of_dicts "
                  f"({len(lu_filter_list_of_dicts)}) are not the same.")
            print(filtered_claim_decomp_flattened_list)
            raise ValueError

        # Use the claim decomposition to determine which LU Filter claims should be grouped as mutually exclusive
        nested_lu_filter_list_of_dicts = perform_mutually_exclusive_grouping(
            nested_arr=filtered_claim_decomp_nested_list,
            flat_arr=lu_filter_list_of_dicts)

        # Assert that the number of claims in the nested LU filter list and the fact check list of dicts are the same
        if len(flatten_nested_list(nested_lu_filter_list_of_dicts)) != len(fact_check_list_of_dicts):
            print(f"Number of claims in lu_filter_list_of_dicts "
                  f"({len(flatten_nested_list(nested_lu_filter_list_of_dicts))}) and fact_check_list_of_dicts "
                  f"({len(fact_check_list_of_dicts)}) are not the same.")
            raise ValueError

        # Now nest the fact check dict according to the nested LU filter list
        #   (which is nested according to the claim decomposition)
        fact_check_nested_list_of_dicts = perform_mutually_exclusive_grouping(
            nested_arr=nested_lu_filter_list_of_dicts,
            flat_arr=fact_check_list_of_dicts)

        # Compute factuality metrics
        supported_count, not_supported_count, proportion_supported, correctness_list, confidence_list = (
            get_lu_factuality_metrics_for_example(nested_lu_filter_list_of_dicts, fact_check_nested_list_of_dicts))
        supported_counts.append(supported_count)
        not_supported_counts.append(not_supported_count)
        proportion_supported_arr.append(proportion_supported)
        list_of_confidence_lists.append(confidence_list)
        list_of_correctness_lists.append(correctness_list)

    # Pool all claims together
    total_supported_count = sum(supported_counts)
    total_not_supported_count = sum(not_supported_counts)
    pooled_proportion_supported = total_supported_count / (total_supported_count + total_not_supported_count)

    # Compute pooled ECE
    flattened_correctness_arr = np.array(
        [correctness for correctness_list in list_of_correctness_lists for correctness in correctness_list])
    flattened_confidence_arr = np.array(
        [confidence for confidence_list in list_of_confidence_lists for confidence in confidence_list])
    pooled_ece_results = compute_ece(
        flattened_confidence_arr, flattened_correctness_arr, ece_type='equal_bin_count', n_bins=ece__n_bins_pooled)
    is_supported_arr = np.array([True] * total_supported_count + [False] * total_not_supported_count)
    ci = bootstrap(
        data=(is_supported_arr,),
        statistic=np.mean,
        n_resamples=10000,
        vectorized=True,
        method='percentile',
        confidence_level=0.95
    )
    ece_ci = ece_bootstrap(
        (np.array([
            np.array(flattened_confidence_arr),
            np.array(flattened_correctness_arr)
        ]),))
    pooled_ece_upper = ece_ci.confidence_interval.high
    pooled_ece_lower = ece_ci.confidence_interval.low

    return {
        'invalid_indices': invalid_indices,
        'claim_decomp_nan_count': claim_decomp_nan_count,
        'lu_filter_nan_count': lu_filter_nan_count,
        'fact_check_nan_count': fact_check_nan_count,
        'prob_identical_claims_step_1_2': prob_identical_claims_step_1_2,
        'supported_counts': supported_counts,
        'not_supported_counts': not_supported_counts,
        'proportion_supported_arr': proportion_supported_arr,
        'avg_supported_count': np.mean(supported_counts),
        'avg_not_supported_count': np.mean(not_supported_counts),
        'avg_proportion_supported': np.mean(proportion_supported_arr),
        'pooled_proportion_supported': pooled_proportion_supported,
        'pooled_proportion_supported_ci_lower': ci.confidence_interval.low,
        'pooled_proportion_supported_ci_upper': ci.confidence_interval.high,
        'pooled_ece': pooled_ece_results['ece'],
        'pooled_ece_ci_lower': pooled_ece_lower,
        'pooled_ece_ci_upper': pooled_ece_upper,
        'pooled_bin_accuracies': pooled_ece_results['bin_accuracies'],
        'pooled_bin_confidences': pooled_ece_results['bin_confidences'],
        'pooled_bin_counts': pooled_ece_results['bin_counts'],
        'pooled_correctness_arr': flattened_correctness_arr,
        'pooled_confidence_arr': flattened_confidence_arr,
    }
