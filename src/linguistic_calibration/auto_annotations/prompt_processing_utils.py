import string
import tqdm
from collections import defaultdict

from linguistic_calibration import utils
from linguistic_calibration.logging import get_logger
from linguistic_calibration.types import Union, List, Mapping, Callable, Optional

logger = get_logger(__name__)


# * Preprocessing (QA) *

def format_forecast_probs_prompt(
    prompt_template_or_dict: Union[str, Mapping[str, str]],
    question: str,
    generated_paragraph: str,
    ground_truth_top_answer: str,
    **kwargs
):
    return prompt_template_or_dict.format(
        question=question,
        generated_paragraph=generated_paragraph,
        ground_truth_top_answer=ground_truth_top_answer)


def format_extract_answers_prompt(
    prompt_template_or_dict: Union[str, Mapping[str, str]],
    question: str,
    generated_paragraph: str,
    **kwargs
):
    return prompt_template_or_dict.format(
        question=question,
        generated_paragraph=generated_paragraph)


def format_multiquery_prompt_semantic_equivalence(
    prompt_template: str,
    n_queries: int,
    questions: List[str],
    generated_paragraphs: List[str],
    ground_truth_top_answers: List[str],
    **kwargs,
):
    # Replace {n_queries} with the number of queries
    prompt_template = prompt_template.replace("{n_queries}", str(n_queries))

    examples_str = ""

    # Iterate through question, generated paragraph, and ground-truth top answer
    for index, (question, generated_paragraph, ground_truth_top_answer) in enumerate(
            zip(questions, generated_paragraphs, ground_truth_top_answers)
    ):
        example_index = index + 1
        examples_str += f'\t## Example {example_index}'
        examples_str += '\n'

        dict_str = f"""\t{{
        "Question": "{question}",
        "Ground-Truth Answer": "{ground_truth_top_answer}",
        "Provided Answer": "{generated_paragraph}"\n\t}}"""
        examples_str += dict_str
        examples_str += '\n\n'

    # Add all query examples
    prompt_template = prompt_template.replace("{examples}", examples_str)
    return prompt_template

# * Preprocessing (FactScore) *


def format_factscore_decomposition_prompt(
    prompt_template_or_dict: Union[str, Mapping[str, str]],
    entity: str,
    generated_paragraph: str,
    **kwargs
):
    return prompt_template_or_dict.format(
        entity=entity,
        generated_paragraph=generated_paragraph)


def format_claims_list_for_confidence_filtering(
    claims_list,
    claim_prefix: str = 'Statement'
):
    if isinstance(claims_list, str):
        claims_list = eval(claims_list)

    assert isinstance(claims_list, list)
    assert all(isinstance(claim, list) or isinstance(claim, str) for claim in claims_list)

    claims_list_strs = []
    counter = 1
    for claim in claims_list:
        if isinstance(claim, str):
            claims_list_strs.append(f"{claim_prefix} {counter}: {claim}")
            counter += 1
        elif isinstance(claim, list):
            for claim_elem in claim:
                claims_list_strs.append(f"{claim_prefix} {counter}: {claim_elem}")
                counter += 1
        else:
            raise ValueError(f"Invalid claim type {type(claim)}")

    return '\n'.join(claims_list_strs)


def format_factscore_confidence_filter_prompt(
    prompt_template_or_dict: Union[str, Mapping[str, str]],
    entity: str,
    generated_paragraph: Union[str, List],
    claim_prefix: str = 'Statement',
    **kwargs
):
    return prompt_template_or_dict.format(
        entity=entity,
        claims_list=format_claims_list_for_confidence_filtering(
            generated_paragraph,
            claim_prefix=claim_prefix)
    )


# * Postprocessing and parsing interpretations, QA *


class JSONExtractAnswersAnnotatorPostprocessor:
    def __init__(self):
        self.is_cot = True

    def __call__(self, string: str):
        try:
            # Remove newlines
            string = string.replace('\n', '')

            # Try loading the json
            json_dict = eval(string)

            # Try extracting the answer
            answer = json_dict["List of answers to the question in the context paragraph"]
        except Exception:
            return None

        return answer


class JSONForecastProbsAnnotatorPostprocessor:
    def __init__(self, extraction_key):
        self.is_cot = True
        self.extraction_key = extraction_key

    def __call__(self, string: str):
        # Try loading the json
        try:
            # Remove newlines
            string = string.replace('\n', '')
            json_dict = eval(string)
            # Try to extract the answer
            answer = json_dict[self.extraction_key]
            # Try to convert to float
            answer = float(answer)
        except Exception:
            return None

        return answer


class MultilineForecastProbsAnnotatorPostprocessor:
    """Outputs formatted as:
        <your reasoning here>[newline]Answer Probability, float in range [0.0, 1.0]: <probability>

        where [newline] may be any number of newlines.
    """
    def __init__(self):
        self.is_cot = True

    def __call__(self, string: str):
        # Try grabbing the probability forecasts
        try:
            prob_forecast = string.split('Answer Probability, float in range [0.0, 1.0]: ')[1]

            # Strip space
            prob_forecast = prob_forecast.strip()

            # Remove period from the prob forecast, if it exists
            if prob_forecast[-1] == '.':
                prob_forecast = prob_forecast[:-1]

            # Convert to float
            prob_forecast = float(prob_forecast)
        except Exception:
            return None

        return prob_forecast


class MultilineExtractAnswersAnnotatorPostprocessor:
    """Outputs formatted as:
    <your reasoning here>[newline]List of Answers to the Question in the Context Paragraph: <list of answers provided in
    the context paragraph, delimited by semicolons. If there are no answers, write "No Answers" without the quotes>
    where [newline] may be any number of newlines.
    """
    def __init__(self):
        self.is_cot = True

    def __call__(self, string: str):
        try:
            # Split by newlines
            lines = string.split('\n')

            # Remove empty lines
            lines = [line for line in lines if line != '']

            # Try grabbing the list of answers
            list_of_answers = lines[1].split('List of Answers to the Question in the Context Paragraph: ')[1]

            # Split by semicolons
            list_of_answers = list_of_answers.split(';')
        except:
            return None

        # Strip space from each answer
        try:
            list_of_answers = [answer.strip() for answer in list_of_answers]
        except:
            return list_of_answers

        try:
            # Remove period from the final answer, if it exists and is not the only character
            if list_of_answers[-1][-1] == '.' and len(list_of_answers[-1]) > 1:
                list_of_answers[-1] = list_of_answers[-1][:-1]
        except:
            pass

        # If the list of answers is ['No Answers'], replace with empty list
        try:
            if list_of_answers == ['No Answers']:
                list_of_answers = []
        except:
            pass

        return list_of_answers


class SemanticEquivalenceAnnotatorPostprocessor:
    def __init__(self):
        self.is_cot = False

    def __call__(self, string: str):
        # Try loading the json

        # Remove newlines
        try:
            string = string.replace('\n', '')
            json_dict = eval(string)

            # Try extracting the answers
            answers = list(json_dict.values())
        except Exception:
            return None

        return answers

# * Postprocessing and parsing interpretations, FactScore *


class JSONNonConfidenceClaimDecompositionAnnotatorPostprocessor:
    def __init__(self):
        self.is_cot = False

    def __call__(self, string: str):
        try:
            # Add the starting bracket
            string = '{' + string

            # Remove newlines
            string = string.replace('\n', '')

            # Try loading the json
            json_dict = eval(string)
        except Exception:
            return None

        return json_dict


class JSONConfidenceClaimDecompositionAnnotatorPostprocessor:
    def __init__(self):
        self.is_cot = False

    def __call__(self, string: str):
        try:
            # Add the starting bracket
            string = '[' + string

            # Remove newlines
            string = string.replace('\n', '')

            # Try loading the nested list
            nested_list = eval(string)
        except Exception:
            return None

        return nested_list


class MultilineConfidenceFilterAnnotatorPostprocessor:
    def __init__(self):
        self.is_cot = False

    def __call__(self, string: str, n_expected_outputs: Optional[int] = None):
        # Add the starting bracket (or equivalent for this output format)
        # Split by double newline
        # Remove empty strings
        try:
            string = 'Statement 1: ' + string
            string = string.split('\n\n')
            string = [s for s in string if s != '']
        except Exception as e:
            print(e)
            return None

        # Process each string
        def process_statement_paragraph(statement_paragraph, index):
            dict_to_return = {}

            # Split by newline
            statement_paragraph = statement_paragraph.split('\n')
            dict_to_return['paragraph'] = statement_paragraph

            try:
                # Split every line on first colon
                line_prefix_to_suffix = {
                    line.split(': ', 1)[0]: line.split(': ', 1)[1]
                    for line in statement_paragraph
                }

                # Convert line prefixes to established names
                prefix_to_name = {
                    f'Statement {index}': 'statement',
                    '[Step 1] Core Claim': 'core_claim',
                    '[Step 2] Statement and Core Claim are the Same': 'statement_and_core_claim_are_the_same',
                    'Classification': 'classification',
                    'Probability': 'probability',
                    'Statement': 'statement',
                }

                # Convert line prefixes to established names
                for prefix, name in prefix_to_name.items():
                    if prefix in line_prefix_to_suffix:
                        dict_to_return[name] = line_prefix_to_suffix[prefix]

                # Postprocess
                keys_to_convert_to_bool = {
                    'statement_and_core_claim_are_the_same',
                }
                for key in keys_to_convert_to_bool:
                    if key in dict_to_return:
                        dict_to_return[key] = eval(dict_to_return[key].strip())

                keys_to_strip = {
                    'statement',
                    'core_claim',
                    'classification',
                }
                for key in keys_to_strip:
                    if key in dict_to_return:
                        dict_to_return[key] = dict_to_return[key].strip()

                if 'probability' in dict_to_return:
                    dict_to_return['probability'] = float(
                        dict_to_return['probability'].strip())

                # Should always have a classification
                if 'classification' not in dict_to_return:
                    return None

                if dict_to_return['classification'] not in {'Direct', 'Numerical Uncertainty',
                                                            'Linguistic Uncertainty'}:
                    return None

                # If the classification is in Numerical Uncertainty or Linguistic Uncertainty, then
                # we should have a probability
                if dict_to_return['classification'] in {'Numerical Uncertainty', 'Linguistic Uncertainty'}:
                    if 'probability' not in dict_to_return:
                        return None

                    if dict_to_return['probability'] < 0.0 or dict_to_return['probability'] > 1.0:
                        return None

                return dict_to_return
            except Exception as e:
                print(e)
                return None

        processed_statements = []
        for i, statement_paragraph_ in enumerate(string):
            processed_statement_paragraph = process_statement_paragraph(
                statement_paragraph_, i + 1)
            if processed_statement_paragraph is None or isinstance(processed_statement_paragraph, float):
                return None

            processed_statements.append(processed_statement_paragraph)

        if n_expected_outputs is not None:
            if len(processed_statements) != n_expected_outputs:
                return None

        return processed_statements


class FactCheckAnnotatorPostprocessor:
    """Fact Check should be a string containing True or False.

    This postprocessing is taken from FactScore.
    """
    def __init__(self):
        self.is_cot = False

    def __call__(self, output: str):
        # From FactScore codebase
        # when logits are unavailable
        generated_answer = output.lower()
        if "true" in generated_answer or "false" in generated_answer:
            if "true" in generated_answer and "false" not in generated_answer:
                is_supported = True
            elif "false" in generated_answer and "true" not in generated_answer:
                is_supported = False
            else:
                is_supported = generated_answer.index("true") > generated_answer.index("false")
        else:
            is_supported = all(
                [keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split()
                 for keyword in ["not", "cannot", "unknown", "information"]])

        return {
            'Final Answer': {
                True: 'Supported',
                False: 'Not Supported'
            }[is_supported]
        }


# * Pre- and post-processing logic *


def construct_prompts(
    generated_paragraphs: List[str],
    questions: List[str],
    ground_truth_top_answers: List[str],
    prompt_template_or_dict: Union[str, dict],
    prompt_string_formatter_fn: Callable,
):
    prompts = []
    data_to_return = defaultdict(list)
    logger.info("Constructing single-query prompts for %d generated paragraphs", len(generated_paragraphs))

    for question, generated_paragraph, ground_truth_top_answer in tqdm.tqdm(
        utils.zip_(questions, generated_paragraphs, ground_truth_top_answers),
        total=len(questions),
    ):
        # Construct prompt
        format_fields = dict(
            prompt_template_or_dict=prompt_template_or_dict,
            question=question,
            generated_paragraph=generated_paragraph,
            ground_truth_top_answer=ground_truth_top_answer,
        )
        prompt_str = prompt_string_formatter_fn(**format_fields)
        data_to_return["question"].append(question)
        data_to_return["generated_paragraph"].append(generated_paragraph)
        data_to_return["ground_truth_top_answer"].append(ground_truth_top_answer)
        data_to_return["interpretation_prompt"].append(prompt_str)
        prompts.append(prompt_str)

    return prompts, data_to_return


def construct_factscore_prompts(
    generated_paragraphs: List[str],
    entities: List[str],
    prompt_template_or_dict: Union[str, dict],
    prompt_string_formatter_fn: Callable,
):
    prompts = []
    data_to_return = defaultdict(list)
    logger.info("Constructing single-query prompts for FactScore, %d generated paragraphs", len(generated_paragraphs))

    for i, (entity, generated_paragraph) in enumerate(
        tqdm.tqdm(utils.zip_(entities, generated_paragraphs), total=len(entities))
    ):
        # Construct prompt
        format_fields = dict(
            prompt_template_or_dict=prompt_template_or_dict,
            entity=entity,
            generated_paragraph=generated_paragraph,
        )
        prompt_str = prompt_string_formatter_fn(**format_fields)
        data_to_return["interpretation_prompt"].append(prompt_str)
        prompts.append(prompt_str)

    data_to_return["entity"] = entities
    data_to_return["generated_paragraph"] = generated_paragraphs
    return prompts, data_to_return


def construct_multiquery_prompts(
    generated_paragraphs: List[str],
    questions: List[str],
    ground_truth_top_answers: List[str],
    prompt_template_or_dict: Union[str, dict],
    multiquery_chunk_size: int,
    prompt_string_formatter_fn: Callable
):
    prompts = []
    data_to_return = defaultdict(list)

    logger.info("Constructing multiquery prompts for %d generated paragraphs", len(generated_paragraphs))
    assert len(generated_paragraphs) == len(questions) == len(ground_truth_top_answers)

    for left_index in tqdm.tqdm(
            range(0, len(generated_paragraphs), multiquery_chunk_size),
            total=len(generated_paragraphs) // multiquery_chunk_size,
    ):
        right_index = min(left_index + multiquery_chunk_size, len(generated_paragraphs))
        indices = list(range(left_index, right_index))
        n_queries = len(indices)
        questions_ = [questions[i] for i in indices]
        generated_paragraphs_ = [generated_paragraphs[i] for i in indices]
        ground_truth_top_answers_ = [ground_truth_top_answers[i] for i in indices]
        format_fields = dict(
            prompt_template=prompt_template_or_dict,
            n_queries=n_queries,
            questions=questions_,
            generated_paragraphs=generated_paragraphs_,
            ground_truth_top_answers=ground_truth_top_answers_,
        )
        prompt_str = prompt_string_formatter_fn(**format_fields)
        for question, generated_paragraph, ground_truth_top_answer in zip(
                questions_, generated_paragraphs_, ground_truth_top_answers_
        ):
            data_to_return["question"].append(question)
            data_to_return["generated_paragraph"].append(generated_paragraph)
            data_to_return["ground_truth_top_answer"].append(ground_truth_top_answer)
            data_to_return["interpretation_prompt"].append(prompt_str)

        prompts.append(prompt_str)

    return prompts, data_to_return
