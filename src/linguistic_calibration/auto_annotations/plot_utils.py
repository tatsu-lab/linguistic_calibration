import os
from collections import defaultdict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_cmap_for_base_color(base_color, N=5):
    target_color_hex = '#FFFFFF'  # White

    # Create a colormap
    colors = [base_color, target_color_hex]
    cmap = mcolors.LinearSegmentedColormap.from_list("my_gradient", colors, N=N)
    return cmap


BASE_COLOR_TO_CMAP = {
    key: get_cmap_for_base_color(key)
    for key in ["#035e8f", "#045941", "#cb3b3b", "#039e72", "#7f6000", "#710e7d", "#1E90FF"]
}

METHOD_NAME_TO_COLOR = {
    'LC RL': ('#035e8f', 0),
    'LC SFT': ('#045941', 0),
    'Summary ICL 8-Shot': ('#045941', 1),

    'Factuality RL': ('#cb3b3b', 0),
    'ICL 8-Shot': ('#cb3b3b', 1),
    'Factuality SFT': ('#cb3b3b', 2),
    'Claude Distill': ('#cb3b3b', 3),

    'Llama 2 Chat 7B': ('#7f6000', 2),

    'GPT-4 0-Shot': ('#710e7d', 0),
    'GPT-4 JAFU 0-Shot': ('#710e7d', 1),
    'GPT-4 ICL 8-Shot': ('#710e7d', 2),
    'GPT-4 Summary ICL 8-Shot': ('#710e7d', 3),

    # Claude 2.0
    'Direct Summary Eval': ('#1E90FF', 0),
}

METHOD_NAME_TO_SHAPE = {
    # LC and Factuality RL: diamond
    'LC RL': 'D',
    'Factuality RL': 'D',

    # LC and Factuality SFT: circle
    'LC SFT': 'o',
    'Factuality SFT': 'o',

    # Claude Distill: triangle_up
    'Claude Distill': '^',

    # ICL methods: square
    'Summary ICL 8-Shot': 's',
    'ICL 8-Shot': 's',

    # Llama 2 Chat 7B: triangle_left
    'Llama 2 Chat 7B': '<',

    # GPT-4: pentagon
    'GPT-4 0-Shot': 'p',

    # GPT-4 JAFU: triangle_down
    'GPT-4 JAFU 0-Shot': 'v',

    # GPT-4 ICL: star
    'GPT-4 ICL 8-Shot': 'P',

    # GPT-4 Summary ICL: hexagon
    'GPT-4 Summary ICL 8-Shot': 'H',

    # Claude 2.0: triangle_right
    'Direct Summary Eval': '>',
}


def set_default_matplotlib_constants():
    # From Tim. G.J. Rudner's utilities.
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['figure.figsize'] = (6, 4)
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 1.0
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.pad'] = 8
    plt.rcParams['ytick.major.pad'] = 8
    plt.rcParams['axes.grid'] = True


def set_reliability_diagram_constants():
    # Make the text huge
    plt.rcParams['font.size'] = 30
    plt.rcParams['axes.labelsize'] = 25
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20


def plot_qa_frontiers(datasets, results_df):
    for dataset in datasets:
        print(f"Plotting for {dataset}")
        filtered_df = results_df[results_df['dataset'] == dataset]
        x = filtered_df['ece'].values
        y = filtered_df['accuracy'].values
        lower_ci = filtered_df['ci_lower'].values
        upper_ci = filtered_df['ci_upper'].values
        lower_ci_ece = filtered_df['ece_lower'].values
        upper_ci_ece = filtered_df['ece_upper'].values
        fig, ax = plt.subplots()
        for i in range(len(filtered_df)):
            color_hex, cmap_position = METHOD_NAME_TO_COLOR[filtered_df['method'].values[i]]
            color = BASE_COLOR_TO_CMAP[color_hex](cmap_position)
            ax.errorbar(
                x[i],
                y[i],
                yerr=[[y[i] - lower_ci[i]], [upper_ci[i] - y[i]]],
                xerr=[[x[i] - lower_ci_ece[i]], [upper_ci_ece[i] - x[i]]],
                fmt='o',
                color=color,
                ecolor='gray',
                capsize=5,
                marker=METHOD_NAME_TO_SHAPE[filtered_df['method'].values[i]],
                markersize=10,
                capthick=2,
                label=filtered_df['method'].values[i])

        ax.set_xlabel('ECE')
        ax.set_ylabel('Accuracy')

        # Plot legend outside of the plot
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.show()

        print("--" * 40)


def plot_factscore_frontiers(datasets, results_df):
    for dataset in datasets:
        print(f"Plotting for {dataset}")
        filtered_df = results_df[results_df['dataset'] == dataset]
        x = filtered_df['pooled_ece'].values
        y = filtered_df['pooled_proportion_supported'].values
        lower_ci = filtered_df['pooled_proportion_supported_ci_lower'].values
        upper_ci = filtered_df['pooled_proportion_supported_ci_upper'].values
        lower_ci_ece = filtered_df['pooled_ece_ci_lower'].values
        upper_ci_ece = filtered_df['pooled_ece_ci_upper'].values
        fig, ax = plt.subplots()
        for i in range(len(filtered_df)):
            color_hex, cmap_position = METHOD_NAME_TO_COLOR[filtered_df['method'].values[i]]
            color = BASE_COLOR_TO_CMAP[color_hex](cmap_position)
            ax.errorbar(
                x[i],
                y[i],
                yerr=[[y[i] - lower_ci[i]], [upper_ci[i] - y[i]]],
                xerr=[[x[i] - lower_ci_ece[i]], [upper_ci_ece[i] - x[i]]],
                fmt='o',
                color=color,
                ecolor='gray',
                capsize=5,
                marker=METHOD_NAME_TO_SHAPE[filtered_df['method'].values[i]],
                markersize=10,
                capthick=2,
                label=filtered_df['method'].values[i])

        ax.set_xlabel('ECE')
        ax.set_ylabel('Accuracy')

        # Plot legend outside of the plot
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.show()

        print("--" * 40)


def plot_reliability_diagram(accuracies_arr, confidences_arr, counts_arr, ece, model_name=None):
    # Drop if counts are 0
    accuracies_arr = accuracies_arr[counts_arr != 0]
    confidences_arr = confidences_arr[counts_arr != 0]
    counts_arr = counts_arr[counts_arr != 0]

    # If x-values are identical, aggregate
    confidence_to_counts = defaultdict(list)
    confidence_to_count_correct = defaultdict(list)

    for confidence, accuracy, count in zip(confidences_arr, accuracies_arr, counts_arr):
        confidence_to_counts[confidence].append(count)
        confidence_to_count_correct[confidence].append(accuracy * count)

    confidences_arr = np.array(list(sorted(confidence_to_counts.keys())))
    accuracies_arr = np.array(
        [np.sum(confidence_to_count_correct[confidence]) / np.sum(confidence_to_counts[confidence]) for confidence in
         confidences_arr])

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated', color='black')
    ax.plot(confidences_arr, accuracies_arr, marker='o', label=f'{model_name} (ECE={ece:.3f})', color='#0074b3')

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')

    ax.legend()
    plt.show()


def load_qa_non_confidence_results_default_eval_hypers(internal_model_name, gen_setting, dataset, results_path,
                                                       n_examples=None):
    if dataset in {'sciq', 'bioasq'}:
        semantic_equivalence_model = "claude-3-opus-20240229"
        semantic_equivalence_prompt = "eval__check_semantic_equivalence_10shot_batch10_claude_chat"
    elif dataset in {'trivia_qa', 'jeopardy'}:
        semantic_equivalence_model = "claude-2.0"
        semantic_equivalence_prompt = "eval__check_semantic_equivalence_10shot_batch10"
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported yet.")

    # ExtractAnswers path
    extract_answers_path = os.path.join(
        results_path,
        "answer_extraction",
        dataset,
        "test",
        internal_model_name,
        'claude-2.0',
        f'max_ex-{n_examples}--seed-1',
        f'gen_prompt-{gen_setting}',
        f'extr_prompt-eval__extract_answers_claude_10shot',
        'gen_temp-0.3',
        'ext_temp-0.2',
        'answer_extractions.csv')

    # SemanticEquivalence path
    semantic_equivalence_path = os.path.join(
        results_path,
        "semantic_equivalence",
        dataset,
        "test",
        internal_model_name,
        'claude-2.0',
        'claude-2.0',
        semantic_equivalence_model,
        f'skip_forecast_probs-True--max_ex-{n_examples}--seed-1',
        f'gen_prompt-{gen_setting}',
        f'extr_prompt-eval__extract_answers_claude_10shot',
        'forecast_prompt-eval__forecast_probs_claude_0shot',
        f'sem_eq_prompt-{semantic_equivalence_prompt}',
        f'gen_temp-0.3',
        f'ext_temp-0.2',
        f'forecast_temp-0.2',
        f'sem_eq_temp-0.2',
        'semantic_equivalence.csv')

    extract_answers_df = pd.read_csv(extract_answers_path)
    semantic_equivalence_df = pd.read_csv(semantic_equivalence_path)
    return extract_answers_df, semantic_equivalence_df


def load_qa_confidence_results_default_eval_hypers(internal_model_name, gen_setting, dataset, results_path,
                                                   n_examples=None):
    if dataset in {'sciq', 'bioasq'}:
        semantic_equivalence_model = "claude-3-opus-20240229"
        semantic_equivalence_prompt = "eval__check_semantic_equivalence_10shot_batch10_claude_chat"
    elif dataset in {'trivia_qa', 'jeopardy'}:
        semantic_equivalence_model = "claude-2.0"
        semantic_equivalence_prompt = "eval__check_semantic_equivalence_10shot_batch10"
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported yet.")

    # InterpretProbs path
    interpret_probs_path = os.path.join(
        results_path,
        "forecast_probs",
        dataset,
        "test",
        internal_model_name,
        'claude-2.0',
        'claude-2.0',
        f'skip_answer_extraction-False--max_ex-{n_examples}--seed-1',
        f'gen_prompt-{gen_setting}',
        f'extr_prompt-eval__extract_answers_claude_10shot',
        f'forecast_prompt-eval__forecast_probs_claude_0shot',
        f'gen_temp-0.3',
        f'ext_temp-0.2',
        f'forecast_temp-0.2',
        'probability_forecasts.csv')

    # SemanticEquivalence path
    semantic_equivalence_path = os.path.join(
        results_path,
        "semantic_equivalence",
        dataset,
        "test",
        internal_model_name,
        'claude-2.0',
        'claude-2.0',
        semantic_equivalence_model,
        f'skip_forecast_probs-False--max_ex-{n_examples}--seed-1',
        f'gen_prompt-{gen_setting}',
        f'extr_prompt-eval__extract_answers_claude_10shot',
        'forecast_prompt-eval__forecast_probs_claude_0shot',
        f'sem_eq_prompt-{semantic_equivalence_prompt}',
        f'gen_temp-0.3',
        f'ext_temp-0.2',
        f'forecast_temp-0.2',
        f'sem_eq_temp-0.2',
        'semantic_equivalence.csv')

    interpret_probs_df = pd.read_csv(interpret_probs_path)
    semantic_equivalence_df = pd.read_csv(semantic_equivalence_path)
    return interpret_probs_df, semantic_equivalence_df


def load_factscore_nonconfidence_results_default_eval_hypers(internal_model_name, gen_setting, dataset, results_path,
                                                             n_examples=None):
    fact_check_path = os.path.join(
        results_path,
        "fact_checker",
        dataset,
        "test",
        internal_model_name,
        "claude-2.0",
        "claude-2.0",
        "claude-2.0",
        f"max_ex-{n_examples}--seed-1",
        f"gen_prompt-{gen_setting}",
        f"decomp_prompt-biography_generation_eval__nonconfidence_decompose_claims_claude_8shot",
        f"filter_prompt-biography_generation_eval__confidence_filter_claude_1shot",
        "gen_temp-0.3",
        "decomp_temp-0.2",
        "filter_temp-0.2",
        "fact_check_temp-0.2",
        "fact_checker.csv")
    return pd.read_csv(fact_check_path)


def load_factscore_confidence_results_default_eval_hypers(internal_model_name, gen_setting, dataset, results_path,
                                                          n_examples=None):
    fact_check_path = os.path.join(
        results_path,
        "fact_checker",
        dataset,
        "test",
        internal_model_name,
        "claude-2.0",
        "claude-2.0",
        "claude-2.0",
        f"max_ex-{n_examples}--seed-1",
        f"gen_prompt-{gen_setting}",
        f"decomp_prompt-biography_generation_eval__confidence_decompose_claims_claude_8shot",
        f"filter_prompt-biography_generation_eval__confidence_filter_claude_1shot",
        "gen_temp-0.3",
        "decomp_temp-0.2",
        "filter_temp-0.2",
        "fact_check_temp-0.2",
        "fact_checker.csv")
    return pd.read_csv(fact_check_path)


def load_eval_results(eval_models, load_result_fn, datasets, results_path, n_examples=None):
    results = {}
    for internal_model_name, gen_setting, display_name in eval_models:
        for dataset in datasets:
            results_key = (display_name, dataset)
            try:
                results[results_key] = load_result_fn(internal_model_name, gen_setting, dataset, results_path,
                                                      n_examples)
                print(f"Loaded {results_key}")
            except:
                print(f"Failed to load {results_key}")

    return results


def get_results_df_given_methods_to_display(methods_to_display, metric_dicts_to_use):
    results_df_entries = []
    for metric_dict in metric_dicts_to_use:
        for (model_name, dataset), metrics in metric_dict.items():
            if methods_to_display is None or model_name in methods_to_display:
                results_df_entries.append({
                    'method': model_name,
                    'dataset': dataset,
                    **metrics
                })

    return pd.DataFrame(results_df_entries)
