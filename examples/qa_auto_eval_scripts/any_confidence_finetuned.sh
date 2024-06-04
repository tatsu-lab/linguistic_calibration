#!/bin/bash

paragraph_generator_model_name=$1
dataset_name=$2
max_n_examples=$3

if [ -z "$max_n_examples" ]
then
    max_n_examples=None
fi

python examples/qa_automated_eval.py --max_n_examples=${max_n_examples} \
--paragraph_generator_model_name=${paragraph_generator_model_name} \
--paragraph_generation_prompt=generate_paragraphs_llama_finetuned \
--answer_extractor_model_name=claude-2.0 \
--answer_extractor_prompt=eval/extract_answers_claude_10shot \
--forecast_probs_model_name=claude-2.0 \
--forecast_probs_prompt=eval/forecast_probs_claude_0shot \
--semantic_equivalence_model_name=claude-2.0 \
--semantic_equivalence_prompt=eval/check_semantic_equivalence_10shot_batch10 \
--skip_answer_extraction=False \
--skip_forecast_probs=False \
--dataset_name=${dataset_name} \
--dataset_split=test \
--per_device_batch_size=8 \
--generation_temperature=0.3 \
--extraction_temperature=0.2 \
--forecast_temperature=0.2 \
--seed=1