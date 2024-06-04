#!/bin/bash

max_n_examples=$1

if [ -z "$max_n_examples" ]
then
    max_n_examples=None
fi

python examples/biography_generation_automated_eval.py --max_n_examples=${max_n_examples} \
--paragraph_generator_model_name=gpt-4-1106-preview \
--paragraph_generation_prompt=generate_paragraphs_chatml_jafu_0shot \
--claim_decomposition_model_name=claude-2.0 \
--claim_decomposition_prompt=biography_generation_eval/confidence_decompose_claims_claude_8shot \
--claim_uncertainty_filter_model_name=claude-2.0 \
--claim_uncertainty_filter_prompt=biography_generation_eval/confidence_filter_claude_1shot \
--atomic_fact_checker_model_name=claude-2.0 \
--dataset_split=test \
--per_device_batch_size=8 \
--generation_temperature=0.3 \
--decomposition_temperature=0.2 \
--filter_temperature=0.2 \
--fact_check_temperature=0.2 \
--seed=1