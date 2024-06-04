#!/bin/bash

paragraph_generator_model_name=$1
max_n_examples=$2

if [ -z "$max_n_examples" ]
then
    max_n_examples=None
fi

python examples/biography_generation_automated_eval.py --max_n_examples=${max_n_examples} \
--paragraph_generator_model_name=${paragraph_generator_model_name} \
--paragraph_generation_prompt=generate_paragraphs_llama_finetuned \
--claim_decomposition_model_name=claude-2.0 \
--claim_decomposition_prompt=biography_generation_eval/nonconfidence_decompose_claims_claude_8shot \
--atomic_fact_checker_model_name=claude-2.0 \
--dataset_split=test \
--per_device_batch_size=8 \
--generation_temperature=0.3 \
--decomposition_temperature=0.2 \
--filter_temperature=0.2 \
--fact_check_temperature=0.2 \
--seed=1