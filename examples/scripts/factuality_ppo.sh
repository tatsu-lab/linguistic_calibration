#!/bin/bash

output_dir=$1
run_name=$2
reward_model_name_or_path=$3
policy_model_name_or_path=$4

config_file="./examples/accelerate_configs/rlhf_ppo_fsdp_llama_8gpu_disable_forward_prefetch.yaml"

accelerate launch --main_process_port=1234 --config_file "${config_file}" examples/rlhf_ppo.py \
  --run_name "${run_name}" \
  --wandb_project "linguistic_calibration" \
  --rl_type "factuality_rl" \
  --flash_attn True \
  --optim paged_adamw_8bit \
  --total_steps 1500 \
  --rollout_batch_size 512 \
  --step_batch_size 512 \
  --step_per_device_batch_size 2 \
  --rollout_per_device_batch_size 16 \
  --eval_rollout_per_device_batch_size 16 \
  --eval_reward_model_per_device_batch_size 32 \
  --noptepochs 1 \
  --temperature 0.7 \
  --kl_coef 0.1 \
  --adaptive_kl False \
  --query_len 128 \
  --response_len 300 \
  --policy_model_name_or_path "${policy_model_name_or_path}" \
  --reward_model_name_or_path "${reward_model_name_or_path}" \
  --output_dir "${output_dir}" \
  --eval_steps 20 \
  --save_steps 20 \
  --save_after 800 \
  --policy_prompt_path "./src/linguistic_calibration/prompts/paragraph_generation/generate_paragraphs_llama_finetuned.txt" \
  --reward_model_prompt_path "./src/linguistic_calibration/prompts/train/reward_model_binary_correctness_llama_finetuned.txt" \
  --value_model_prompt_path "./src/linguistic_calibration/prompts/train/value_model_llama_finetuned.txt" \
  --learning_rate 1e-5 \
  --init_value_with_reward True \
  --warmup_steps 5 \
  --bf16 True \
  --gradient_checkpointing True \
  --reward_offset 0.0