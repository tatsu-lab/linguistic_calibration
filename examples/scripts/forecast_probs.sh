#!/bin/bash

output_dir=$1
run_name=$2
policy_model_name_or_path=$3

config_file="./examples/accelerate_configs/fsdp_llama2_4gpu.yaml"

accelerate launch --main_process_port 1234 --config_file "${config_file}" examples/reward_modeling.py \
  --fp16 False \
  --bf16 True \
  --optim paged_adamw_8bit \
  --seed 42 \
  --model_name_or_path "${policy_model_name_or_path}" \
  --prompt_template_path "./src/linguistic_calibration/prompts/train/reward_model_forecastprobs_llama_finetuned.txt" \
  --dataset_name "reward_model_training" \
  --reward_model_type "forecast_probs" \
  --output_dir "${output_dir}" \
  --model_max_length 1024 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 32 \
  --eval_steps 25 \
  --save_strategy "steps" \
  --save_steps 25 \
  --save_total_limit 50 \
  --learning_rate 5e-6 \
  --weight_decay 0.0 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --evaluation_strategy "steps" \
  --logging_steps 1 \
  --wandb_project "linguistic_calibration" \
  --run_name "${run_name}" \
  --tf32 True \
  --flash_attn False \
  --save_only_model True