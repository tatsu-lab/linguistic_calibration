# Copyright 2024 Neil Band
# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import tqdm
import transformers
from accelerate import DistributedType
from accelerate.optimizer import AcceleratedOptimizer
from scipy.special import expit
from torch import nn
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, TensorDataset
from transformers.trainer_utils import enable_full_determinism, set_seed

from . import kl_controller
from .. import accelerate_patch, common, data_preprocessor, logging, utils
from ..inference import decode, score
from ..types import LRScheduler, Tensor, Mapping, List

FIRST_STEP_IDX = 1

logger = logging.get_logger(__name__)


class RLTrainer(object):
    def __init__(
        self,
        args,
        train_dataset: data_preprocessor.LCQueryDataset,
        eval_dataset: data_preprocessor.LCQueryDataset,
        data_collator: Callable,
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        accelerator: accelerate_patch.MyAccelerator,
        answer_extractor: Optional[nn.Module] = None,
        answer_extractor_tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        super(RLTrainer, self).__init__()
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.answer_extractor = answer_extractor
        self.answer_extractor_tokenizer = answer_extractor_tokenizer
        self.lr_scheduler = lr_scheduler
        self.kl_ctl = kl_controller.make_kl_controller(args, self.accelerator)
        self.log_history = []
        self.args.set_truncate_token_ids(self.tokenizer)
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)

        self.value_model_prompt_template = utils.read(self.args.value_model_prompt_path)
        self.reward_model_prompt_template = utils.read(self.args.reward_model_prompt_path)
        if self.args.rl_type == "decision_based_rl":
            assert self.answer_extractor is not None, "answer_extractor must be provided for decision-based RL."
            self.answer_extractor_prompt_template = utils.read(self.args.answer_extractor_prompt_path)
        elif self.args.rl_type == "factuality_rl":
            self.answer_extractor_prompt_template = None
        else:
            raise ValueError(f"Invalid RL type: {self.args.rl_type}")

        self.reward_operations_list = []
        self.reward_operations_names = []

        if self.args.rl_type == "decision_based_rl":
            self.reward_operations_list.append(self.compute_cross_entropy)
            self.reward_operations_names.append("cross_entropy")

            # Add normalization term
            self.reward_operations_list.append(self.compute_answer_extractor_normalization)
            self.reward_operations_names.append("answer_extractor_normalization")
        else:
            # Binary correctness term
            self.reward_operations_list.append(self.compute_binary_correctness)
            self.reward_operations_names.append("binary_correctness")

        logger.info("Loaded reward operations: " + ", ".join(self.reward_operations_names))

    @abc.abstractmethod
    @torch.inference_mode()
    def rollout(self, queries_data) -> Dict[str, Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, rollouts: Dict[str, Tensor]) -> Tuple[Tensor, Dict]:
        raise NotImplementedError

    @abc.abstractmethod
    @torch.inference_mode()
    def record_step_stats(self, train_stats, rollouts, step_idx, **kwargs):
        raise NotImplementedError

    def apply_reward_operations(
        self,
        logits: Sequence[float],
        **kwargs
    ) -> Mapping:
        """Apply the reward operations to the logits.

        Args:
            logits: logits for each example.

        Returns:
            Dict containing the combined outputs.
        """
        interpretations = expit(np.array(logits))

        outputs = []
        for reward_operation in self.reward_operations_list:
            outputs.append(reward_operation(
                interpretations=interpretations, **kwargs))

        return self.combine_reward_operation_outputs(outputs)

    def combine_reward_operation_outputs(
        self,
        outputs: Sequence[Dict]
    ):
        """Combine the outputs of the reward operations into a single dict.

        Args:
            outputs: Sequence of dicts, each containing the outputs of a reward operation.

        Returns:
            Dict containing the combined outputs.
        """
        combined_output = {}
        for reward_op_name, output_dict in utils.zip_(
                self.reward_operations_names, outputs):
            for key, value in output_dict.items():
                if isinstance(value, torch.Tensor):
                    combined_output[f"{reward_op_name}_{key}"] = value.detach().cpu().numpy()
                else:
                    combined_output[f"{reward_op_name}_{key}"] = value
                if key in {'interpretations'}:
                    combined_output[key] = value
                elif key in {'scores', 'rewards'}:
                    # Aggregate the scores and rewards.
                    total_key = f"total_{key}"
                    if total_key in combined_output:
                        combined_output[total_key] += value
                    else:
                        combined_output[total_key] = value

        return combined_output

    def compute_cross_entropy(
        self,
        interpretations: np.ndarray,
        n_samples_per_example: Sequence[int],
        **kwargs
    ) -> Dict:
        """Compute the cross entropy for a given set of interpretations.

        Args:
            interpretations: probs for each example.
            n_samples_per_example: Sequence of number of samples per example.
                If provided, we need to only compute the cross-entropy for
                the first element in every n_samples_per_example elements.
                This first element corresponds to the ground-truth answer;
                the remaining elements are the extracted answers.

        Returns:
            Dict containing the cross-entropy scores.
        """
        interpretations_to_return = []
        new_interpretations = []
        running_index = 0
        for i, n_samples in enumerate(n_samples_per_example):
            new_interpretations.append(interpretations[running_index])
            interpretations_to_return.append(interpretations[running_index:running_index + n_samples])
            running_index += n_samples

        interpretations = np.array(new_interpretations)
        interpretations = np.clip(interpretations, self.args.cross_entropy_epsilon, 1)
        rewards = np.log(interpretations)

        return {
            'interpretations': interpretations_to_return,
            'scores': -rewards,
            'rewards': torch.tensor(rewards, device=self.accelerator.device)
        }

    def compute_binary_correctness(
        self,
        interpretations: np.ndarray,
        **kwargs
    ) -> Dict:
        """Compute the binary correctness based on a given set of probabilistic interpretations.

        This just amounts to thresholding the interpretations at 0.5.

        Args:
            interpretations: probs for each example.

        Returns:
            Binary correctness for each example.
        """
        rewards = (interpretations > 0.5).astype(int)
        return {
            'interpretations': interpretations,
            'scores': -rewards,
            'rewards': torch.tensor(rewards, device=self.accelerator.device)
        }

    def compute_answer_extractor_normalization(
        self,
        interpretations: np.ndarray,
        n_samples_per_example: Sequence[int],
        **kwargs
    ) -> Dict:
        """Compute the normalization term using interpretations.

        Args:
            interpretations: probs for each example.
            n_samples_per_example: Sequence of number of answer choices per example.

        Returns:
            Normalization term for each example.

        # Test case:
        interpretations = np.array([0.5, 0.2, 0.3, 0.4, 0.9, 0.6, 1.2])
        n_samples_per_example = [3, 2, 1, 1]
        """
        interpretation_sums = []
        running_index = 0
        for i, n_samples in enumerate(n_samples_per_example):
            interpretations_per_example = interpretations[running_index:running_index + n_samples]

            # Exclude the first entry, which is the ground truth answer
            interpretations_per_example = interpretations_per_example[1:]

            # Sum interpretations per example
            interpretation_sum = np.sum(interpretations_per_example)
            interpretation_sums.append(interpretation_sum)

            running_index += n_samples

        # Compute normalization term
        # Concatenate interpretation sums
        interpretation_sums = np.array(interpretation_sums)

        # * Compute normalization term *
        abs_distance_to_one = np.abs(interpretation_sums - 1)

        # Multiply by lambda
        scores = self.args.answer_extractor_normalization_lambda * abs_distance_to_one
        rewards = -scores

        return {
            'scores': scores,
            'rewards': torch.tensor(rewards, device=self.accelerator.device)
        }

    @staticmethod
    def postprocess_answer_extractions(answer_extractions: Sequence[str]) -> List[List[str]]:
        """Postprocess answer extractions.

        Args:
            answer_extractions: List of answer extractions.

        Returns:
            List of answer extraction lists.
        """
        answer_extraction_lists = []
        for answer_extraction in answer_extractions:
            if answer_extraction == 'No Answer':
                answer_extraction_list = []
            else:
                answer_extraction_list = answer_extraction.split(';')
                # Strip whitespace
                answer_extraction_list = [
                    answer.strip() for answer in answer_extraction_list]

                # Include only unique answers
                # Note this operation might change order, but that doesn't matter --
                # we use it to construct independent sequences passed to the ForecastProbs model.
                answer_extraction_list = list(set(answer_extraction_list))

            answer_extraction_lists.append(answer_extraction_list)

        return answer_extraction_lists

    def construct_rm_text_sequences_for_answer_extraction(
        self,
        answer_extractions: Sequence[str],
        list_data_dict: Sequence[Dict],
        return_answer_extraction_lists: bool = False,
    ):
        # Process answer extractions into a list of lists
        # Note that we deduplicate and the order of the answers may change. This doesn't matter
        # as we use it to construct independent sequences passed to the ForecastProbs model.
        answer_extraction_lists = self.postprocess_answer_extractions(
            answer_extractions=answer_extractions)

        n_samples_per_example = []
        sequences = []
        answer_extraction_lists_to_return = []
        for answer_extraction_list, data_dict in zip(answer_extraction_lists, list_data_dict):
            # We always include the top ground-truth answer
            top_answer_prompt = self.reward_model_prompt_template.format(**data_dict)
            sequences.append(top_answer_prompt)
            n_answer_choices = 1
            answer_extraction_list_to_return = [data_dict['ground_truth_top_answer']]

            # Include the extracted answers
            for answer in answer_extraction_list:
                answer_prompt = self.reward_model_prompt_template.format(
                    question=data_dict['question'],
                    generated_paragraph=data_dict['generated_paragraph'],
                    ground_truth_top_answer=answer)
                sequences.append(answer_prompt)

            n_answer_choices += len(answer_extraction_list)
            n_samples_per_example.append(n_answer_choices)
            answer_extraction_list_to_return += answer_extraction_list
            answer_extraction_lists_to_return.append(answer_extraction_list_to_return)

        if return_answer_extraction_lists:
            return n_samples_per_example, sequences, answer_extraction_lists_to_return
        else:
            return n_samples_per_example, sequences

    @property
    def optimizable_params(self):
        return [p for p in self.policy.parameters() if p.requires_grad and p.grad is not None]

    @torch.inference_mode()
    def _compute_grad_norm(self):
        grad_norm = torch.stack([p.grad.norm(2) for p in self.optimizable_params]).norm(2)
        if (
            self.accelerator.distributed_type == DistributedType.FSDP
            and self.policy.sharding_strategy != ShardingStrategy.NO_SHARD
        ):
            # When parameters are sharded, we need to gather each grad norm and then aggregate.
            grad_norms = [torch.zeros_like(grad_norm) for _ in range(self.accelerator.num_processes)]
            dist.all_gather(grad_norms, grad_norm)
            grad_norm = torch.stack(grad_norms).norm(2)
        return grad_norm

    @torch.inference_mode()
    def _compute_param_norm(self):
        param_norm = torch.stack([p.norm(2) for p in self.optimizable_params]).norm(2)
        if (
            self.accelerator.distributed_type == DistributedType.FSDP
            and self.policy.sharding_strategy != ShardingStrategy.NO_SHARD
        ):
            # When parameters are sharded, we need to gather each grad norm and then aggregate.
            param_norms = [torch.zeros_like(param_norm) for _ in range(self.accelerator.num_processes)]
            dist.all_gather(param_norms, param_norm)
            param_norm = torch.stack(param_norms).norm(2)
        return param_norm

    def _make_fsdp_happy(self):
        """Simply do a forward pass with the wrapped model at first.

        FSDP has some weird bugs; need this flush before running a non-forward method!
        This function should assume grad context of caller and
        not be wrapped with `torch.no_grad` or `torch.enable_grad`!!!
        """
        if self.accelerator.distributed_type == DistributedType.FSDP:
            inputs = self.tokenizer("fsdp are you happy now? :)" * 50, return_tensors="pt")
            inputs = common.prepare_inputs(inputs, device=self.accelerator.device)
            self.policy(inputs["input_ids"], inputs["attention_mask"], inputs["input_ids"])

    def step_with_rollouts(self, rollouts):
        """Based on fixed rollouts, run PPO for multiple epochs."""
        assert isinstance(self.optimizer, AcceleratedOptimizer), (
            "`optimizer` must be pushed through `accelerator.prepare`. "
            "Otherwise the `accelerator.accumulate` context manager won't correctly disable `zero_grad` or `step`."
        )
        rollouts_dataloader = self.get_rollouts_dataloader(rollouts=rollouts)
        stats_list = []
        for epoch_idx in range(self.args.noptepochs):
            for batch_idx, rollouts_batch in tqdm.tqdm(
                enumerate(rollouts_dataloader, 1), disable=not self.accelerator.is_main_process, desc="gradstep"
            ):
                with self.accelerator.accumulate(self.policy):
                    ppo_loss, stats_for_this_step = self.compute_loss(rollouts_batch)
                    self.accelerator.backward(ppo_loss)
                    if self.accelerator.sync_gradients:
                        # Gradient norm almost blows up at some point, but stabilizes eventually, even w/o clipping.
                        if self.args.max_grad_norm is not None:
                            self.accelerator.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
                        stats_for_this_step["loss/grad_norm"] = self._compute_grad_norm()
                        stats_list.append(stats_for_this_step)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
        return common.merge_dict(stats_list, torch.stack)  # list of dict -> dict: str -> 1-D tensor

    def step(self, train_dataloader, step_idx: int):
        queries_batches = [next(train_dataloader) for _ in range(self.args.rollout_accumulation_steps)]
        rollouts = self.rollout(queries_batches)
        train_stats = self.step_with_rollouts(rollouts)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        stats = self.record_step_stats(
            rollouts=rollouts, train_stats=train_stats, step_idx=step_idx, kl_coef=self.kl_ctl.value
        )
        self.kl_ctl.step(stats["objective/kl_sum_seq"])
        return stats

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # TODO(@nband): generalize to other optimizers. For now, 8-bit paged Adam is required to fit 7B PPO in memory.
        # optimizer = trainer_utils.create_optimizer(args=self.args, model=self.policy, optimizer=self.optimizer)
        # lr_scheduler = trainer_utils.create_scheduler(
        #     args=self.args, optimizer=optimizer, lr_scheduler=self.lr_scheduler, num_training_steps=num_training_steps
        # )
        from bitsandbytes.optim import AdamW

        is_paged = True
        optim_bits = 8
        optimizer_cls = AdamW
        optimizer_kwargs = {
            'lr': self.args.learning_rate,
            'eps': self.args.adam_epsilon,
        }
        bnb_kwargs = {"is_paged": is_paged, "optim_bits": optim_bits}
        optimizer_kwargs.update(bnb_kwargs)
        optimizer = optimizer_cls(self.policy.parameters(), **optimizer_kwargs)
        logger.warning(f"optimizer: {optimizer}", main_process_only=True)

        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        logger.warning(f"lr_scheduler: {lr_scheduler}", main_process_only=True)

        self.optimizer, self.lr_scheduler = self.accelerator.prepare(optimizer, lr_scheduler)
        self.accelerator.register_for_checkpointing(self.lr_scheduler)  # LR scheduler needs another call to save.
        return self.optimizer, self.lr_scheduler

    def train(self):
        """Entry point for training."""
        if self.args.total_steps is not None:
            total_steps = self.args.total_steps
            total_episodes = total_steps * self.args.rollout_batch_size
            total_epochs = total_episodes / len(self.train_dataset)
        else:
            total_epochs = self.args.total_epochs
            total_episodes = len(self.train_dataset) * total_epochs  # noqa
            total_steps = total_episodes // self.args.rollout_batch_size  # noqa

        logger.warning(
            f"***Training starts***\n"
            f"Total epochs: {total_epochs} => Total episodes: {total_episodes} => Total steps: {total_steps}"
        )

        self.create_optimizer_and_scheduler(total_steps)
        infinite_train_dataloader = self.get_train_dataloader()
        for step_idx in tqdm.tqdm(
            range(FIRST_STEP_IDX, total_steps + FIRST_STEP_IDX),
            disable=not self.accelerator.is_main_process,
            desc="steps",
            total=total_steps,
        ):
            if step_idx % self.args.save_steps == 0 or step_idx in self.args.save_steps_extra_list:
                if step_idx >= self.args.save_after:
                    self.save_model(utils.join(self.args.output_dir, f"checkpoint-{step_idx}"))
            if (self.args.eval_steps is not None and step_idx % self.args.eval_steps == 0) or step_idx == 1:
                self.evaluate(step_idx)
            self.log_history.append(self.step(infinite_train_dataloader, step_idx))
        return self.log_history

    @torch.inference_mode()
    def evaluate(self, step_idx: int, unwrapped_policy=None):
        """Evaluate by generating sequences with test prefixes.

        FSDP compat: all devices should do the forward pass, since sharded params need to be summoned.
                     only write results in the main process.
        """
        # TODO: unhardcode inference args.
        logger.warning(f"Start evaluation at step: {step_idx}", main_process_only=True)

        prompts, list_dict_data = self.eval_dataset.prompts, self.eval_dataset.list_dict_data
        if any(item is None for item in (prompts, list_dict_data)):
            logger.warning("No evaluation data, skipping evaluation.", main_process_only=True)
            return

        # Constants.
        model_name = Path(self.args.output_dir).stem  # Don't use the helper in common, as no checkpoint is saved yet.
        model_name_at_step = f"{model_name}_ckpt_{step_idx}"
        temperature = 0.7
        del model_name

        # Start evaluation.
        self.policy.eval()
        self._make_fsdp_happy()

        if self.args.rl_type == "decision_based_rl":
            self.answer_extractor.eval()

        self.accelerator.wait_for_everyone()

        with FSDP.summon_full_params(self.policy, writeback=False, recurse=False):
            outputs = decode.respond_to_prompts_with_huggingface_given_wrapped_model(
                model=self.policy,
                tokenizer=self.tokenizer,
                prompts=prompts,
                temperature=temperature,
                per_device_batch_size=self.args.eval_rollout_per_device_batch_size,
                divide_work=True,
                is_policy=True,
            )

        ground_truth_top_answers = self.eval_dataset.unfiltered_ground_truth_top_answers
        list_data_dict = [{
                "question": question,
                "generated_paragraph": generated_paragraph,
                "ground_truth_top_answer": ground_truth_top_answer,
            } for question, generated_paragraph, ground_truth_top_answer in utils.zip_(
                self.eval_dataset.unfiltered_user_decision_questions, outputs, ground_truth_top_answers)]

        if self.args.rl_type == "decision_based_rl":
            answer_extractor_sequences = [
                self.answer_extractor_prompt_template.format(
                    question=data_dict['question'],
                    generated_paragraph=data_dict['generated_paragraph'],
                ) for data_dict in list_data_dict
            ]
            self.accelerator.wait_for_everyone()
            with FSDP.summon_full_params(self.answer_extractor, writeback=False, recurse=False):
                answer_extractions = decode.respond_to_prompts_with_huggingface_given_wrapped_model(
                    model=self.answer_extractor,
                    tokenizer=self.answer_extractor_tokenizer,
                    prompts=answer_extractor_sequences,
                    decoding_args=decode.HFDecodingArguments(
                        max_new_tokens=self.args.answer_extractor_response_len,
                        temperature=self.args.answer_extractor_temperature),
                    per_device_batch_size=self.args.eval_rollout_per_device_batch_size,
                    divide_work=True,
                    is_policy=False,
                )

            n_samples_per_example, sequences, answer_extraction_lists = (
                self.construct_rm_text_sequences_for_answer_extraction(
                    answer_extractions=answer_extractions,
                    list_data_dict=list_data_dict,
                    return_answer_extraction_lists=True))
            self.accelerator.wait_for_everyone()
        else:
            n_samples_per_example = None
            sequences = [self.reward_model_prompt_template.format(**data_dict) for data_dict in list_data_dict]

        rewards = score.score_sequences_with_accelerate_given_model(
            model=self.reward_model,
            tokenizer=self.tokenizer,
            sequences=sequences,
            per_device_batch_size=self.args.eval_reward_model_per_device_batch_size,
            mixed_precision=self.accelerator.mixed_precision,
        )

        if self.accelerator.is_main_process:
            reward_dict = self.apply_reward_operations(rewards, n_samples_per_example=n_samples_per_example)
            rewards = reward_dict['total_rewards']
            rewards = rewards.cpu().numpy()
            scores = reward_dict['total_scores']
            interpretations = reward_dict['interpretations']
            results = [
                {
                    model_name_at_step: output,
                    "temperature": temperature,
                    **example}
                for output, example in utils.zip_(outputs, list_dict_data)
            ]

            # Add interpretations to results
            for i in range(len(results)):
                for key in reward_dict:
                    if key == "total_rewards":
                        results[i]["reward"] = reward_dict[key][i].cpu().item()
                    else:
                        value = reward_dict[key][i]
                        if isinstance(value, torch.Tensor):
                            value = value.cpu().item()

                        results[i][key] = value

                if self.args.rl_type == "decision_based_rl":
                    results[i]["sample_to_interpretation"] = dict(utils.zip_(
                        answer_extraction_lists[i], interpretations[i]))

            if self.args.output_dir is not None:
                utils.jdump(results, utils.join(self.args.output_dir, f"eval_results_{step_idx}.json"))

            reward_dict = {"reward": utils.mean(rewards), "score": utils.mean(scores)}

            # Compute the average number of answer choices per example.
            if self.args.rl_type == "decision_based_rl":
                avg_n_answer_choices = np.array(n_samples_per_example).mean()
                reward_dict["avg_n_answer_choices"] = avg_n_answer_choices
            else:
                reward_dict["interpretation"] = utils.mean(interpretations)

            self.accelerator.log({"gold_eval": reward_dict}, step=step_idx)
            logger.warning(f"End evaluation at step: {step_idx}. Processed {len(results)} examples")

    @abc.abstractmethod
    @torch.inference_mode()
    def save_model(self, output_dir: Optional[str] = None):
        raise NotImplementedError

    def _log_batch_size(self, loader: DataLoader, loader_name):
        batch = next(iter(loader))
        if isinstance(batch, torch.Tensor):
            batch_size = batch.shape[0]
        elif isinstance(batch, (list, tuple)):
            batch_size = batch[0]
        else:
            tensor = list(batch.values())[0]
            batch_size = tensor.size(0)
        logger.warning(f"Batch size of {loader_name} dataloader: {batch_size}", main_process_only=True)

    def get_train_dataloader(self):
        logger.warning(f"Train dataset size: {len(self.train_dataset)}", main_process_only=True)  # noqa
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.data_collator,
            batch_size=self.args.rollout_per_device_batch_size,
            shuffle=True,
            drop_last=True,
        )
        train_dataloader = self.accelerator.prepare(train_dataloader)  # noqa
        self._log_batch_size(train_dataloader, "train_dataloader")
        return utils.InfiniteLoader(train_dataloader)

    def get_rollouts_dataloader(self, rollouts: Dict[str, Tensor], shuffle=True, drop_last=True, keys=None):
        if keys is None:
            keys = tuple(rollouts.keys())

        def collate_rollouts(instances: Sequence[tuple]):
            return {key: torch.stack([instance[idx] for instance in instances]) for idx, key in enumerate(keys)}

        rollouts_dataset = TensorDataset(*[rollouts[key] for key in keys])
        rollouts_dataloader = DataLoader(
            dataset=rollouts_dataset,
            batch_size=self.args.step_per_device_batch_size,
            collate_fn=collate_rollouts,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        # Do not prepare, since we don't need to shard the rollouts sampled on each batch.
        return rollouts_dataloader
