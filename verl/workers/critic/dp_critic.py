# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Implement a multiprocess PPOCritic
"""

import logging
import os

import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic import BasePPOCritic

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOCritic(BasePPOCritic):
    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Critic use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.device_name = get_device_name()
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.has_logged_prompt_inputs = False
    def _forward_prompt_value(self, micro_batch):
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs
            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["prompt_input_ids"]
            attention_mask = micro_batch["prompt_attention_mask"]
            position_ids = micro_batch["prompt_position_ids"]
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            output = self.critic_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
                output_hidden_states=True,
            )

            if hasattr(self.critic_module, "v_head"):
                hidden_states = output.hidden_states[-1] 
                values_seq = self.critic_module.v_head(hidden_states)
                prompt_value_logit = values_seq[:, -1].squeeze(-1) # (batch_size,)
            else:
                logits = output.logits # (batch_size, prompt_seq_len, 1)
                prompt_value_logit = logits[:, -1].squeeze(-1) # (batch_size,)

            return prompt_value_logit
    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()
        response_length = data.batch["responses"].size(-1)
        prompt_input_ids = data.batch["input_ids"][:, :-response_length]
        prompt_attention_mask = data.batch["attention_mask"][:, :-response_length]
        
        pos_ids = data.batch["position_ids"]
        if pos_ids.dim() == 3: # (c, b, s)
            prompt_position_ids = pos_ids[:, :, :-response_length]
        else: # (b, s)
            prompt_position_ids = pos_ids[:, :-response_length]

        prompt_data_batch = {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "prompt_position_ids": prompt_position_ids,
        }
        current_batch_size = prompt_input_ids.shape[0] 
        prompt_data_batch_td = TensorDict(
            source=prompt_data_batch,
            batch_size=[current_batch_size]
        )
        prompt_data_non_tensor = {}
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            prompt_data_non_tensor["multi_modal_inputs"] = data.non_tensor_batch["multi_modal_inputs"]
        
        prompt_data = DataProto(
            batch=prompt_data_batch_td,
            non_tensor_batch=prompt_data_non_tensor,
            meta_info=data.meta_info
        )

        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(prompt_data, max_token_len=max_token_len)
        else:
            micro_batches = prompt_data.split(micro_batch_size)

        values_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                values = self._forward_prompt_value(model_inputs) # (micro_batch_size)
            values_lst.append(values)
        values = torch.concat(values_lst, dim=0)

        if use_dynamic_bsz:
            values = restore_dynamic_batch(values, batch_idx_list)

        return values

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        self.critic_module.train()
        metrics = {}
        select_keys = [
            "input_ids", "responses", "response_mask", "attention_mask", "position_ids",
            "values",  
            "returns" 
        ]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        mini_batches = data.split(self.config.ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                response_length = mini_batch.batch["responses"].size(-1)
                prompt_input_ids = mini_batch.batch["input_ids"][:, :-response_length]
                prompt_attention_mask = mini_batch.batch["attention_mask"][:, :-response_length]
                pos_ids = mini_batch.batch["position_ids"]
                if pos_ids.dim() == 3:
                    prompt_position_ids = pos_ids[:, :, :-response_length]
                else:
                    prompt_position_ids = pos_ids[:, :-response_length]

                prompt_mb_batch = {
                    "prompt_input_ids": prompt_input_ids,
                    "prompt_attention_mask": prompt_attention_mask,
                    "prompt_position_ids": prompt_position_ids,
                }
                prompt_mb_non_tensor = {}
                if has_multi_modal_inputs:
                    prompt_mb_non_tensor["multi_modal_inputs"] = mini_batch.non_tensor_batch["multi_modal_inputs"]

                R_targets = mini_batch.batch["returns"]
                current_batch_size = prompt_input_ids.shape[0]
                prompt_mb_batch_td = TensorDict(
                    source=prompt_mb_batch,
                    batch_size=[current_batch_size]
                )
                prompt_micro_batch_data = DataProto(
                    batch=prompt_mb_batch_td,
                    non_tensor_batch=prompt_mb_non_tensor,
                    meta_info=mini_batch.meta_info
                )
                
                if self.config.use_dynamic_bsz:
                    raise NotImplementedError("not implemented dp critic with dynamic bsz")
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    R_targets_micro_batches = R_targets.split(self.config.ppo_micro_batch_size_per_gpu)
                    prompt_micro_batches = prompt_micro_batch_data.split(self.config.ppo_micro_batch_size_per_gpu)

                self.critic_optimizer.zero_grad()

                for micro_batch_idx, micro_batch in enumerate(prompt_micro_batches):
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                    R_targets_micro = R_targets_micro_batches[micro_batch_idx].to(get_device_id())
                    R_targets_micro_float = R_targets_micro.float()

                    vpreds_logits = self._forward_prompt_value(model_inputs)
                    vf_loss_per_sample = self.bce_loss_fn(vpreds_logits, R_targets_micro_float)
                    vf_loss = vf_loss_per_sample.mean()
                    vf_clipfrac = torch.tensor(0.0, device=vf_loss.device)
                    
                    
                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = micro_batch.batch["prompt_input_ids"].shape[0] / self.config.ppo_mini_batch_size
                        loss = vf_loss * loss_scale_factor
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                        loss = vf_loss * loss_scale_factor

                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "critic/vf_loss": vf_loss.detach().item() * loss_scale_factor,
                            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                            "critic/vpred_prob_mean": torch.sigmoid(vpreds_logits).mean().detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)
                
                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"critic/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

        self.critic_optimizer.zero_grad()
        return metrics