set -x

source uv_verl/bin/activate

train_path=data/dapo-math-17k_dedup.parquet
aime_test_path=data/offline_eval/math__aime_repeated_8x_240.parquet
math_test_path=data/offline_eval/math__math_500.parquet

train_files="['$train_path']"
test_files="['$aime_test_path', '$math_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=sequence_level_adv \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.shuffle=True \
    data.trust_remote_code=True \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.total_training_steps=200 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.max_model_len=20480 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    critic.enable=True \
    critic.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=False \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.use_dynamic_bsz=False \
    critic.ppo_mini_batch_size=32 \
    critic.ppo_micro_batch_size_per_gpu=1 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='SPPO' \
    trainer.experiment_name='DAPO-R1-7B-Seq_shuffle' \
    trainer.default_local_dir='./' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=2 \
    trainer.val_before_train=True \
    trainer.total_training_steps=200 $@