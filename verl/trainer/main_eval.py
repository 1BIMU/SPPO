import json
from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm
from transformers import AutoTokenizer
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local
from verl.utils.reward_score import default_compute_score

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False


tokenizer_cache = {}

def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError(f"k = {k} cannot be greater than n = {n}")
    if n - c < k:
        return 1.0
    prod = 1.0
    for j in range(n - c + 1, n + 1):
        prod *= 1.0 - k / j
    return 1.0 - prod


@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data, extra_info, model_path):
    global tokenizer_cache
    if model_path not in tokenizer_cache:
        print(f"Worker process caching tokenizer for model: {model_path}")
        tokenizer_cache[model_path] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = tokenizer_cache[model_path]
    ground_truth = reward_data.get("ground_truth")
    score_lst = []
    length_lst = []
    for r in response_lst:
        score_data = reward_fn(data_source, r, ground_truth, extra_info)
        score_lst.append(score_data["score"])
        if isinstance(r, str):
            length_lst.append(len(tokenizer.encode(r)))
        else:
            length_lst.append(0)
    return (data_source, score_lst, length_lst)


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
    model_path = config.model.path

    if POLARS_AVAILABLE and "livecodebench" in local_path:
        dataset = pl.read_parquet(local_path)
    else:
        dataset = pd.read_parquet(local_path)

    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]
    extra_info_data = dataset.get("extra_info")

    total = len(dataset)

    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    data_source_reward = defaultdict(list)
    compute_score = get_custom_reward_fn(config) or default_compute_score

    remote_tasks = [
        process_item.remote(
            compute_score,
            data_sources[i],
            responses[i],
            reward_model_data[i],
            extra_info_data[i] if extra_info_data is not None else {},
            model_path
        ) for i in range(total)
    ]

    is_polars_series = POLARS_AVAILABLE and isinstance(responses, pl.Series)
    if isinstance(responses, pd.Series) or is_polars_series:
        max_k = len(responses.to_list()[-1])
    else:
        max_k = len(responses.tolist()[-1])

    candidate_ks = [2**i for i in range(int(np.log2(max_k)) + 1) if 2**i <= max_k]
    pass_k_stat = {k: 0 for k in candidate_ks}
    avg_pass = 0
    total_lengths = []

    with tqdm(total=total, desc="Evaluating responses") as pbar:
        while len(remote_tasks) > 0:
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, score_lst, length_lst = ray.get(result_id)
                pass_count = sum(1 for score in score_lst if score == 1)
                avg_score = float(np.mean(score_lst))
                avg_pass += avg_score
                data_source_reward[data_source].append(avg_score)
                total_lengths.extend(length_lst)
                for k_val in candidate_ks:
                    pass_k_stat[k_val] += unbiased_pass_at_k(max_k, pass_count, k_val)
                pbar.update(1)

    avg_length = float(np.mean(total_lengths)) if total_lengths else 0
    metric_output_path = config.data.path.replace(".parquet", "_metric.json")
    metric_data = {
        **{f"pass@{k_val}": pass_k_stat[k_val] / total * 100.0 for k_val in candidate_ks},
        f"pass@1_(avg{max_k})": avg_pass / total * 100.0,
        "average_response_length": avg_length,
    }
    with open(metric_output_path, "w") as f:
        json.dump(metric_data, f, indent=4)

    print("\n--- Overall Metrics ---")
    print(json.dumps(metric_data, indent=4))

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f"test_score(avg@k)/{data_source}"] = float(np.mean(rewards))
    print("\n--- Per-Datasource Scores ---")
    print(json.dumps(metric_dict, indent=4))


if __name__ == "__main__":
    main()