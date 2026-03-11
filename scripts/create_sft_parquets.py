"""
Create train/val parquets from KernelBench synthetic_data.jsonl for verl SFT.

Each parquet has a "messages" column in the format verl MultiTurnSFTDataset expects:
  [{"role": "user", "content": "<prompt>"}, {"role": "assistant", "content": "<kernel code>"}]

Usage:
  python scripts/create_sft_parquets.py --run_dirs hip_level1_20260308_171300 hip_level2_20260308_195930 hip_level3_20260308_232812 --out_dir RL
  # Creates RL/train_level1.parquet, RL/val_level1.parquet, etc.
"""
import argparse
import json
import os

import pandas as pd

# One-shot HIP prompt template (must match prompts.toml / what you send to the model)
HIP_ONE_SHOT_PROMPT = """You write custom AMD HIP kernels to replace the pytorch operators in the given architecture to get speedups.

You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom AMD HIP kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.

Here's an example to show you the syntax of inline embedding custom AMD HIP kernels in PyTorch:

Example:

Input architecture:

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []

Optimized with AMD HIP kernels:

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Must include this line so PyTorch could use HIP compiler for AMD GPUs
os.environ["CXX"] = "hipcc"

elementwise_add_cpp_source = '''
#include <hip/hip_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_hip(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
'''

elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    functions=["elementwise_add_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_hip(a, b)

You are given the following architecture:


{ref_arch_src}

Note: The kernels should be optimized for FP32 (32-bit floating point) precision.

Optimize the architecture named Model with custom AMD HIP kernels. Name the optimized architecture ModelNew.

Return ONLY a single markdown code block containing the full ModelNew implementation.

Strict requirements:
	• Output exactly one markdown code block.
	• No text before the code block.
	• No text after the code block.
	• No explanations, reasoning, or commentary.
	• No additional code blocks.
	• No testing or example code.
	• The code must be complete, functional, and compile successfully.
	• The code block must contain the entire ModelNew implementation."""


def load_data_level(run_dir: str):
    path = os.path.join(run_dir, "synthetic_data.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No synthetic_data.jsonl in {run_dir}")
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def create_parquet(run_dir: str, level_name: str, out_dir: str, train_ratio: float = 0.7):
    data = load_data_level(run_dir)
    correct_kernels = [k for k in data if k.get("correct") is True]

    rows = []
    for correct in correct_kernels:
        user_content = HIP_ONE_SHOT_PROMPT.replace("{ref_arch_src}", correct["ref_arch_src"])
        assistant_content = correct["kernel_src"]
        # verl MultiTurnSFTDataset expects "messages" = list of {role, content}
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        rows.append({
            "messages": messages,
            "data_source": "kernelbench",
            "prompt": [{"role": "user", "content": user_content}],
            "response": assistant_content,
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": correct["ref_arch_src"]},
            "extra_info": {
                "level": correct.get("level"),
                "problem_id": correct.get("problem_id"),
                "problem_name": correct.get("problem_name", ""),
            },
        })

    df = pd.DataFrame(rows)
    n = len(df)
    if n == 0:
        print(f"[{level_name}] No correct kernels in {run_dir}; skipping.")
        return
    split = int(n * train_ratio)
    train_df = df.iloc[:split]
    val_df = df.iloc[split:]

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, f"train_{level_name}.parquet")
    val_path = os.path.join(out_dir, f"val_{level_name}.parquet")
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    print(f"[{level_name}] train={len(train_df)} val={len(val_df)} -> {train_path}, {val_path}")


def main():
    parser = argparse.ArgumentParser(description="Create SFT parquets for verl from synthetic_data.jsonl")
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        default=["hip_level1_20260308_171300", "hip_level2_20260308_195930", "hip_level3_20260308_232812"],
        help="Directories containing synthetic_data.jsonl",
    )
    parser.add_argument(
        "--out_dir",
        default="RL",
        help="Output directory for train_levelN.parquet and val_levelN.parquet",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Fraction of data for training (default 0.7)",
    )
    args = parser.parse_args()

    level_names = [f"level{i+1}" for i in range(len(args.run_dirs))]
    for run_dir, level_name in zip(args.run_dirs, level_names):
        create_parquet(run_dir, level_name, args.out_dir, args.train_ratio)


if __name__ == "__main__":
    main()
