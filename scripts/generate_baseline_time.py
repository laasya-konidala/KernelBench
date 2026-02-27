import argparse
import torch
import numpy as np
from kernelbench.dataset import construct_kernelbench_dataset, fetch_ref_arch_from_dataset
from kernelbench.timing import measure_ref_program_time
from kernelbench.utils import read_file
import os
import json
import sys
from tqdm import tqdm

"""
Generate baseline time for KernelBench
This profiles the wall clock time for each KernelBench reference problem

You can find a list of pre-generated baseline time in /results/timing/
But we recommend you run this script to generate the baseline time for your own hardware configurations

Using various configurations
- torch (Eager)

Torch Compile with various modes
https://pytorch.org/docs/main/generated/torch.compile.html
- torch.compile: backend="inductor", mode="default" (this is usually what happens when you do torch.compile(model))
- torch.compile: backend="inductor", mode="reduce-overhead" 
- torch.compile: backend="inductor", mode="max-autotune"
- torch.compile: backend="inductor", mode="max-autotune-no-cudagraphs"

In addition to default Torch Compile backend, you can always use other or your custom backends
https://pytorch.org/docs/stable/torch.compiler.html
- torch.compile: backend="cudagraphs" (CUDA graphs with AOT Autograd)
"""

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")

TIMING_DIR = os.path.join(REPO_TOP_PATH, "results", "timing")



def record_baseline_times(use_torch_compile: bool = False, 
                          torch_compile_backend: str="inductor", 
                          torch_compile_options: str="default",
                          file_name: str="baseline_time.json",
                          precision: str="fp32"):
    """
    Generate baseline time for KernelBench, 
    configure profiler options for PyTorch
    save to specified file
    """
    device = torch.device("cuda:0")
    json_results = {}
    
    for level in [1, 2, 3]:
        dataset = construct_kernelbench_dataset(level)
        json_results[f"level{level}"] = {}

        num_problems = len(dataset)
        for problem_id in tqdm(dataset.get_problem_ids()):
            ref_arch_path, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(dataset, problem_id)
            runtime_stats = measure_ref_program_time(
                ref_arch_name=ref_arch_name,
                ref_arch_src=ref_arch_src,
                use_torch_compile=use_torch_compile,
                torch_compile_backend=torch_compile_backend,
                torch_compile_options=torch_compile_options,
                device=device,
                verbose=False, # do not print 
                precision=precision,
            )
            json_results[f"level{level}"][ref_arch_name] = runtime_stats

    save_path = os.path.join(TIMING_DIR, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(json_results, f)
    return json_results

def test_measure_particular_program(level_num: int, problem_id: int):
    """
    Test measure_program_time on a particular program
    """
    device = torch.device("cuda:0")

    dataset = construct_kernelbench_dataset(level_num)

    ref_arch_path, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(dataset, problem_id)

    exec_stats = measure_ref_program_time(
        ref_arch_name=ref_arch_name,
        ref_arch_src=ref_arch_src,
        use_torch_compile=True,
        torch_compile_backend="inductor",
        torch_compile_options="default",
        device=device,
        verbose=False, 
        precision="bf16"
    )

    print(f"Execution time for {ref_arch_name}: {exec_stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate baseline timing for your hardware (works on NVIDIA and AMD/ROCm).")
    parser.add_argument("--hardware", "-H", type=str, default="H100_PCIe_LambdaLabs",
                        help="Hardware name; results saved to results/timing/<hardware>/ (e.g. MI350_matx for AMD)")
    parser.add_argument("--precision", "-p", type=str, default="fp32", choices=["fp32", "fp16", "bf16"],
                        help="Precision for baseline (match your eval precision, default fp32)")
    parser.add_argument("--eager-only", action="store_true",
                        help="Only record torch eager baseline (skip torch.compile; use on AMD if inductor/cudagraphs fail)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")
    args = parser.parse_args()
    hardware_name = args.hardware

    if not args.yes:
        input(f"You are about to start recording baseline time for {hardware_name}, press Enter to continue...")
    if os.path.exists(os.path.join(TIMING_DIR, hardware_name)) and not args.yes:
        input(f"Directory {hardware_name} already exists. Overwrite? Enter to continue...")

    # 1. Record Torch Eager (works on AMD/ROCm)
    record_baseline_times(use_torch_compile=False,
                          torch_compile_backend=None,
                          torch_compile_options=None,
                          file_name=f"{hardware_name}/baseline_time_torch.json",
                          precision=args.precision)

    if args.eager_only:
        print("Eager baseline saved. Done (--eager-only).")
        sys.exit(0)

    # 2. Record Torch Compile using Inductor (usually works on AMD)
    for torch_compile_mode in ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]:
        try:
            record_baseline_times(use_torch_compile=True,
                                  torch_compile_backend="inductor",
                                  torch_compile_options=torch_compile_mode,
                                  file_name=f"{hardware_name}/baseline_time_torch_compile_inductor_{torch_compile_mode}.json",
                                  precision=args.precision)
        except Exception as e:
            print(f"Warning: inductor {torch_compile_mode} failed: {e}. Continuing.")

    # 3. Record Torch Compile using cudagraphs (NVIDIA-only; may fail on AMD)
    try:
        record_baseline_times(use_torch_compile=True,
                              torch_compile_backend="cudagraphs",
                              torch_compile_options=None,
                              file_name=f"{hardware_name}/baseline_time_torch_compile_cudagraphs.json",
                              precision=args.precision)
    except Exception as e:
        print(f"Warning: cudagraphs failed (expected on AMD): {e}. Skipped.")
    



    # Random debuging
    # get_torch_compile_triton(2, 12)
    # record_baseline_times()

    # run_profile(2, 43)
    # get_time(2, 43, torch_compile=False)
    # get_time(2, 43, torch_compile=True)



