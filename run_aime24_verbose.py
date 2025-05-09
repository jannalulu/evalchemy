#!/usr/bin/env python
"""
Verbose AIME24 evaluation runner script.
This script temporarily replaces the standard AIME24 benchmark with the verbose version,
runs the evaluation, and then restores the original file.
"""

import os
import shutil
import sys
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run AIME24 evaluation with real-time output")
parser.add_argument("--model", default="hf", help="Model type (e.g., hf)")
parser.add_argument("--model_args", default="pretrained=Qwen/Qwen3-4B,trust_remote_code=True", 
                    help="Model arguments")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
parser.add_argument("--limit", default=1, type=int, help="Number of examples to evaluate")
parser.add_argument("--output_path", default="results", help="Output directory")
args = parser.parse_args()

# Paths to files
orig_file = "eval/chat_benchmarks/AIME24/eval_instruct.py"
verbose_file = "eval/chat_benchmarks/AIME24/eval_instruct_verbose.py"
backup_file = "eval/chat_benchmarks/AIME24/eval_instruct.py.bak"

# Create backup of original file
print("Creating backup of original evaluation file...")
shutil.copy(orig_file, backup_file)

try:
    # Replace original with verbose version
    print("Installing verbose version...")
    shutil.copy(verbose_file, orig_file)
    
    # Run the evaluation
    print("\n" + "=" * 80)
    print("RUNNING VERBOSE AIME24 EVALUATION")
    print("=" * 80 + "\n")
    
    cmd = (f"python -m eval.eval "
           f"--model {args.model} "
           f"--tasks AIME24 "
           f"--model_args '{args.model_args}' "
           f"--batch_size {args.batch_size} "
           f"--output_path {args.output_path} "
           f"--verbosity DEBUG "
           f"--log_samples "
           #f"--debug"
           )
    
    print(f"Executing: {cmd}")
    os.system(cmd)
    
finally:
    # Restore original file
    print("\nRestoring original evaluation file...")
    shutil.copy(backup_file, orig_file)
    os.remove(backup_file)
    print("Done!")