# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Simple MCore Inference Example using the vLLM-like API.

This script demonstrates how to use the new simplified MCore inference API
that mirrors the vLLM API for easy migration and adoption.

Usage:
    torchrun --nproc-per-node 1 mcore_simple_inference.py \
        --model /path/to/checkpoint \
        --tokenizer Qwen/Qwen2.5-1.5B

The API is designed to be similar to vLLM for easy switching:

    # vLLM:
    from vllm import LLM, SamplingParams
    llm = LLM(model='Qwen/Qwen2.5-1.5B')
    outputs = llm.generate(prompts, SamplingParams(max_tokens=256))

    # MCore (NEW):
    from megatron.core.inference import LLM, SamplingParams
    llm = LLM(model='/path/to/checkpoint')
    outputs = llm.generate(prompts, SamplingParams(max_tokens=256))
"""

import argparse
import time
import torch


def print_cuda_memory_usage(stage: str = ""):
    """Print CUDA memory usage statistics."""
    if not torch.cuda.is_available():
        return
    
    i = 0
    allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
    max_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)  # GB
    max_reserved = torch.cuda.max_memory_reserved(i) / (1024 ** 3)    # GB
    
    print(f"[{stage}] GPU {i} Memory:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Max Allocated: {max_allocated:.2f} GB")
    print(f"  Max Reserved:  {max_reserved:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Simple MCore Inference")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer (defaults to model path)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p (nucleus) sampling")
    parser.add_argument("--num-prompts", type=int, default=4,
                        help="Number of prompts to generate")
    args = parser.parse_args()

    # Import the new simplified API
    from megatron.core.inference import LLM, SamplingParams

    # Initialize the LLM - this is similar to vLLM's API
    print("=" * 80)
    print("Initializing MCore LLM...")
    print("=" * 80)
    
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_cuda_graph=True,
        max_model_len=2048,
    )
    
    print_cuda_memory_usage("After Model Load")

    # Define sampling parameters - same API as vLLM
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Create some prompts
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a galaxy far away",
        "The best way to learn programming is",
        "In the year 2050, humans will",
    ][:args.num_prompts]

    # Reset peak memory stats before inference
    torch.cuda.reset_peak_memory_stats()
    print_cuda_memory_usage("Before Inference")

    # Generate - same API as vLLM!
    print("=" * 80)
    print("Generating completions...")
    print("=" * 80)
    
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()
    latency = end_time - start_time

    print_cuda_memory_usage("After Inference")

    # Print results
    print("=" * 80)
    print("Results:")
    print("=" * 80)
    
    total_prompt_tokens = 0
    total_generated_tokens = 0
    
    for output in outputs:
        prompt = output.prompt or "[Token IDs provided]"
        generated_text = output.outputs[0].text
        
        total_prompt_tokens += len(output.prompt_token_ids)
        total_generated_tokens += len(output.outputs[0].token_ids)
        
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"Generated ({len(output.outputs[0].token_ids)} tokens): {generated_text[:100]}...")

    # Print statistics
    total_tokens = total_prompt_tokens + total_generated_tokens
    
    print("\n" + "-" * 80)
    print(f"Total time: {latency:.2f} seconds")
    print(f"Throughput: {len(outputs) / latency:.2f} requests/sec")
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total generated tokens: {total_generated_tokens}")
    print(f"Total tokens: {total_tokens}")
    print(f"Token throughput: {total_tokens / latency:.2f} tokens/sec")
    print(f"Generation throughput: {total_generated_tokens / latency:.2f} tokens/sec")

    # Print peak memory usage
    if torch.cuda.is_available():
        i = 0
        peak_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved(i) / (1024 ** 3)
        print(f"GPU {i} Peak Memory - Allocated: {peak_allocated:.2f} GB, Reserved: {peak_reserved:.2f} GB")
    
    print("-" * 80)


if __name__ == "__main__":
    main()

