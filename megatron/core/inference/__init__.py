# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Megatron-Core Inference Module.

This module provides inference capabilities for Megatron-Core models, including:
- A vLLM-like API for easy model serving (LLM class)
- Dynamic batching inference engine
- Static batching inference engine
- Sampling utilities and parameters

Quick Start:
    ```python
    from megatron.core.inference import LLM, SamplingParams

    llm = LLM(model='/path/to/checkpoint')
    outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=100))
    ```
"""

from megatron.core.inference.mcore_llm import (
    CompletionOutput,
    LLM,
    RequestOutput,
    SamplingParams,
)
from megatron.core.inference.sampling_params import SamplingParams as MCoreSamplingParams

__all__ = [
    "LLM",
    "SamplingParams",
    "MCoreSamplingParams",
    "RequestOutput",
    "CompletionOutput",
]
