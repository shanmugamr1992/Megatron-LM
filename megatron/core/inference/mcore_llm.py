# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
MCore LLM API - A vLLM-like interface for Megatron-Core inference.

This module provides a simplified, user-friendly API for running inference
with Megatron-Core models, designed to be similar to the vLLM API for
easy adoption and migration.

Example Usage:
    ```python
    from megatron.core.inference import LLM, SamplingParams

    # Initialize the LLM
    llm = LLM(
        model='/path/to/checkpoint',
        tensor_parallel_size=1,
        max_model_len=2048,
    )

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=256,
    )

    # Generate completions
    prompts = ["The future of AI is", "Once upon a time"]
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
    ```
"""

import io
import os
import sys
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequestRecord
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.sampling_params import SamplingParams as MCoreSamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.transformer.module import MegatronModule


@dataclass
class CompletionOutput:
    """Output for a completion request, similar to vLLM's CompletionOutput.

    Attributes:
        index: The index of this output in the request (for n > 1).
        text: The generated text.
        token_ids: The token IDs of the generated text.
        cumulative_logprob: The cumulative log probability of the generated tokens.
        logprobs: The log probabilities of the tokens (if requested).
        finish_reason: The reason the generation stopped ('stop', 'length', etc.).
        stop_reason: The stop string or token that caused the generation to stop.
    """

    index: int = 0
    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    cumulative_logprob: Optional[float] = None
    logprobs: Optional[List[float]] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None


@dataclass
class RequestOutput:
    """Output for a request, similar to vLLM's RequestOutput.

    Attributes:
        request_id: The unique ID of the request.
        prompt: The input prompt string.
        prompt_token_ids: The token IDs of the prompt.
        prompt_logprobs: The log probabilities of prompt tokens (if requested).
        outputs: The list of completion outputs.
        finished: Whether the request is finished.
        metrics: Optional metrics about the request processing.
    """

    request_id: str
    prompt: Optional[str]
    prompt_token_ids: List[int]
    prompt_logprobs: Optional[List[float]] = None
    outputs: List[CompletionOutput] = field(default_factory=list)
    finished: bool = True
    metrics: Optional[Dict[str, Any]] = None


@dataclass
class SamplingParams:
    """Sampling parameters for text generation, similar to vLLM's SamplingParams.

    This class mirrors the vLLM SamplingParams API for easy migration.

    Attributes:
        n: Number of output sequences to return for the given prompt.
        temperature: Randomness of sampling. Lower = more deterministic.
        top_p: Cumulative probability for nucleus sampling.
        top_k: Number of top tokens to consider. -1 means all tokens.
        max_tokens: Maximum number of tokens to generate.
        stop: List of strings that stop generation when produced.
        stop_token_ids: List of token IDs that stop generation.
        logprobs: Number of log probabilities to return per output token.
        prompt_logprobs: Number of log probabilities to return per prompt token.
        skip_special_tokens: Whether to skip special tokens in output.
        include_stop_str_in_output: Whether to include stop string in output.
    """

    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: Optional[int] = 16
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    skip_special_tokens: bool = True
    include_stop_str_in_output: bool = False

    def to_mcore_sampling_params(
        self, termination_id: Optional[int] = None
    ) -> MCoreSamplingParams:
        """Convert to MCore SamplingParams.

        Args:
            termination_id: The token ID that signals end of generation.

        Returns:
            MCoreSamplingParams: The converted sampling parameters.
        """
        # Map top_k: vLLM uses -1 for no limit, mcore uses 0
        top_k = 0 if self.top_k == -1 else self.top_k

        return MCoreSamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=top_k,
            num_tokens_to_generate=self.max_tokens,
            return_log_probs=self.logprobs is not None and self.logprobs > 0,
            skip_prompt_log_probs=self.prompt_logprobs is None or self.prompt_logprobs == 0,
            top_n_logprobs=self.logprobs if self.logprobs else 0,
            termination_id=termination_id,
            stop_words=self.stop,
        )


class LLM:
    """An LLM class for generating text, designed to mirror the vLLM API.

    This class provides a simple interface for loading a Megatron-Core model
    and generating text completions. It wraps the complexity of the MCore
    inference system behind an API similar to vLLM.

    Args:
        model: Path to the model checkpoint or HuggingFace model name.
        tokenizer: Path to the tokenizer (if different from model path).
        tokenizer_mode: Tokenizer mode ('auto', 'huggingface', 'sentencepiece').
        trust_remote_code: Whether to trust remote code for tokenizer.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        pipeline_parallel_size: Number of stages for pipeline parallelism.
        dtype: Data type for model weights ('auto', 'float16', 'bfloat16', 'float32').
        max_model_len: Maximum sequence length the model can handle.
        gpu_memory_utilization: Fraction of GPU memory to use for KV cache.
        seed: Random seed for reproducibility.
        enable_cuda_graph: Whether to enable CUDA graphs for faster inference.
        disable_log_stats: Whether to disable logging statistics.
        max_num_seqs: Maximum number of sequences to process in parallel.
        max_num_batched_tokens: Maximum number of tokens to batch together.

    Example:
        ```python
        llm = LLM(model="/path/to/checkpoint", tensor_parallel_size=2)
        outputs = llm.generate(["Hello, world!"])
        ```
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        seed: int = 0,
        enable_cuda_graph: bool = True,
        disable_log_stats: bool = True,
        max_num_seqs: Optional[int] = 256,
        max_num_batched_tokens: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the LLM.

        Note: This requires Megatron to be initialized. If not already initialized,
        this will attempt to initialize it with appropriate defaults for inference.
        """
        self.model_path = model
        self.tokenizer_path = tokenizer or model
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.seed = seed
        self.enable_cuda_graph = enable_cuda_graph
        self.disable_log_stats = disable_log_stats
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens or 8192
        self.extra_kwargs = kwargs

        # These will be set during initialization
        self.model: Optional[MegatronModule] = None
        self.engine: Optional[DynamicInferenceEngine] = None
        self.tokenizer = None
        self.context: Optional[DynamicInferenceContext] = None
        self.controller: Optional[TextGenerationController] = None
        self._args = None
        self._initialized = False

        # Request counter for generating unique IDs
        self._request_counter = 0

        # Initialize the inference system
        self._initialize()

    def _get_dtype(self) -> torch.dtype:
        """Get the torch dtype from the string specification."""
        if self.dtype == "auto" or self.dtype == "bfloat16" or self.dtype == "bf16":
            return torch.bfloat16
        elif self.dtype == "float16" or self.dtype == "fp16":
            return torch.float16
        elif self.dtype == "float32" or self.dtype == "fp32":
            return torch.float32
        else:
            warnings.warn(f"Unknown dtype {self.dtype}, defaulting to bfloat16")
            return torch.bfloat16

    def _initialize(self) -> None:
        """Initialize the Megatron inference system.

        This method handles the initialization of Megatron, model loading,
        tokenizer setup, and inference engine creation.
        """
        import megatron
        from megatron.training import get_args, get_model as _get_model, initialize_megatron
        from megatron.training.checkpointing import load_checkpoint

        # Check if Megatron is already initialized
        try:
            args = get_args()
            self._args = args
        except AssertionError:
            # Megatron not initialized, need to initialize it
            # Build command line args from our parameters
            self._initialize_megatron()
            args = get_args()
            self._args = args

        # Build and load the model
        self._load_model()

        # Build tokenizer
        self._build_tokenizer()

        # Build inference context
        self._build_context()

        # Build inference engine
        self._build_engine()

        self._initialized = True

    def _initialize_megatron(self) -> None:
        """Initialize Megatron with inference-appropriate defaults.

        This sets up the necessary command line arguments for inference.
        """
        from megatron.training import initialize_megatron

        # Build sys.argv for Megatron initialization
        original_argv = sys.argv.copy()

        # Create minimal args for inference
        sys.argv = [sys.argv[0]]  # Keep program name

        # Add required arguments
        sys.argv.extend(["--load", self.model_path])
        sys.argv.extend(["--tensor-model-parallel-size", str(self.tensor_parallel_size)])
        sys.argv.extend(["--pipeline-model-parallel-size", str(self.pipeline_parallel_size)])

        # Add dtype
        if self.dtype in ("bfloat16", "bf16", "auto"):
            sys.argv.append("--bf16")
        elif self.dtype in ("float16", "fp16"):
            sys.argv.append("--fp16")

        # Add tokenizer args
        sys.argv.extend(["--tokenizer-type", "HuggingFaceTokenizer"])
        sys.argv.extend(["--tokenizer-model", self.tokenizer_path])

        # Enable use of checkpoint args to get model config
        sys.argv.append("--use-checkpoint-args")

        # Disable things not needed for inference
        sys.argv.append("--no-load-optim")
        sys.argv.append("--no-load-rng")

        # Set max sequence length if provided
        if self.max_model_len:
            sys.argv.extend(["--seq-length", str(self.max_model_len)])

        # Set micro batch size (needed by Megatron)
        sys.argv.extend(["--micro-batch-size", "1"])

        # Add any extra kwargs as command line args
        for key, value in self.extra_kwargs.items():
            arg_name = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    sys.argv.append(arg_name)
            else:
                sys.argv.extend([arg_name, str(value)])

        try:
            initialize_megatron(
                args_defaults={"no_load_rng": True, "no_load_optim": True},
            )
        finally:
            # Restore original argv
            sys.argv = original_argv

    def _load_model(self) -> None:
        """Load the model from checkpoint."""
        import megatron
        from megatron.training import get_args, get_model as _get_model
        from megatron.training.checkpointing import load_checkpoint

        # Import model builders
        # These need to be in the path - typically at repo root
        try:
            from model_provider import model_provider
            from gpt_builders import gpt_builder
        except ImportError:
            # Try importing from examples
            try:
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                from model_provider import model_provider
                from gpt_builders import gpt_builder
            except ImportError as e:
                raise ImportError(
                    "Could not import model_provider and gpt_builder. "
                    "Please ensure these are in your Python path. "
                    f"Original error: {e}"
                )

        args = get_args()

        # Add safe globals for deserialization
        torch.serialization.add_safe_globals([io.BytesIO])
        torch.serialization.add_safe_globals([megatron.core.rerun_state_machine.RerunState])
        torch.serialization.add_safe_globals([megatron.core.rerun_state_machine.RerunDiagnostic])

        # Build model
        model = _get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)

        # Load checkpoint
        args.exit_on_missing_checkpoint = True
        load_checkpoint(
            ddp_model=model,
            optimizer=None,
            opt_param_scheduler=None,
            strict=False,
        )

        # Get the first model (no virtual pipeline parallelism for inference)
        self.model = model[0]
        self.model.eval()

    def _build_tokenizer(self) -> None:
        """Build the tokenizer."""
        from megatron.training import get_args

        args = get_args()

        # Try to build tokenizer using mcore utilities
        try:
            from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer

            self.tokenizer = build_tokenizer(args)
        except Exception as e:
            # Fall back to legacy tokenizer
            try:
                from megatron.training import get_tokenizer

                self.tokenizer = get_tokenizer()
            except Exception:
                raise RuntimeError(f"Could not build tokenizer: {e}")

    def _build_context(self) -> None:
        """Build the dynamic inference context."""
        from megatron.training import get_args

        args = get_args()

        # Calculate buffer size from GPU memory utilization
        # This is a rough estimate - actual memory depends on model size
        total_gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        buffer_size_gb = total_gpu_memory_gb * self.gpu_memory_utilization * 0.5  # 50% for KV cache

        # Determine max sequence length
        max_seq_len = self.max_model_len or args.seq_length or 2048

        # Determine max tokens
        max_tokens = self.max_num_batched_tokens or 8192

        self.context = DynamicInferenceContext(
            params_dtype=args.params_dtype,
            num_layers=args.num_layers // args.pipeline_model_parallel_size,
            kv_channels=args.kv_channels,
            num_attention_heads=(
                args.num_query_groups if args.group_query_attention else args.num_attention_heads
            ),
            max_sequence_length=max_seq_len,
            num_cuda_graphs=16 if self.enable_cuda_graph else None,
            block_size_tokens=getattr(args, "inference_dynamic_batching_block_size", 256),
            buffer_size_gb=buffer_size_gb,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            materialize_only_last_token_logits=True,
            cache_mla_latent=(
                getattr(args, "multi_latent_attention", False)
                and getattr(args, "cache_mla_latents", False)
            ),
            kv_lora_rank=(
                getattr(args, "kv_lora_rank", None)
                if getattr(args, "multi_latent_attention", False)
                else None
            ),
            qk_pos_emb_head_dim=getattr(args, "qk_pos_emb_head_dim", None),
            use_cuda_graphs_for_non_decode_steps=not getattr(
                args, "decode_only_cuda_graphs", False
            ),
        )

    def _build_engine(self) -> None:
        """Build the inference engine."""
        from megatron.training import get_args

        args = get_args()

        # Wrap model in inference wrapper
        wrapped_model = GPTInferenceWrapper(self.model, args, self.context)
        wrapped_model.model_is_pipeline_parallel = False

        # Create text generation controller
        self.controller = TextGenerationController(wrapped_model, self.tokenizer)

        # Create inference engine
        self.engine = DynamicInferenceEngine(
            self.controller,
            self.context,
            enable_cuda_graph=self.enable_cuda_graph,
            random_seed=self.seed,
            enable_chunked_prefill=True,
        )

    def _get_next_request_id(self) -> int:
        """Get the next request ID."""
        request_id = self._request_counter
        self._request_counter += 1
        return request_id

    def _tokenize_prompt(self, prompt: Union[str, List[int], Dict]) -> torch.Tensor:
        """Tokenize a prompt.

        Args:
            prompt: Either a string, list of token IDs, or a dict with 'prompt_token_ids'.

        Returns:
            torch.Tensor: The token IDs as a tensor.
        """
        if isinstance(prompt, str):
            token_ids = self.controller.tokenize_prompt(prompt)
            return torch.tensor(token_ids, dtype=torch.long, device=torch.cuda.current_device())
        elif isinstance(prompt, dict):
            if "prompt_token_ids" in prompt:
                token_ids = prompt["prompt_token_ids"]
                return torch.tensor(
                    token_ids, dtype=torch.long, device=torch.cuda.current_device()
                )
            elif "prompt" in prompt:
                token_ids = self.controller.tokenize_prompt(prompt["prompt"])
                return torch.tensor(
                    token_ids, dtype=torch.long, device=torch.cuda.current_device()
                )
            else:
                raise ValueError("Dict prompt must have 'prompt' or 'prompt_token_ids' key")
        elif isinstance(prompt, (list, tuple)):
            return torch.tensor(prompt, dtype=torch.long, device=torch.cuda.current_device())
        elif isinstance(prompt, torch.Tensor):
            return prompt.to(dtype=torch.long, device=torch.cuda.current_device())
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    @torch.inference_mode()
    def generate(
        self,
        prompts: Union[str, List[str], List[Dict], List[List[int]]],
        sampling_params: Optional[SamplingParams] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """Generate completions for the given prompts.

        This method mirrors the vLLM generate() API.

        Args:
            prompts: The prompts to generate completions for. Can be:
                - A single string prompt
                - A list of string prompts
                - A list of dicts with 'prompt' or 'prompt_token_ids' keys
                - A list of token ID lists
            sampling_params: Sampling parameters. If None, uses defaults.
            use_tqdm: Whether to show a progress bar.

        Returns:
            List[RequestOutput]: The generated outputs for each prompt.
        """
        if not self._initialized:
            raise RuntimeError("LLM not initialized. Call _initialize() first.")

        # Normalize prompts to a list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Default sampling params
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Get termination ID
        termination_id = getattr(self.tokenizer, "eod", None)
        if sampling_params.stop_token_ids:
            termination_id = sampling_params.stop_token_ids[0]

        # Convert to mcore sampling params
        mcore_params = sampling_params.to_mcore_sampling_params(termination_id)

        # Reset the engine for a fresh batch
        self.engine.reset()

        # Add all requests to the engine
        request_id_map = {}  # Maps internal request_id to prompt info
        for idx, prompt in enumerate(prompts):
            request_id = self._get_next_request_id()

            # Tokenize the prompt
            prompt_tokens = self._tokenize_prompt(prompt)

            # Store prompt info for later
            if isinstance(prompt, str):
                prompt_text = prompt
                prompt_token_ids = prompt_tokens.tolist()
            elif isinstance(prompt, dict):
                prompt_text = prompt.get("prompt")
                prompt_token_ids = (
                    prompt.get("prompt_token_ids") or prompt_tokens.tolist()
                )
            else:
                prompt_text = None
                prompt_token_ids = (
                    prompt if isinstance(prompt, list) else prompt_tokens.tolist()
                )

            request_id_map[request_id] = {
                "prompt": prompt_text,
                "prompt_token_ids": prompt_token_ids,
                "idx": idx,
            }

            # Add request to engine
            self.engine.add_request(request_id, prompt_tokens, mcore_params)

        # Process all requests
        results: Dict[int, DynamicInferenceRequestRecord] = {}

        while self.engine.has_unfinished_requests():
            result = self.engine.step_modern()
            finished_request_records = result["finished_request_records"]

            for record in finished_request_records:
                finished_request = record.merge(self.tokenizer)
                results[finished_request.request_id] = finished_request

        # Build output in the same order as input prompts
        outputs = []
        for request_id, info in sorted(request_id_map.items(), key=lambda x: x[1]["idx"]):
            if request_id in results:
                finished = results[request_id]

                # Build CompletionOutput
                completion = CompletionOutput(
                    index=0,
                    text=finished.generated_text or "",
                    token_ids=list(finished.generated_tokens),
                    logprobs=(
                        finished.generated_log_probs if finished.generated_log_probs else None
                    ),
                    finish_reason="stop" if finished.succeeded() else "length",
                )

                # Build RequestOutput
                output = RequestOutput(
                    request_id=str(request_id),
                    prompt=info["prompt"],
                    prompt_token_ids=info["prompt_token_ids"],
                    prompt_logprobs=finished.prompt_log_probs,
                    outputs=[completion],
                    finished=True,
                )
                outputs.append(output)
            else:
                # Request not found - shouldn't happen normally
                warnings.warn(f"Request {request_id} not found in results")

        return outputs

    def __del__(self):
        """Cleanup when the LLM is deleted."""
        # Clean up CUDA resources
        if hasattr(self, "engine") and self.engine is not None:
            try:
                self.engine.reset()
            except Exception:
                pass

