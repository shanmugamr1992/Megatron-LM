# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class SpeculativeMethod(Enum):
    """Enum for speculative decoding methods."""

    MTP = "mtp"  # Multi-Token Prediction head based speculative decoding


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding.

    Speculative decoding accelerates inference by drafting multiple tokens
    using a faster method and then verifying them with the main model.

    Args:
        method (SpeculativeMethod): The speculative decoding method to use.
            Currently only MTP (Multi-Token Prediction) is supported.
        num_speculative_tokens (int): Number of tokens to speculate ahead.
            This should be <= the number of MTP layers in the model.
            Defaults to 1.
        acceptance_threshold (float): Minimum probability ratio for accepting
            a speculated token. Tokens with p_target/p_draft >= threshold are
            accepted. Range [0, 1]. Defaults to 0.0 (accept all valid tokens).
    """

    method: SpeculativeMethod = SpeculativeMethod.MTP
    num_speculative_tokens: int = 1
    acceptance_threshold: float = 0.0

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_speculative_tokens < 1:
            raise ValueError(
                f"num_speculative_tokens must be >= 1, got {self.num_speculative_tokens}"
            )
        if not 0.0 <= self.acceptance_threshold <= 1.0:
            raise ValueError(
                f"acceptance_threshold must be in [0, 1], got {self.acceptance_threshold}"
            )
        if not isinstance(self.method, SpeculativeMethod):
            if isinstance(self.method, str):
                self.method = SpeculativeMethod(self.method.lower())
            else:
                raise ValueError(f"Invalid speculative method: {self.method}")

    def serialize(self) -> dict:
        """Return a dictionary that is msgpack-serializable."""
        return {
            "method": self.method.value,
            "num_speculative_tokens": self.num_speculative_tokens,
            "acceptance_threshold": self.acceptance_threshold,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "SpeculativeConfig":
        """Construct SpeculativeConfig from a msgpack-compatible dictionary."""
        return cls(
            method=SpeculativeMethod(data["method"]),
            num_speculative_tokens=data["num_speculative_tokens"],
            acceptance_threshold=data.get("acceptance_threshold", 0.0),
        )


@dataclass
class SamplingParams:
    """Inference parameters sent along with the prompts.
    This class contains request-level attributes that control the sampling techniques used when
    generating text. This is distinct from megatron.core.inference.contexts.BaseInferenceContext,
        which is sets model-level
    inference attributes such as the maximum sequence length, and contains the KV cache.

    For an explanation of these parameters refer to this blog
    https://ivibudh.medium.com/a-guide-to-controlling-llm-model-output-exploring-top-k-top-p-and-
    temperature-parameters-ed6a31313910
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0
    return_log_probs: bool = False
    skip_prompt_log_probs: bool = False
    return_segments: bool = False  # Whether to return individually detokenized tokens
    num_tokens_to_generate: int = 30
    num_tokens_total: Optional[int] = None  # Cannot set both this and num_tokens_to_generate
    termination_id: Optional[int] = None
    top_n_logprobs: int = 0
    return_prompt_top_n_logprobs: bool = False  # Deprecated field for backwards compatibility
    add_BOS: bool = False
    stop_words: Optional[List[str]] = (
        None  # List of strings that will stop generation when produced
    )
    speculative_config: Optional[SpeculativeConfig] = None  # Config for speculative decoding

    def __post_init__(self):
        """Ensure backward compatibility for return_prompt_top_n_logprobs.

        Sets return_prompt_top_n_logprobs based on skip_prompt_log_probs and top_n_logprobs:
        - return_prompt_top_n_logprobs = not skip_prompt_log_probs and top_n_logprobs > 0
        """
        self._sync_prompt_logprobs_fields()

    def _sync_prompt_logprobs_fields(self):
        """Synchronize return_prompt_top_n_logprobs with skip_prompt_log_probs."""

        if self.return_prompt_top_n_logprobs:
            warnings.warn(
                "return_prompt_top_n_logprobs is deprecated, use skip_prompt_log_probs instead",
                DeprecationWarning,
            )
            assert (
                self.skip_prompt_log_probs
            ), "return_prompt_top_n_logprobs requires skip_prompt_log_probs to be False"
        if self.top_n_logprobs > 0:
            self.return_prompt_top_n_logprobs = not self.skip_prompt_log_probs
        else:
            self.return_prompt_top_n_logprobs = False

    def add_attributes(self, attribute_value_pair: dict):
        """Utility to add more attributes to sampling params

        Use this method to pass in a custom dictionary to add more sampling parameter attributes.
        c = SamplingParams
        c.add_attributes({'min_length':4, 'eod_id':153})

        Args:
            attribute_value_pair (dict): A dictionary containing attributes as the key names and
            their values as the values.
        """
        for key, value in attribute_value_pair.items():
            setattr(self, key, value)

        # Synchronize fields after setting attributes
        self._sync_prompt_logprobs_fields()

    def serialize(self) -> dict:
        """Return a dictionary that is msgpack-serializable."""
        result = self.__dict__.copy()
        # Handle speculative_config serialization
        if self.speculative_config is not None:
            result["speculative_config"] = self.speculative_config.serialize()
        return result

    @classmethod
    def deserialize(cls, data: dict) -> "SamplingParams":
        """Construct SamplingParams from a msgpack-compatible dictionary."""
        # Handle speculative_config deserialization
        data = data.copy()
        if "speculative_config" in data and data["speculative_config"] is not None:
            data["speculative_config"] = SpeculativeConfig.deserialize(data["speculative_config"])
        obj = cls()
        obj.add_attributes(data)
        return obj
