# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Speculative decoding implementation for dynamic inference.

This module implements speculative decoding with Multi-Token Prediction (MTP) heads.
MTP-based speculative decoding uses the auxiliary MTP layers trained during pretraining
to draft multiple tokens in parallel, which are then verified by the main model.

The workflow is:
1. Draft Phase: Use MTP heads to generate k draft tokens in one forward pass
2. Verify Phase: Run main model on draft tokens to get target probabilities
3. Accept/Reject: Accept tokens where target prob >= draft prob (modified rejection sampling)
4. Resample: For rejected tokens, sample from adjusted distribution
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from megatron.core.inference.sampling_params import SpeculativeConfig, SpeculativeMethod


@dataclass
class SpeculativeDecodingState:
    """State for tracking speculative decoding across requests.

    Attributes:
        draft_tokens: Tensor of shape [batch, num_speculative_tokens] with drafted tokens
        draft_probs: Tensor of shape [batch, num_speculative_tokens] with draft probabilities
        num_accepted: Tensor of shape [batch] tracking accepted tokens per request
        is_speculative_step: Whether current step is using speculative decoding
    """

    draft_tokens: Optional[Tensor] = None
    draft_probs: Optional[Tensor] = None
    num_accepted: Optional[Tensor] = None
    is_speculative_step: bool = False


class MTPSpeculativeDecoder:
    """Speculative decoder using Multi-Token Prediction heads.

    This class handles the draft and verify phases of speculative decoding
    using MTP heads that are part of the model architecture.

    Args:
        num_mtp_layers (int): Number of MTP layers in the model (determines max speculation depth)
        vocab_size (int): Vocabulary size for token generation
        device (torch.device): Device for tensor operations
    """

    def __init__(
        self,
        num_mtp_layers: int,
        vocab_size: int,
        device: torch.device = None,
    ):
        self.num_mtp_layers = num_mtp_layers
        self.vocab_size = vocab_size
        self.device = device or torch.cuda.current_device()

        # Validation
        if num_mtp_layers < 1:
            raise ValueError(
                f"MTP speculative decoding requires at least 1 MTP layer, got {num_mtp_layers}"
            )

    def validate_config(self, config: SpeculativeConfig) -> None:
        """Validate that speculative config is compatible with model.

        Args:
            config: The speculative configuration to validate

        Raises:
            ValueError: If configuration is incompatible with model
        """
        if config.method != SpeculativeMethod.MTP:
            raise ValueError(
                f"MTPSpeculativeDecoder only supports MTP method, got {config.method}"
            )

        if config.num_speculative_tokens > self.num_mtp_layers:
            raise ValueError(
                f"num_speculative_tokens ({config.num_speculative_tokens}) cannot exceed "
                f"num_mtp_layers ({self.num_mtp_layers})"
            )

    def draft_tokens_from_mtp_logits(
        self,
        mtp_logits_list: List[Tensor],
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate draft tokens from MTP layer logits.

        Args:
            mtp_logits_list: List of logits tensors from each MTP layer,
                each of shape [1, logits_seq_len, vocab_size]
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            generator: Random number generator for reproducibility

        Returns:
            draft_tokens: Tensor of shape [logits_seq_len, num_mtp_layers] with drafted tokens
            draft_probs: Tensor of shape [logits_seq_len, num_mtp_layers] with corresponding probabilities
        """
        logits_seq_len = mtp_logits_list[0].shape[1]
        num_drafts = len(mtp_logits_list)

        draft_tokens = torch.zeros(
            (logits_seq_len, num_drafts), dtype=torch.long, device=self.device
        )
        draft_probs = torch.zeros(
            (logits_seq_len, num_drafts), dtype=torch.float32, device=self.device
        )

        # Should maybe use _dyanmic_step_sample_logits here 
        for i, logits in enumerate(mtp_logits_list):
            # Get last token logits for each batch element
            # Shape: [batch, vocab_size]
            last_logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                last_logits = last_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                threshold = top_k_values[:, -1].unsqueeze(-1)
                last_logits = torch.where(
                    last_logits >= threshold, last_logits, torch.full_like(last_logits, float("-inf"))
                )

            # Apply top-p filtering
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                last_logits = last_logits.masked_fill(indices_to_remove, float("-inf"))

            # Compute probabilities
            probs = torch.softmax(last_logits, dim=-1)

            # Sample tokens
            if top_k == 1:
                # Greedy sampling
                tokens = torch.argmax(probs, dim=-1)
            else:
                tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

            # Store draft tokens and their probabilities
            draft_tokens[:, i] = tokens
            draft_probs[:, i] = probs.gather(1, tokens.unsqueeze(-1)).squeeze(-1)

        return draft_tokens, draft_probs

    def verify_and_accept(
        self,
        draft_tokens: Tensor,
        draft_probs: Tensor,
        target_logits: Tensor,
        acceptance_threshold: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Verify draft tokens against target model and determine acceptance.

        Uses modified rejection sampling: accept token if
        p_target(token) >= acceptance_threshold * p_draft(token)

        Args:
            draft_tokens: Tensor of shape [batch, num_drafts] with drafted tokens
            draft_probs: Tensor of shape [batch, num_drafts] with draft probabilities
            target_logits: Tensor of shape [batch, num_drafts + 1, vocab_size] from target model
            acceptance_threshold: Minimum probability ratio for acceptance
            generator: Random number generator

        Returns:
            accepted_tokens: Final accepted tokens including one newly sampled token
            num_accepted: Number of accepted draft tokens per batch element
            acceptance_mask: Boolean mask of accepted drafts
        """
        batch_size, num_drafts = draft_tokens.shape

        # Compute target probabilities for draft positions
        # target_logits has shape [batch, num_drafts + 1, vocab_size]
        target_probs = torch.softmax(target_logits, dim=-1)

        # Get target probabilities for the drafted tokens
        # Shape: [batch, num_drafts]
        target_probs_for_drafts = torch.zeros(
            (batch_size, num_drafts), dtype=torch.float32, device=self.device
        )
        for i in range(num_drafts):
            target_probs_for_drafts[:, i] = target_probs[:, i, :].gather(
                1, draft_tokens[:, i : i + 1]
            ).squeeze(-1)

        # Compute acceptance probability using modified rejection sampling
        # Accept if uniform < min(1, p_target / p_draft)
        acceptance_ratios = target_probs_for_drafts / (draft_probs + 1e-10)

        if acceptance_threshold > 0.0:
            # Apply threshold-based acceptance
            acceptance_mask = acceptance_ratios >= acceptance_threshold
        else:
            # Probabilistic acceptance (standard speculative decoding)
            uniform_samples = torch.rand(
                (batch_size, num_drafts), device=self.device, generator=generator
            )
            acceptance_mask = uniform_samples < torch.clamp(acceptance_ratios, max=1.0)

        # Find first rejection point for each batch element (tokens accepted sequentially)
        # After first rejection, all subsequent tokens are rejected
        cumulative_acceptance = acceptance_mask.cumprod(dim=1)
        num_accepted = cumulative_acceptance.sum(dim=1)

        # Build final accepted tokens
        # Start with the draft tokens that were accepted
        max_accepted = num_drafts + 1  # Can accept all drafts + sample one more
        accepted_tokens = torch.zeros(
            (batch_size, max_accepted), dtype=torch.long, device=self.device
        )

        for b in range(batch_size):
            n_acc = num_accepted[b].item()
            if n_acc > 0:
                accepted_tokens[b, :n_acc] = draft_tokens[b, :n_acc]

            # Sample the next token from target distribution at rejection point
            rejection_idx = min(n_acc, num_drafts)
            if rejection_idx < num_drafts and not acceptance_mask[b, rejection_idx]:
                # Sample from adjusted distribution: max(0, p_target - p_draft)
                adjusted_probs = torch.clamp(
                    target_probs[b, rejection_idx] - draft_probs[b, rejection_idx] * 
                    torch.nn.functional.one_hot(
                        draft_tokens[b, rejection_idx], num_classes=self.vocab_size
                    ).float(),
                    min=0.0,
                )
                adjusted_probs = adjusted_probs / (adjusted_probs.sum() + 1e-10)
                new_token = torch.multinomial(
                    adjusted_probs.unsqueeze(0), num_samples=1, generator=generator
                ).squeeze()
            else:
                # All drafts accepted, sample from target at last position
                new_token = torch.multinomial(
                    target_probs[b, rejection_idx].unsqueeze(0), 
                    num_samples=1, 
                    generator=generator
                ).squeeze()

            accepted_tokens[b, n_acc] = new_token

        # Update num_accepted to include the newly sampled token
        num_accepted = num_accepted + 1

        return accepted_tokens, num_accepted, cumulative_acceptance.bool()

# TODO : Should add parallel state here to check for pipeline parallelism ?
def create_speculative_decoder(
    model,
    speculative_config: SpeculativeConfig,
) -> Optional[MTPSpeculativeDecoder]:
    """Factory function to create appropriate speculative decoder.

    Args:
        model: The model to use for speculative decoding
        speculative_config: Configuration for speculative decoding

    Returns:
        MTPSpeculativeDecoder if MTP method and model supports it, else None
    """

    if speculative_config.method == SpeculativeMethod.MTP:
        # Check if model has MTP layers
        from megatron.core.utils import get_attr_wrapped_model

        mtp_num_layers = get_attr_wrapped_model(model, "config", None)
        if mtp_num_layers is not None:
            mtp_num_layers = getattr(mtp_num_layers, "mtp_num_layers", 0) or 0

        if mtp_num_layers == 0:
            raise ValueError(
                "MTP speculative decoding requires a model with MTP layers. "
                "Set mtp_num_layers > 0 in model config."
            )

        vocab_size = get_attr_wrapped_model(model, "vocab_size")
        assert vocab_size is not None and vocab_size > 0, "Vocabulary size must be greater than 0"

        decoder = MTPSpeculativeDecoder(
            num_mtp_layers=mtp_num_layers,
            vocab_size=vocab_size,
        )
        decoder.validate_config(speculative_config)
        return decoder

    raise ValueError(f"Unsupported speculative method: {speculative_config.method}")

