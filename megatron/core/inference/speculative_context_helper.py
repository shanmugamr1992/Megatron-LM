# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Helper module for speculative decoding context management.

This module provides helper functions that work with DynamicInferenceContext
to manage speculative token storage and verification without modifying the
core context class.
"""

import logging
from typing import Optional, Tuple

import torch
from torch import Tensor


class SpeculativeContextState:
    """Manages speculative decoding state for a DynamicInferenceContext.
    
    This class tracks the state of speculative decoding including draft tokens,
    verification status, and token acceptance.
    
    Attributes:
        draft_tokens: Tensor of drafted tokens [batch_size, num_speculative_tokens]
        draft_probs: Tensor of draft probabilities [batch_size, num_speculative_tokens]
        num_speculative_tokens: Number of tokens being speculated
        is_verification_step: Whether we're in verification phase
        original_token_count: Token count before adding speculative tokens
        original_query_lengths: Query lengths before adding speculative tokens
    """
    
    def __init__(self):
        self.draft_tokens: Optional[Tensor] = None
        self.draft_probs: Optional[Tensor] = None
        self.num_speculative_tokens: int = 0
        self.is_verification_step: bool = False
        self.original_token_count: int = 0
        self.original_query_lengths: Optional[Tensor] = None
        
    def reset(self):
        """Reset speculative state after verification."""
        self.draft_tokens = None
        self.draft_probs = None
        self.num_speculative_tokens = 0
        self.is_verification_step = False
        self.original_token_count = 0
        self.original_query_lengths = None


def add_speculative_tokens_to_context(
    context,
    draft_tokens: Tensor,
    draft_probs: Optional[Tensor] = None,
    state: Optional[SpeculativeContextState] = None,
) -> Optional[SpeculativeContextState]:
    """Add speculative draft tokens to the context for verification.

    This function prepares the context for the verification forward pass by
    adding the draft tokens to the input sequence. The KV cache positions
    are updated to accommodate the draft tokens.

    Args:
        context: DynamicInferenceContext instance
        draft_tokens: Tensor of shape [batch_size, num_speculative_tokens]
            containing the drafted tokens for each active request.
        draft_probs: Optional tensor of draft probabilities
        state: Optional existing state to update, or create new if None

    Returns:
        SpeculativeContextState with saved state for later restoration
    """
    batch_size, num_speculative_tokens = draft_tokens.shape
    active_request_count = context.total_request_count - context.paused_request_count

    if batch_size != active_request_count:
        logging.warning(
            f"draft_tokens batch size ({batch_size}) must match "
            f"active_request_count ({active_request_count})"
        )
        return None

    # Create or update state
    if state is None:
        state = SpeculativeContextState()
    
    state.draft_tokens = draft_tokens
    state.draft_probs = draft_probs
    state.num_speculative_tokens = num_speculative_tokens
    state.original_token_count = context.active_token_count
    state.is_verification_step = True
    
    # Save original query lengths
    active_slice = slice(context.paused_request_count, context.total_request_count)
    state.original_query_lengths = context.request_query_lengths[active_slice].clone()

    # Calculate total tokens needed for verification pass
    total_new_tokens = active_request_count * num_speculative_tokens

    # Check if we have space
    # TODO : Can maybe add enough tokens ? Makes things unecessarily complex
    if context.active_token_count + total_new_tokens > context.max_tokens:
        logging.warning(
            f"Not enough token space for speculative tokens. "
            f"Active: {context.active_token_count}, New: {total_new_tokens}, Max: {context.max_tokens}"
        )
        state.reset()
        return None

    # Add draft tokens to input_ids
    token_offset = context.active_token_count
    
    for req_idx in range(active_request_count):
        global_req_idx = context.paused_request_count + req_idx

        for spec_idx in range(num_speculative_tokens):
            token_pos = token_offset + req_idx * num_speculative_tokens + spec_idx

            # Set the input token
            context.token_to_input_ids[token_pos] = draft_tokens[req_idx, spec_idx]

            # Set position ID (current sequence length + spec_idx + 1)
            current_seq_len = (
                context.request_kv_length_offsets[global_req_idx]
                + context.request_query_lengths[global_req_idx]
            )
            context.token_to_pos_ids[token_pos] = current_seq_len + spec_idx

            # Set request index
            context.token_to_request_idx[token_pos] = global_req_idx

            # Set position in request
            context.token_to_position_in_request[token_pos] = current_seq_len + spec_idx

            # Calculate block index and local position
            position_in_kv = current_seq_len + spec_idx
            block_idx_in_request = position_in_kv // context.block_size_tokens
            local_position = position_in_kv % context.block_size_tokens

            # Get the block ID (may need to allocate new blocks)
            if block_idx_in_request < context.request_kv_block_counts[global_req_idx]:
                block_id = context.request_to_kv_block_ids[global_req_idx, block_idx_in_request]
            else:
                # Need a new block - use dummy for now (speculative tokens)
                block_id = context.block_allocator.dummy_block_idx

            context.token_to_block_idx[token_pos] = block_id
            context.token_to_local_position_within_kv_block[token_pos] = local_position

    # Update active token count
    context.active_token_count += total_new_tokens

    # Update query lengths to include speculative tokens
    context.request_query_lengths[active_slice] += num_speculative_tokens

    return state


def accept_speculative_tokens_in_context(
    context,
    accepted_tokens: Tensor,
    num_accepted: Tensor,
    state: SpeculativeContextState,
) -> None:
    """Accept verified speculative tokens and update context state.

    After verification, this function commits the accepted tokens to the
    context and updates all relevant bookkeeping tensors.

    Args:
        context: DynamicInferenceContext instance
        accepted_tokens: Tensor of shape [batch_size, max_accepted]
            containing the tokens that were accepted for each request.
        num_accepted: Tensor of shape [batch_size] containing the number
            of accepted tokens for each request.
        state: SpeculativeContextState containing saved state
    """
    if not state.is_verification_step:
        return

    active_request_count = context.total_request_count - context.paused_request_count
    active_slice = slice(context.paused_request_count, context.total_request_count)

    # Restore original query lengths (will be updated properly below)
    if state.original_query_lengths is not None:
        context.request_query_lengths[active_slice] = state.original_query_lengths

    # Reset query lengths to 1 (decode mode)
    context.request_query_lengths[active_slice] = 1

    # Update KV length offsets to account for accepted tokens
    for req_idx in range(active_request_count):
        global_req_idx = context.paused_request_count + req_idx
        n_accepted = int(num_accepted[req_idx].item())

        # Advance the KV offset by the number of accepted tokens
        context.request_kv_length_offsets[global_req_idx] += n_accepted

        # Update position tracking
        new_position = context.request_kv_length_offsets[global_req_idx].item()

        # Update last block offset
        context.request_last_kv_block_offset[global_req_idx] = (
            new_position - 1
        ) % context.block_size_tokens

        # Check if we need new blocks
        new_block_count = (new_position + context.block_size_tokens - 1) // context.block_size_tokens
        current_block_count = int(context.request_kv_block_counts[global_req_idx].item())

        if new_block_count > current_block_count:
            # Need to allocate new blocks
            blocks_needed = new_block_count - current_block_count
            new_blocks = context.block_allocator.allocate_memory_blocks(blocks_needed)
            if new_blocks is not None and len(new_blocks) == blocks_needed:
                for i, block_id in enumerate(new_blocks):
                    context.request_to_kv_block_ids[
                        global_req_idx, current_block_count + i
                    ] = block_id
                context.request_kv_block_counts[global_req_idx] = new_block_count
                context.request_last_kv_block_id[global_req_idx] = new_blocks[-1]

    # Reset active token count to decode mode (1 token per request)
    context.active_token_count = active_request_count

    # Update token bookkeeping for next decode step
    for req_idx in range(active_request_count):
        global_req_idx = context.paused_request_count + req_idx
        n_accepted = int(num_accepted[req_idx].item())

        # Set the input token to the last accepted token
        context.token_to_input_ids[req_idx] = accepted_tokens[req_idx, n_accepted - 1]

        # Set position ID
        context.token_to_pos_ids[req_idx] = context.request_kv_length_offsets[global_req_idx]

        # Set request index
        context.token_to_request_idx[req_idx] = global_req_idx

        # Set position in request
        context.token_to_position_in_request[req_idx] = context.request_kv_length_offsets[
            global_req_idx
        ]

        # Set block index
        context.token_to_block_idx[req_idx] = context.request_last_kv_block_id[global_req_idx]

        # Set local position within block
        context.token_to_local_position_within_kv_block[req_idx] = (
            context.request_last_kv_block_offset[global_req_idx]
        )

    # Clear speculative state
    state.reset()


def restore_context_after_failed_speculation(
    context,
    state: SpeculativeContextState,
) -> None:
    """Restore context state if speculation failed or was aborted.

    Args:
        context: DynamicInferenceContext instance
        state: SpeculativeContextState containing saved state
    """
    if not state.is_verification_step:
        return

    active_slice = slice(context.paused_request_count, context.total_request_count)

    # Restore original token count
    context.active_token_count = state.original_token_count

    # Restore original query lengths
    if state.original_query_lengths is not None:
        context.request_query_lengths[active_slice] = state.original_query_lengths

    # Clear speculative state
    state.reset()

