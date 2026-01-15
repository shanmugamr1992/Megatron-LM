# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Utility functions for speculative decoding with MTP heads.

This module provides utilities to enable and configure speculative decoding
for dynamic inference. Speculative decoding uses Multi-Token Prediction (MTP)
heads to draft multiple tokens in parallel, which are then verified by the
main model in a single forward pass.

Example usage:

    from megatron.core.inference.speculative_utils import (
        create_speculative_config,
        setup_speculative_decoding,
    )

    # Create speculative config
    speculative_config = create_speculative_config(
        method='mtp',
        num_speculative_tokens=3,
        acceptance_threshold=0.0,
    )

    # Setup speculative decoding on the controller
    setup_speculative_decoding(controller, speculative_config)

    # Use the speculative generation method
    result = await controller.async_generate_output_tokens_dynamic_batch_speculative()
"""

import argparse
from typing import Optional

from megatron.core.inference.sampling_params import SamplingParams, SpeculativeConfig, SpeculativeMethod


def create_speculative_config(
    method: str = 'mtp',
    num_speculative_tokens: int = 1,
    acceptance_threshold: float = 0.0,
) -> SpeculativeConfig:
    """Create a SpeculativeConfig for speculative decoding.

    Args:
        method: The speculative decoding method. Currently only 'mtp' is supported.
        num_speculative_tokens: Number of tokens to speculate ahead. Should be <= 
            the number of MTP layers in the model.
        acceptance_threshold: Minimum probability ratio for accepting speculated tokens.
            Range [0, 1]. Default 0.0 means probabilistic acceptance.

    Returns:
        SpeculativeConfig: Configuration for speculative decoding.

    Raises:
        ValueError: If method is not supported or parameters are invalid.
    """
    if method.lower() != 'mtp':
        raise ValueError(f"Unsupported speculative method: {method}. Only 'mtp' is supported.")

    return SpeculativeConfig(
        method=SpeculativeMethod.MTP,
        num_speculative_tokens=num_speculative_tokens,
        acceptance_threshold=acceptance_threshold,
    )


def setup_speculative_decoding(
    controller,
    speculative_config: SpeculativeConfig,
) -> None:
    """Setup speculative decoding on a TextGenerationController.

    This function initializes the speculative decoder and related state
    on the controller, enabling the use of 
    `async_generate_output_tokens_dynamic_batch_speculative()`.

    Args:
        controller: The TextGenerationController instance to configure.
        speculative_config: Configuration for speculative decoding.

    Raises:
        ValueError: If the model doesn't support MTP-based speculative decoding.
    """
    controller.initialize_speculative_decoding(speculative_config)


def add_speculative_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add speculative decoding arguments to an argument parser.

    This function can be used to add speculative decoding command-line arguments
    to a custom inference script.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        The modified argument parser.

    Example:
        parser = argparse.ArgumentParser()
        parser = add_speculative_args_to_parser(parser)
        args = parser.parse_args()

        if args.speculative_decoding_method:
            config = create_speculative_config(
                method=args.speculative_decoding_method,
                num_speculative_tokens=args.speculative_num_tokens,
                acceptance_threshold=args.speculative_acceptance_threshold,
            )
    """
    group = parser.add_argument_group(title='speculative decoding')

    group.add_argument(
        '--speculative-decoding-method',
        type=str,
        default=None,
        choices=['mtp', None],
        help='Speculative decoding method. Currently only "mtp" (Multi-Token Prediction) '
        'is supported. MTP uses the model\'s MTP layers to draft tokens for verification.',
    )
    group.add_argument(
        '--speculative-num-tokens',
        type=int,
        default=1,
        help='Number of tokens to speculate ahead during speculative decoding. '
        'This should be <= the number of MTP layers in the model. Default is 1.',
    )
    group.add_argument(
        '--speculative-acceptance-threshold',
        type=float,
        default=0.0,
        help='Minimum probability ratio for accepting a speculated token. '
        'Tokens with p_target/p_draft >= threshold are accepted. '
        'Range [0, 1]. Default is 0.0 (probabilistic acceptance).',
    )

    return parser


def get_speculative_config_from_args(args) -> Optional[SpeculativeConfig]:
    """Create SpeculativeConfig from parsed arguments.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        SpeculativeConfig if speculative decoding is enabled, None otherwise.
    """
    method = getattr(args, 'speculative_decoding_method', None)
    if method is None:
        return None

    return create_speculative_config(
        method=method,
        num_speculative_tokens=getattr(args, 'speculative_num_tokens', 1),
        acceptance_threshold=getattr(args, 'speculative_acceptance_threshold', 0.0),
    )
