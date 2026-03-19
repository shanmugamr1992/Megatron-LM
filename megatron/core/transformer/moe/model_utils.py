# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch

from megatron.core import tensor_parallel


def initialize_moe_layer_metadata(layers: Iterable[object]) -> int:
    """Assign dense-to-sparse layer indices and propagate the total MoE layer count."""
    moe_layers = [layer for layer in layers if getattr(layer, "is_moe_layer", False)]

    for moe_layer_idx, layer in enumerate(moe_layers):
        assigned = layer.set_moe_layer_number(moe_layer_idx)
        if assigned is False:
            raise RuntimeError(
                f"MoE layer numbering should happen exactly once, but layer {layer!r} "
                f"rejected index {moe_layer_idx}."
            )

    num_moe_layers = len(moe_layers)
    for layer in moe_layers:
        layer.set_num_moe_layers(num_moe_layers)

    return num_moe_layers


def sequence_parallelize_extra_input_like_tensor(
    extra_input: torch.Tensor | None,
    *,
    batch_size: int,
    seq_length: int,
    reduce_scatter_embeddings: bool,
    tp_group: Any,
) -> torch.Tensor | None:
    """Adapted from VocabParallelEmbedding.forward sequence parallel case."""
    if extra_input is None:
        return None

    assert extra_input.dim() >= 2
    leading_dims = tuple(extra_input.shape[:2])
    assert leading_dims == (batch_size, seq_length), (
        f"Leading dimensions do not match the batch-first layout: {leading_dims}"
    )

    if reduce_scatter_embeddings:
        extra_input = (
            extra_input.transpose(0, 1).contiguous()
        )
        extra_input = tensor_parallel.scatter_to_sequence_parallel_region(
            extra_input, group=tp_group
        ).clone()

    return extra_input
