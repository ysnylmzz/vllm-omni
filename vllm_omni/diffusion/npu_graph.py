# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_omni.diffusion.cuda_graph import GraphedForward
from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner


class _CaptureAttention(TorchDispatchMode):
    """Keep inference attention on the ACL capture stream, including checkpoint SDPA calls."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.ops.npu.npu_fusion_attention.default:
            # The default async mask path leaves a secondary stream unjoined even
            # for keep_prob=1. Preserve the caller's layout, scales and mask.
            options = {param.name: arg for param, arg in zip(func._schema.arguments, args)}
            options.update(kwargs, sync=True)
            return func(**options)
        if func is torch.ops.aten.scaled_dot_product_attention.default:
            return self._sdpa(*args, **kwargs)
        return func(*args, **kwargs)

    @staticmethod
    def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        # TaoMate uses unmasked BNSD attention and upper-left causal text GQA.
        # Keep native SDPA semantics for other inputs (e.g. additive masks).
        if (
            query.device.type != "npu"
            or query.dtype not in (torch.bfloat16, torch.float16)
            or query.ndim != 4
            or key.ndim != 4
            or key.shape != value.shape
            or query.shape[0] != key.shape[0]
            or query.shape[-1] != key.shape[-1]
            or (query.shape[1] != key.shape[1] and not enable_gqa)
            or attn_mask is not None
            or dropout_p != 0
        ):
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa
            )
        blocked = None
        if is_causal:
            blocked = torch.ones(query.size(2), key.size(2), device=query.device, dtype=torch.bool).triu_(1)
        return torch.ops.npu.npu_fusion_attention.default(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            query.size(1),
            "BNSD",
            atten_mask=blocked,
            scale=query.size(-1) ** -0.5 if scale is None else scale,
            sync=True,
        )[0]


class NPUGraphedForward(GraphedForward):
    """Capture fixed-shape forwards with independent Ascend graph pools."""

    graph_kind = "ACL"
    _sync_probe = False

    def _graphs_supported(self) -> bool:
        return self.device.type == "npu" and NPUExactGraphRunner.is_supported()

    @contextmanager
    def _warmup_stream(self) -> Iterator[None]:
        current = torch.npu.current_stream(self.device)
        side = torch.npu.Stream(device=self.device)
        side.wait_stream(current)
        try:
            with torch.npu.stream(side):
                yield
        finally:
            current.wait_stream(side)

    def _capture_graph(self, body: Callable[[], Any]) -> tuple[Any, Any]:
        with torch.npu.device(self.device):
            graph = torch.npu.NPUGraph()
            # The outer VAE autocast scope may already hold FP16 weight copies.
            # Record the casts so replay never reads a released autocast buffer.
            with (
                torch.autocast(
                    "npu",
                    enabled=torch.is_autocast_enabled("npu"),
                    dtype=torch.get_autocast_dtype("npu"),
                    cache_enabled=False,
                ),
                torch.npu.graph(graph),
                _CaptureAttention(),
            ):
                outputs = body()
        return graph, outputs
