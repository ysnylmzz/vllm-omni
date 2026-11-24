# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

from vllm_omni.diffusion.cuda_graph import GraphedForward
from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner


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
            with torch.npu.graph(graph):
                outputs = body()
        return graph, outputs
