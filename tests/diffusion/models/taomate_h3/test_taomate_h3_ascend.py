# SPDX-License-Identifier: Apache-2.0
"""Hardware checks for the experimental Ascend 950 paths."""

import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend device required")


@torch.inference_mode()
def test_graph_replay_updates_inputs_and_preserves_previous_output():
    from vllm_omni.diffusion.npu_graph import NPUGraphedForward

    graph = NPUGraphedForward(lambda x: x + 2, device=torch.device("npu"))
    x = torch.ones(8, device="npu")
    first = graph(x=x)
    second = graph(x=x * 3)
    torch.testing.assert_close(first, x + 2)
    torch.testing.assert_close(second, x + 4)
    assert graph.captures == 1 and graph.replays == 2


@torch.inference_mode()
def test_per_channel_linear_matches_dequantized_operands():
    import torch_npu

    from vllm_omni.quantization.npu_fp8_per_channel import NPUPerChannelFP8Linear

    layer = torch.nn.Linear(128, 64, device="npu", dtype=torch.bfloat16)
    method = NPUPerChannelFP8Linear()
    method.process_weights_after_loading(layer)
    inputs = torch.randn(2, 8, 128, device="npu", dtype=torch.bfloat16)
    actual = method.apply(layer, inputs, layer.bias)
    quant, scale = torch_npu.npu_dynamic_quant(inputs.flatten(0, 1), dst_type=torch.float8_e4m3fn)
    tokens = quant.cpu().float() * scale.cpu().reshape(-1, 1)
    weights = layer.weight.cpu().float() * layer.weight_scale.cpu().reshape(-1, 1)
    expected = (tokens @ weights.T).bfloat16() + layer.bias.cpu()
    torch.testing.assert_close(actual.cpu(), expected.reshape_as(actual), atol=0.02, rtol=0.02)
