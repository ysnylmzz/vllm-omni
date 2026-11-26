# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm_omni.diffusion.models.minimax_h3.npu import audio_tconv as patch


@pytest.mark.parametrize("output_padding,requested", [(0, None), (1, None), (0, [80])])
def test_transpose_matches_stock(monkeypatch, output_padding, requested):
    monkeypatch.setattr(patch, "_tconv_enabled_on_input", lambda x: True)
    layer = nn.ConvTranspose1d(4, 6, 10, stride=5, padding=3, output_padding=output_padding).double()
    x = torch.randn(2, 4, 16, dtype=torch.float64)
    expected = layer(x, requested)
    assert patch.install_audio_vae_tconv_conv1d(layer) == (0, 1)
    torch.testing.assert_close(layer(x, requested), expected)


@pytest.mark.parametrize("taps", [9, 10])
def test_resampler_matches_stock(monkeypatch, taps):
    class UpSample1d(nn.Module):
        stride, pad, pad_left, pad_right, ratio = 5, 2, 3, 2, 5

        def forward(self, x):
            padded = F.pad(x, (2, 2), mode="replicate")
            return 5 * F.conv_transpose1d(padded, self.filter.expand(3, -1, -1), stride=5, groups=3)[..., 3:-2]

    monkeypatch.setattr(patch, "_tconv_enabled_on_input", lambda x: True)
    module = UpSample1d()
    module.register_buffer("filter", torch.randn(1, 1, taps, dtype=torch.float64))
    x = torch.randn(2, 3, 16, dtype=torch.float64)
    expected = module(x)
    assert patch.install_audio_vae_tconv_conv1d(module) == (1, 0)
    torch.testing.assert_close(module(x), expected)
