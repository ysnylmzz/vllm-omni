# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online per-output-channel E4M3 weights and per-token activations on Ascend 950."""

from fnmatch import fnmatchcase

import torch
import torch_npu
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import replace_parameter


class NPUPerChannelFP8Config(QuantizationConfig):
    def __init__(self, ignore: list[str] | None = None):
        super().__init__()
        self.ignore = ignore or []

    @classmethod
    def get_name(cls):
        return "npu_fp8_per_channel"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @classmethod
    def get_config_filenames(cls):
        return []

    @classmethod
    def from_config(cls, config):
        return cls(ignore=config.get("ignore"))

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            if any(fnmatchcase(prefix, pattern) for pattern in self.ignore):
                return UnquantizedLinearMethod()
            return NPUPerChannelFP8Linear()
        return None


class NPUPerChannelFP8Linear(LinearMethodBase):
    def create_weights(self, layer, *args, **kwargs):
        # Load checkpoint rows with the same shard layout and loader as BF16.
        UnquantizedLinearMethod().create_weights(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer):
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        weight, scale = torch_npu.npu_dynamic_quant(layer.weight.contiguous(), dst_type=torch.float8_e4m3fn)
        # Keep [out, in], including for square layers inspected by the LoRA manager.
        replace_parameter(layer, "weight", weight)
        replace_parameter(layer, "weight_scale", scale.reshape(-1))
        layer._already_called_process_weights_after_loading = True

    def apply(self, layer, x, bias=None):
        tokens = x.reshape(-1, x.shape[-1]).contiguous()
        quantized, scale = torch_npu.npu_dynamic_quant(tokens, dst_type=torch.float8_e4m3fn)
        result = torch_npu.npu_quant_matmul(
            quantized,
            layer.weight.t(),
            layer.weight_scale,
            pertoken_scale=scale.reshape(-1),
            output_dtype=x.dtype,
        )
        if bias is not None:
            result = result + bias
        return result.reshape(*x.shape[:-1], layer.weight.shape[0])
