# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
import torch_npu

import vllm_ascend.ops.fused_moe.routed_experts as routed_experts_module
from vllm_ascend.weight_switch import WeightSwitchConfig, WeightSwitchGatherSpec, WeightSwitchMixin


@pytest.mark.parametrize("gather_dim", [0, 1])
@pytest.mark.parametrize("weight_format", [2, 29])
def test_weight_switch_preserves_local_format_and_gathers_nd(gather_dim, weight_format, monkeypatch):
    monkeypatch.setattr(torch.npu.config, "allow_internal_format", True)
    reference = torch.randn(8, 256, 256, dtype=torch.bfloat16)
    weight = torch_npu.npu_format_cast(reference.to("npu"), weight_format)

    class LinearMethod(WeightSwitchMixin):
        supports_weight_switch = True
        weight_switch_gather_specs = (WeightSwitchGatherSpec("weight", gather_dim=gather_dim),)

    state = LinearMethod().enable_weight_switch(
        SimpleNamespace(weight=weight),
        WeightSwitchConfig(group=object(), world_size=2, rank=0, shard_axis="input"),
    )
    part = state.gather_parts["weight"]

    assert part.local_tensor.data_ptr() == weight.data_ptr()
    assert torch_npu.get_npu_format(part.local_tensor) == weight_format
    assert torch_npu.get_npu_format(part.gather_input) != 29
    assert part.gather_input.is_contiguous()
    torch.testing.assert_close(part.local_tensor.cpu(), reference, rtol=0, atol=0)
    torch.testing.assert_close(part.gather_input.cpu(), reference.movedim(gather_dim, 0), rtol=0, atol=0)


@pytest.mark.parametrize("nz_mode,dtype", [(2, torch.bfloat16), (1, torch.bfloat16), (2, torch.float16)])
def test_moe_nz_loading_repacks_only_bf16_nz2(nz_mode, dtype, monkeypatch):
    monkeypatch.setattr(torch.npu.config, "allow_internal_format", True)
    cls = routed_experts_module.AscendUnquantizedFusedMoEMethod
    method = cls.__new__(cls)
    monkeypatch.setattr(method, "_maybe_pad_weight", lambda weight: weight)
    # Isolate the Ascend loader from upstream weight preprocessing.
    upstream_base = routed_experts_module.UnquantizedFusedMoEMethod.__mro__[1]
    monkeypatch.setattr(upstream_base, "process_weights_after_loading", lambda self, layer: None)
    monkeypatch.setattr(
        routed_experts_module,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_fused_mc2=False, weight_nz_mode=nz_mode),
    )
    layer = torch.nn.Module()
    originals = {}
    for name in ("w13_weight", "w2_weight"):
        originals[name] = torch.randn(8, 256, 256, dtype=dtype)
        layer.register_parameter(name, torch.nn.Parameter(originals[name].to("npu"), requires_grad=False))

    before_cast = []
    replace_parameter = routed_experts_module.replace_parameter

    def record_replace(layer, name, data):
        replace_parameter(layer, name, data)
        before_cast.append(getattr(layer, name).data)

    cast_count = 0

    def checked_cast(data):
        nonlocal cast_count
        source = before_cast[cast_count]
        cast_count += 1
        assert (data.data_ptr() != source.data_ptr()) == (nz_mode == 2 and dtype == torch.bfloat16)
        assert data.storage_offset() == 0
        torch.testing.assert_close(data.cpu(), source.cpu(), rtol=0, atol=0)
        return torch_npu.npu_format_cast(data, 29) if nz_mode == 2 else data

    monkeypatch.setattr(routed_experts_module, "replace_parameter", record_replace)
    monkeypatch.setattr(routed_experts_module, "maybe_trans_nz", checked_cast)
    method.process_weights_after_loading(layer)

    assert cast_count == 2
    for name, original in originals.items():
        torch.testing.assert_close(getattr(layer, name).cpu(), original.transpose(1, 2), rtol=0, atol=0)
