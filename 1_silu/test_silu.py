import pytest
import torch
import torch_npu
from custom_vector_ops import silu_ascendc

device = "npu:0"   # must have a valid device ID
dtype = torch.float16  # operator code assumes half precision

def ref_silu(x):
    return x * torch.sigmoid(x)


@pytest.mark.parametrize("num_elements", [10, 256, 4011, 18000])
@torch.inference_mode()
def test_silu_1d_input(num_elements) -> None:
    x = torch.empty(num_elements, dtype=dtype).uniform_(1, 4).to(device)
    ref_out = ref_silu(x)
    npu_out = silu_ascendc(x)
    assert torch.allclose(npu_out, ref_out, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("M", [83, 256])
@pytest.mark.parametrize("N", [77, 213, 512, 2048])
@torch.inference_mode()
def test_silu_2d_input(M, N) -> None:
    x = (torch.empty(M, N, dtype=dtype)
        .uniform_(1, 4).to(device)
        .contiguous()  # so that can be treated as 1D array in physical memory
      )
    ref_out = ref_silu(x)
    npu_out = silu_ascendc(x)
    assert torch.allclose(npu_out, ref_out, atol=1e-3, rtol=1e-2)
