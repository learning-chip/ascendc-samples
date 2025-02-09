#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "acl/acl.h"

extern void launch_silu(
    uint32_t block_dim, void *l2ctrl, void *stream,
    uint8_t *x, uint8_t *y, int32_t num_elements);

at::Tensor silu_ascendc(at::Tensor x)
{
    int32_t num_elements = x.numel();
    at::Tensor y = torch::empty(x.sizes(), x.options());
    uint8_t *x_ptr = reinterpret_cast<uint8_t *>(x.storage().data_ptr().get());
    uint8_t *y_ptr = reinterpret_cast<uint8_t *>(y.storage().data_ptr().get());

    uint32_t block_dim = 20 * 2;  // hard-coded to 910B4 vector core number

    int device_id;
    aclrtGetDevice(&device_id);
    auto npu_stream = c10_npu::getCurrentNPUStream(device_id);
    auto acl_stream = npu_stream.stream();
    launch_silu(block_dim, nullptr, acl_stream, x_ptr, y_ptr, num_elements);
    aclrtSynchronizeStream(acl_stream);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_ascendc", &silu_ascendc, "silu_ascendc");
}
