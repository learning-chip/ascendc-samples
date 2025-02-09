#include "kernel_operator.h"

#define UB_BLOCK_SIZE 32  // UB block size
#define NUM_PER_VECTOR 128  // vector unit: 8 datablock * 32 Bytes (per_block) / 2 Bytes (FP16)
#define DIV_ROUNDUP(x,y) (((x)+(y)-1) / (y))

extern "C" __global__ __aicore__ void silu_kernel(GM_ADDR x, GM_ADDR y, int32_t num_elements)
{
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<half, AscendC::MaskMode::NORMAL>((uint64_t)-1, (uint64_t)-1);

    // split elements evenly across multi cores
    int32_t num_cores = AscendC::GetBlockNum();
    int32_t elements_per_core = DIV_ROUNDUP(num_elements, num_cores);
    int32_t offset_this_core = elements_per_core * AscendC::GetBlockIdx();  // array offset
    int32_t elements_to_process = elements_per_core;  // remaining elements to process in current core
    if (offset_this_core + elements_to_process > num_elements) { // detect out-of-range numbers
        elements_to_process = num_elements - offset_this_core;  // usually only affects last core
    }
    if (elements_to_process <= 0) {  // skip empty work
        return;
    }

    AscendC::GlobalTensor<half> x_gm;
    AscendC::GlobalTensor<half> y_gm;
    x_gm.SetGlobalBuffer((__gm__ half *)x + offset_this_core);
    y_gm.SetGlobalBuffer((__gm__ half *)y + offset_this_core);

    // Double buffer (2 for input, 2 for output)
    uint64_t UB_ALLOC_BYTES = 48 * 1024;  // Bytes, UB has 192KB in total
    // allocate local tensor on UB without using Tque methods
    // See TBuf: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/ascendcopapi/atlasascendc_api_07_0161.html
    // using VECIN or VECCALC or VECOUT does not matter, as we are not using Tque method
    using VecBuf_t = AscendC::TBuf<AscendC::QuePosition::VECCALC>;
    VecBuf_t tbuf_0;
    VecBuf_t tbuf_1;
    VecBuf_t tbuf_2;
    VecBuf_t tbuf_3;

    AscendC::TPipe pipe;
    pipe.InitBuffer(tbuf_0, UB_ALLOC_BYTES);
    pipe.InitBuffer(tbuf_1, UB_ALLOC_BYTES);
    pipe.InitBuffer(tbuf_2, UB_ALLOC_BYTES);
    pipe.InitBuffer(tbuf_3, UB_ALLOC_BYTES);

    AscendC::LocalTensor<half> local_buf_0 = tbuf_0.Get<half>();
    AscendC::LocalTensor<half> local_buf_1 = tbuf_1.Get<half>();
    AscendC::LocalTensor<half> local_buf_2 = tbuf_2.Get<half>();
    AscendC::LocalTensor<half> local_buf_3 = tbuf_3.Get<half>();

    // allowed elements per iteration
    int32_t elements_per_tile = UB_ALLOC_BYTES / sizeof(half);

    // Config for vector operation
    // See AscendC doc about repeatTimes、dataBlockStride、repeatStride
    // https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0022.html
    uint64_t vec_repeat = DIV_ROUNDUP(elements_per_tile, NUM_PER_VECTOR);  // must be smaller than 255
    auto unary_params = AscendC::UnaryRepeatParams(1, 1, 8, 8);
    auto binary_params = AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8);

    // Config for copy operation
    // See AscendC doc about DataCopyParams
    // https://www.hiascend.com/document/detail/zh/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0102.html
    uint16_t copy_block_len = DIV_ROUNDUP(UB_ALLOC_BYTES, UB_BLOCK_SIZE);
    auto copy_params = AscendC::DataCopyParams(1, copy_block_len, 0, 0);

    int32_t x_offset = 0;
    int32_t y_offset = 0;
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t num_processed = 0, ping = 1;
        num_processed < elements_to_process;
        num_processed += elements_per_tile)
    {
        auto x_buf = ping ? local_buf_0 : local_buf_2;
        auto cal_buf = ping ? local_buf_1 : local_buf_3;
        auto event_id = ping ? EVENT_ID0 : EVENT_ID1;

        int32_t num_to_process = elements_per_tile;
        if (num_processed + num_to_process > elements_to_process) {
            num_to_process = elements_to_process - num_processed;
            copy_block_len = DIV_ROUNDUP(num_to_process * 2, UB_BLOCK_SIZE);  // `* 2` converts numbers to Bytes
            copy_params = AscendC::DataCopyParams(1, copy_block_len, 0, 0);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        AscendC::DataCopy(x_buf, x_gm[x_offset], copy_params);
        x_offset += num_to_process;  // to read next tile
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);

        AscendC::Muls<half, false>(
            cal_buf, x_buf, (half)-1.0, (uint64_t)0, vec_repeat, unary_params
            );  // y = -x
        AscendC::Exp<half, false>(
            cal_buf, cal_buf, (uint64_t)0, vec_repeat, unary_params
            );  // y = e^-x
        AscendC::Adds<half, false>(
            cal_buf, cal_buf, (half)1.0, (uint64_t)0, vec_repeat, unary_params
            );  // y = 1 + e^-x
        AscendC::Div<half, false>(
            cal_buf, x_buf, cal_buf, (uint64_t)0, vec_repeat, binary_params
            );  // y = x / (1 + e^-x)

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);
        AscendC::DataCopy(y_gm[y_offset], cal_buf, copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);

        y_offset += num_to_process;  // to write next tile
        ping = 1 - ping;
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::PipeBarrier<PIPE_ALL>();
}

void launch_silu(
    uint32_t block_dim, void *l2ctrl, void *stream,
    uint8_t *x, uint8_t *y, int32_t num_elements)
{
    silu_kernel<<<block_dim, l2ctrl, stream>>>(x, y, num_elements);
}
