#include <cudaarithm.hpp>

extern "C"
struct TensorWrapper min(
        struct THCState *state, struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper dst)
{
    if (dst.isNull()) {
        cuda::GpuMat sr1 = src1.toGpuMat(), sr2 = src2.toGpuMat(), result;
        cuda::min(src1.toGpuMat(), src2.toGpuMat(), result);
        return TensorWrapper(result, state);
    } else {
        cuda::min(src1.toGpuMat(), src2.toGpuMat(), dst.toGpuMat());
        return dst;
    }
}