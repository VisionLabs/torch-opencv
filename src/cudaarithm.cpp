#include <cudaarithm.hpp>

extern "C"
struct TensorWrapper min(
        struct THCState *state, struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper dst)
{
    if (dst.isNull()) {
        cuda::GpuMat result;

        std::cout << "here" << std::endl;
        cuda::min(src1.toGpuMat(), src2.toGpuMat(), result);

        cv::Mat temp;
        result.download(temp);
        std::cout << temp << std::endl;

        return TensorWrapper(result, state);
    } else {
        cuda::min(src1.toGpuMat(), src2.toGpuMat(), dst.toGpuMat());
        return dst;
    }
}