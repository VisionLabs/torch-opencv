#include <cudaarithm.hpp>

extern "C"
struct TensorWrapper min(
        struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper dst)
{
    auto g = src1.toGpuMat();
    cv::Mat m;
    cuda::multiply(g, cv::Scalar_<float>(20, 0, 0, 0), g);
    g.download(m);
    std::cout << m << std::endl;
}