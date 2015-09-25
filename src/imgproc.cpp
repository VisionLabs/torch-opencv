#include <imgproc.hpp>

extern "C"
struct TensorWrapper getGaussianKernel(int ksize, double sigma, int ktype) {
    cv::Mat retval = cv::getGaussianKernel(ksize, sigma, ktype);
    return TensorWrapper(retval);
}

extern "C"
struct MultipleTensorWrapper getDerivKernels(
                     int dx, int dy, int ksize,
                     bool normalize, int ktype) {

    std::vector<cv::Mat> output(2);

    cv::getDerivKernels(
            output[0], output[1],
            dx, dy, ksize, normalize, ktype);

    return MultipleTensorWrapper(output);
}