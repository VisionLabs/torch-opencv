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

extern "C"
struct TensorWrapper getGaborKernel(int ksize_rows, int ksize_cols, double sigma, double theta,
                                    double lambd, double gamma, double psi, int ktype)
{
    cv::Mat retval = cv::getGaborKernel(cv::Size(ksize_rows, ksize_cols), sigma,
                                        theta, lambd, gamma, psi, ktype);
    return TensorWrapper(retval);
}