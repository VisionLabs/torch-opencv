#include <imgproc.hpp>

extern "C"
struct TensorWrapper getGaussianKernel(int ksize, double sigma, int ktype) {
    return TensorWrapper(
            cv::getGaussianKernel(ksize, sigma, ktype));
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
    return TensorWrapper(
            cv::getGaborKernel(
                    cv::Size(ksize_rows, ksize_cols), sigma, theta, lambd, gamma, psi, ktype));
}

extern "C"
struct TensorWrapper getStructuringElement(int shape, int ksize_rows, int ksize_cols,
                                           int anchor_x, int anchor_y)
{
    return TensorWrapper(
            cv::getStructuringElement(
                    shape, cv::Size(ksize_rows, ksize_cols), cv::Point(anchor_x, anchor_y)));
}

extern "C"
struct TensorWrapper medianBlur(struct TensorWrapper src, struct TensorWrapper dst, int ksize)  {
    if (dst.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::medianBlur(src.toMat(), retval, ksize);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::medianBlur(source, source, ksize);
    } else {
        cv::medianBlur(src.toMat(), dst.toMat(), ksize);
    }
    return dst;
}

extern "C"
struct TensorWrapper GaussianBlur(struct TensorWrapper src, struct TensorWrapper dst,
                                  int ksize_x, int ksize_y, double sigmaX,
                                  double sigmaY, int borderType)
{
    if (dst.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::GaussianBlur(
                src.toMat(), retval, cv::Size(ksize_x, ksize_y), sigmaX, sigmaY, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::GaussianBlur(
                source, source, cv::Size(ksize_x, ksize_y), sigmaX, sigmaY, borderType);
    } else {
        cv::GaussianBlur(
                src.toMat(), dst.toMat(), cv::Size(ksize_x, ksize_y), sigmaX, sigmaY, borderType);
    }
    return dst;
}