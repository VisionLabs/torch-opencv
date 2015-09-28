#include <imgproc.hpp>

extern "C"
struct TensorWrapper getGaussianKernel(int ksize, double sigma, int ktype)
{
    return TensorWrapper(
            cv::getGaussianKernel(ksize, sigma, ktype));
}

extern "C"
struct MultipleTensorWrapper getDerivKernels(
        int dx, int dy, int ksize,
        bool normalize, int ktype)
{
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
struct TensorWrapper medianBlur(struct TensorWrapper src, struct TensorWrapper dst, int ksize)
{
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

extern "C"
struct TensorWrapper bilateralFilter(struct TensorWrapper src, struct TensorWrapper dst, int d,
                                     double sigmaColor, double sigmaSpace,
                                     int borderType)
{
    if (dst.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::bilateralFilter(
                src.toMat(), retval, d, sigmaColor, sigmaSpace, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::bilateralFilter(
                source, source, d, sigmaColor, sigmaSpace, borderType);
    } else {
        cv::bilateralFilter(
                src.toMat(), dst.toMat(), d, sigmaColor, sigmaSpace, borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper boxFilter(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int ksize_x, int ksize_y, int anchor_x, int anchor_y,
        bool normalize, int borderType)
{
    if (dst.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::boxFilter(
                src.toMat(), retval, ddepth, cv::Size(ksize_x, ksize_y),
                cv::Point(anchor_x, anchor_y), normalize, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::boxFilter(
                source, source, ddepth, cv::Size(ksize_x, ksize_y),
                cv::Point(anchor_x, anchor_y), normalize, borderType);
    } else {
        cv::boxFilter(
                src.toMat(), dst.toMat(), ddepth, cv::Size(ksize_x, ksize_y),
                cv::Point(anchor_x, anchor_y), normalize, borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper sqrBoxFilter(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int ksize_x, int ksize_y, int anchor_x, int anchor_y,
        bool normalize, int borderType)
{
    if (dst.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::sqrBoxFilter(src.toMat(), retval, ddepth,
                cv::Point(ksize_x, ksize_y), cv::Point(anchor_x, anchor_y),
                normalize, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place 
        cv::Mat source = src.toMat();
        cv::sqrBoxFilter(source, source, ddepth, cv::Point(ksize_x, ksize_y), cv::Point(anchor_x, anchor_y),
                normalize, borderType);
    } else {
        cv::sqrBoxFilter(src.toMat(), dst.toMat(), ddepth, cv::Point(ksize_x, ksize_y), cv::Point(anchor_x, anchor_y),
                normalize, borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper blur(
        struct TensorWrapper src, struct TensorWrapper dst,
        int ksize_x, int ksize_y, int anchor_x, int anchor_y, int borderType)
{
    if (dst.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::blur(src.toMat(), retval, cv::Point(ksize_x, ksize_y), cv::Point(anchor_x, anchor_y), borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place 
        cv::Mat source = src.toMat();
        cv::blur(source, source, cv::Point(ksize_x, ksize_y), cv::Point(anchor_x, anchor_y), borderType);
    } else {
        cv::blur(src.toMat(), dst.toMat(), cv::Point(ksize_x, ksize_y), cv::Point(anchor_x, anchor_y), borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper filter2D(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct TensorWrapper kernel, int anchor_x, int anchor_y,
        double delta, int borderType)
{
    if (dst.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::filter2D(src.toMat(), retval, ddepth,
        kernel.toMat(), cv::Point(anchor_x, anchor_y),
        delta, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place 
        cv::Mat source = src.toMat();
        cv::filter2D(source, source, ddepth,
        kernel.toMat(), cv::Point(anchor_x, anchor_y),
        delta, borderType);
    } else {
        cv::filter2D(src.toMat(), dst.toMat(), ddepth,
        kernel.toMat(), cv::Point(anchor_x, anchor_y),
        delta, borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper sepFilter2D(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct TensorWrapper kernelX,struct TensorWrapper kernelY,
        int anchor_x, int anchor_y, double delta, int borderType)
{
    if (dst.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::sepFilter2D(src.toMat(), retval, ddepth,
        kernelX.toMat(),kernelY.toMat(),
        cv::Point(anchor_x, anchor_y), delta, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place 
        cv::Mat source = src.toMat();
        cv::sepFilter2D(source, source, ddepth,
        kernelX.toMat(),kernelY.toMat(),
        cv::Point(anchor_x, anchor_y), delta, borderType);
    } else {
        cv::sepFilter2D(src.toMat(), dst.toMat(), ddepth,
        kernelX.toMat(),kernelY.toMat(),
        cv::Point(anchor_x, anchor_y), delta, borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper Sobel(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int dx, int dy, int ksize, double scale, double delta, int borderType)
{
    if (dst.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::Sobel(src.toMat(), retval, ddepth,
                dx, dy, ksize, scale, delta, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place 
        cv::Mat source = src.toMat();
        cv::Sobel(source, source, ddepth,
                dx, dy, ksize, scale, delta, borderType);
    } else {
        cv::Sobel(src.toMat(), dst.toMat(), ddepth,
                dx, dy, ksize, scale, delta, borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper Scharr(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int dx, int dy, double scale, double delta, int borderType)
{
    if (dst.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::Scharr(src.toMat(), retval, ddepth,
                dx, dy, scale, delta, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place 
        cv::Mat source = src.toMat();
        cv::Scharr(source, source, ddepth,
                dx, dy, scale, delta, borderType);
    } else {
        cv::Scharr(src.toMat(), dst.toMat(), ddepth,
                dx, dy, scale, delta, borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper Laplacian(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int ksize, double scale, double delta, int borderType)
{
    if (dst.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::Laplacian(src.toMat(), retval, ddepth,
                ksize, scale, delta, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place 
        cv::Mat source = src.toMat();
        cv::Laplacian(source, source, ddepth,
                ksize, scale, delta, borderType);
    } else {
        cv::Laplacian(src.toMat(), dst.toMat(), ddepth,
                ksize, scale, delta, borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper Canny(
        struct TensorWrapper image, struct TensorWrapper edges,
        double threshold1, double threshold2, int apertureSize, bool L2gradient)
{
    if (edges.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::Canny(image.toMat(), retval, threshold1, threshold2, apertureSize, L2gradient);
        return TensorWrapper(retval);
    } else if (edges.tensorPtr == image.tensorPtr) {
        // in-place 
        cv::Mat source = image.toMat();
        cv::Canny(source, source, threshold1, threshold2, apertureSize, L2gradient);
    } else {
        cv::Canny(image.toMat(), edges.toMat(), threshold1, threshold2, apertureSize, L2gradient);
    }
    return edges;
}

extern "C"
struct TensorWrapper cornerMinEigenVal(
        struct TensorWrapper src, struct TensorWrapper dst,
        int blockSize, int ksize, int borderType)
{
    if (dst.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::cornerMinEigenVal(src.toMat(), retval, blockSize, ksize, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place 
        cv::Mat source = src.toMat();
        cv::cornerMinEigenVal(source, source, blockSize, ksize, borderType);
    } else {
        cv::cornerMinEigenVal(src.toMat(), dst.toMat(), blockSize, ksize, borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper cornerHarris(
        struct TensorWrapper src, struct TensorWrapper dst, int blockSize,
        int ksize, double k, int borderType)
{
    if (dst.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::cornerHarris(src.toMat(), retval, blockSize,
                ksize, k, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place 
        cv::Mat source = src.toMat();
        cv::cornerHarris(source, source, blockSize,
                ksize, k, borderType);
    } else {
        cv::cornerHarris(src.toMat(), dst.toMat(), blockSize,
                ksize, k, borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper cornerEigenValsAndVecs(
        struct TensorWrapper src, struct TensorWrapper dst,
        int blockSize, int ksize, int borderType)
{
    if (dst.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::cornerEigenValsAndVecs(src.toMat(), retval, blockSize, ksize, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place 
        cv::Mat source = src.toMat();
        cv::cornerEigenValsAndVecs(source, source, blockSize, ksize, borderType);
    } else {
        cv::cornerEigenValsAndVecs(src.toMat(), dst.toMat(), blockSize, ksize, borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper preCornerDetect(
        struct TensorWrapper src, struct TensorWrapper dst, int ksize, int borderType)
{
    if (dst.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::preCornerDetect(src.toMat(), retval, ksize, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place 
        cv::Mat source = src.toMat();
        cv::preCornerDetect(source, source, ksize, borderType);
    } else {
        cv::preCornerDetect(src.toMat(), dst.toMat(), ksize, borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper HoughLines(
        struct TensorWrapper image, struct TensorWrapper lines,
        double rho, double theta, int threshold, double srn, double stn,
        double min_theta, double max_theta)
{
    if (lines.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::HoughLines(image.toMat(), retval, rho, theta, threshold, srn, stn,
                min_theta, max_theta);
        return TensorWrapper(retval);
    } else if (lines.tensorPtr == image.tensorPtr) {
        // in-place 
        cv::Mat source = image.toMat();
        cv::HoughLines(source, source, rho, theta, threshold, srn, stn,
                min_theta, max_theta);
    } else {
        cv::HoughLines(image.toMat(), lines.toMat(), rho, theta, threshold, srn, stn,
                min_theta, max_theta);
    }
    return lines;
}

extern "C"
struct TensorWrapper HoughLinesP(
        struct TensorWrapper image, struct TensorWrapper lines, double rho,
        double theta, int threshold, double minLineLength, double maxLineGap)
{
    if (lines.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::HoughLinesP(image.toMat(), retval, rho,
                theta, threshold, minLineLength, maxLineGap);
        return TensorWrapper(retval);
    } else if (lines.tensorPtr == image.tensorPtr) {
        // in-place 
        cv::Mat source = image.toMat();
        cv::HoughLinesP(source, source, rho,
                theta, threshold, minLineLength, maxLineGap);
    } else {
        cv::HoughLinesP(image.toMat(), lines.toMat(), rho,
                theta, threshold, minLineLength, maxLineGap);
    }
    return lines;
}

extern "C"
struct TensorWrapper HoughCircles(
        struct TensorWrapper image, struct TensorWrapper circles,
        int method, double dp, double minDist, double param1, double param2,
        int minRadius, int maxRadius)
{
    if (circles.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::HoughCircles(image.toMat(), retval, method, dp, minDist, param1, param2,
                minRadius, maxRadius);
        return TensorWrapper(retval);
    } else if (circles.tensorPtr == image.tensorPtr) {
        // in-place 
        cv::Mat source = image.toMat();
        cv::HoughCircles(source, source, method, dp, minDist, param1, param2,
                minRadius, maxRadius);
    } else {
        cv::HoughCircles(image.toMat(), circles.toMat(), method, dp, minDist, param1, param2,
                minRadius, maxRadius);
    }
    return circles;
}

extern "C"
struct TensorWrapper cornerSubPix(
        struct TensorWrapper image, struct TensorWrapper corners,
        int winSize_x, int winSize_y, int zeroZone_x, int zeroZone_y,
        int crit_type, int crit_max_iter, double crit_eps)
{
    if (corners.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::cornerSubPix(image.toMat(), retval, cv::Size(winSize_x, winSize_y), cv::Size(zeroZone_x, zeroZone_y),
                cv::TermCriteria(crit_type, crit_max_iter, crit_eps));
        return TensorWrapper(retval);
    } else if (corners.tensorPtr == image.tensorPtr) {
        // in-place
        cv::Mat source = image.toMat();
        cv::cornerSubPix(source, source, cv::Size(winSize_x, winSize_y), cv::Size(zeroZone_x, zeroZone_y),
                cv::TermCriteria(crit_type, crit_max_iter, crit_eps));
    } else {
        cv::cornerSubPix(image.toMat(), corners.toMat(), cv::Size(winSize_x, winSize_y), cv::Size(zeroZone_x, zeroZone_y),
                cv::TermCriteria(crit_type, crit_max_iter, crit_eps));
    }
    return corners;
}

extern "C"
struct TensorWrapper goodFeaturesToTrack(
        struct TensorWrapper image, struct TensorWrapper corners,
        int maxCorners, double qualityLevel, double minDistance,
        struct TensorWrapper mask, int blockSize, bool useHarrisDetector, double k)
{
    if (corners.tensorPtr == nullptr) {
        cv::Mat retval;
        cv::goodFeaturesToTrack(image.toMat(), retval, maxCorners, qualityLevel, minDistance,
        mask.toMat(), blockSize, useHarrisDetector, k);
        return TensorWrapper(retval);
    } else if (corners.tensorPtr == image.tensorPtr) {
        // in-place 
        cv::Mat source = image.toMat();
        cv::goodFeaturesToTrack(source, source, maxCorners, qualityLevel, minDistance,
        mask.toMat(), blockSize, useHarrisDetector, k);
    } else {
        cv::goodFeaturesToTrack(image.toMat(), corners.toMat(), maxCorners, qualityLevel, minDistance,
        mask.toMat(), blockSize, useHarrisDetector, k);
    }
    return corners;
}