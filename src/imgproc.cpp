#include <imgproc.hpp>

// Ехали медведи, на велосипеде

extern "C"
struct TensorWrapper getGaussianKernel(int ksize, double sigma, int ktype)
{
    return TensorWrapper(
            cv::getGaussianKernel(ksize, sigma, ktype));
}

extern "C"
struct TensorArray getDerivKernels(
        int dx, int dy, int ksize,
        bool normalize, int ktype)
{
    std::vector<cv::Mat> output(2);

    cv::getDerivKernels(
            output[0], output[1],
            dx, dy, ksize, normalize, ktype);

    return TensorArray(output);
}

extern "C"
struct TensorWrapper getGaborKernel(struct SizeWrapper ksize, double sigma, double theta,
                                    double lambd, double gamma, double psi, int ktype)
{
    return TensorWrapper(
            cv::getGaborKernel(
                    ksize, sigma, theta, lambd, gamma, psi, ktype));
}

extern "C"
struct TensorWrapper getStructuringElement(int shape, struct SizeWrapper ksize,
                                           struct PointWrapper anchor)
{
    return TensorWrapper(
            cv::getStructuringElement(
                    shape, ksize, anchor));
}

extern "C"
struct TensorWrapper medianBlur(struct TensorWrapper src, struct TensorWrapper dst, int ksize)
{
    if (dst.isNull()) {
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
                                  struct SizeWrapper ksize, double sigmaX,
                                  double sigmaY, int borderType)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::GaussianBlur(
                src.toMat(), retval, ksize, sigmaX, sigmaY, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::GaussianBlur(
                source, source, ksize, sigmaX, sigmaY, borderType);
    } else {
        cv::GaussianBlur(
                src.toMat(), dst.toMat(), ksize, sigmaX, sigmaY, borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper bilateralFilter(struct TensorWrapper src, struct TensorWrapper dst, int d,
                                     double sigmaColor, double sigmaSpace,
                                     int borderType)
{
    if (dst.isNull()) {
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
        struct SizeWrapper ksize, struct PointWrapper anchor,
        bool normalize, int borderType)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::boxFilter(
                src.toMat(), retval, ddepth, ksize,
                anchor, normalize, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::boxFilter(
                source, source, ddepth, ksize,
                anchor, normalize, borderType);
    } else {
        cv::boxFilter(
                src.toMat(), dst.toMat(), ddepth, ksize,
                anchor, normalize, borderType);
    }
    return dst;
}


extern "C"
struct TensorWrapper sqrBoxFilter(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct SizeWrapper ksize, struct PointWrapper anchor,
        bool normalize, int borderType)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::sqrBoxFilter(src.toMat(), retval, ddepth,
                         ksize, anchor,
                         normalize, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place 
        cv::Mat source = src.toMat();
        cv::sqrBoxFilter(source, source, ddepth, ksize, anchor,
                         normalize, borderType);
    } else {
        cv::sqrBoxFilter(src.toMat(), dst.toMat(), ddepth, ksize, anchor,
                         normalize, borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper blur(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper ksize, struct PointWrapper anchor, int borderType)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::blur(src.toMat(), retval, ksize, anchor, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place 
        cv::Mat source = src.toMat();
        cv::blur(source, source, ksize, anchor, borderType);
    } else {
        cv::blur(src.toMat(), dst.toMat(), ksize, anchor, borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper filter2D(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct TensorWrapper kernel, struct PointWrapper anchor,
        double delta, int borderType)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::filter2D(src.toMat(), retval, ddepth,
                     kernel.toMat(), anchor,
                     delta, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place 
        cv::Mat source = src.toMat();
        cv::filter2D(source, source, ddepth,
                     kernel.toMat(), anchor,
                     delta, borderType);
    } else {
        cv::filter2D(src.toMat(), dst.toMat(), ddepth,
                     kernel.toMat(), anchor,
                     delta, borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper sepFilter2D(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct TensorWrapper kernelX,struct TensorWrapper kernelY,
        struct PointWrapper anchor, double delta, int borderType)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::sepFilter2D(src.toMat(), retval, ddepth,
                        kernelX.toMat(),kernelY.toMat(),
                        anchor, delta, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place 
        cv::Mat source = src.toMat();
        cv::sepFilter2D(source, source, ddepth,
                        kernelX.toMat(),kernelY.toMat(),
                        anchor, delta, borderType);
    } else {
        cv::sepFilter2D(src.toMat(), dst.toMat(), ddepth,
                        kernelX.toMat(),kernelY.toMat(),
                        anchor, delta, borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper Sobel(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int dx, int dy, int ksize, double scale, double delta, int borderType)
{
    if (dst.isNull()) {
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
    if (dst.isNull()) {
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
    if (dst.isNull()) {
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
    if (edges.isNull()) {
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
    if (dst.isNull()) {
        cv::Mat retval;
        cv::cornerMinEigenVal(src.toMat(), retval, blockSize, ksize, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        std::cout << "inplace" << std::endl;
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
    if (dst.isNull()) {
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
    if (dst.isNull()) {
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
    if (dst.isNull()) {
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
        struct TensorWrapper image,
        double rho, double theta, int threshold, double srn, double stn,
        double min_theta, double max_theta)
{
    cv::Mat retval;
    cv::HoughLines(image.toMat(), retval, theta, threshold, srn, stn,
                   min_theta, max_theta);
    return TensorWrapper(retval);
}

extern "C"
struct TensorWrapper HoughLinesP(
        struct TensorWrapper image, double rho,
        double theta, int threshold, double minLineLength, double maxLineGap)
{
    cv::Mat retval;
    cv::HoughLinesP(image.toMat(), retval, rho,
                    theta, threshold, minLineLength, maxLineGap);
    return TensorWrapper(retval);
}

extern "C"
struct TensorWrapper HoughCircles(
        struct TensorWrapper image,
        int method, double dp, double minDist, double param1, double param2,
        int minRadius, int maxRadius)
{
    cv::Mat retval;
    cv::HoughCircles(image.toMat(), retval, method, dp, minDist, param1, param2,
                     minRadius, maxRadius);
    return TensorWrapper(retval);
}

extern "C"
void cornerSubPix(
        struct TensorWrapper image, struct TensorWrapper corners,
        struct SizeWrapper winSize, struct SizeWrapper zeroZone,
        struct TermCriteriaWrapper criteria)
{
    cv::cornerSubPix(image.toMat(), corners.toMat(), winSize,
                     zeroZone, criteria);
}

extern "C"
struct TensorWrapper goodFeaturesToTrack(
        struct TensorWrapper image,
        int maxCorners, double qualityLevel, double minDistance,
        struct TensorWrapper mask, int blockSize, bool useHarrisDetector, double k)
{
    cv::Mat retval;
    cv::goodFeaturesToTrack(image.toMat(), retval, maxCorners, qualityLevel, minDistance,
                            TO_MAT_OR_NOARRAY(mask), blockSize, useHarrisDetector, k);
    return retval;
}

extern "C"
struct TensorWrapper erode(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper kernel, struct PointWrapper anchor,
        int iterations, int borderType, struct ScalarWrapper borderValue)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::erode(src.toMat(), retval, kernel.toMat(), anchor, iterations,
                  borderType, borderValue.orDefault(cv::morphologyDefaultBorderValue()));
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::erode(source, source, kernel.toMat(), anchor, iterations,
                  borderType, borderValue.orDefault(cv::morphologyDefaultBorderValue()));
    } else {
        cv::erode(src.toMat(), dst.toMat(), kernel.toMat(), anchor, iterations,
                  borderType, borderValue.orDefault(cv::morphologyDefaultBorderValue()));
    }
    return dst;
}

extern "C"
struct TensorWrapper dilate(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper kernel, struct PointWrapper anchor,
        int iterations, int borderType, struct ScalarWrapper borderValue)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::dilate(src.toMat(), retval, kernel.toMat(), anchor, iterations,
                   borderType, borderValue.orDefault(cv::morphologyDefaultBorderValue()));
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::dilate(source, source, kernel.toMat(), anchor, iterations,
                   borderType, borderValue.orDefault(cv::morphologyDefaultBorderValue()));
    } else {
        cv::dilate(src.toMat(), dst.toMat(), kernel.toMat(), anchor, iterations,
                   borderType, borderValue.orDefault(cv::morphologyDefaultBorderValue()));
    }
    return dst;
}

extern "C"
struct TensorWrapper morphologyEx(
        struct TensorWrapper src, struct TensorWrapper dst,
        int op, struct TensorWrapper kernel, struct PointWrapper anchor,
        int iterations, int borderType, struct ScalarWrapper borderValue)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::morphologyEx(src.toMat(), retval, op, kernel.toMat(), anchor, iterations,
                         borderType, borderValue.orDefault(cv::morphologyDefaultBorderValue()));
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::morphologyEx(source, source, op, kernel.toMat(), anchor, iterations,
                         borderType, borderValue.orDefault(cv::morphologyDefaultBorderValue()));
    } else {
        cv::morphologyEx(src.toMat(), dst.toMat(), op, kernel.toMat(), anchor, iterations,
                         borderType, borderValue.orDefault(cv::morphologyDefaultBorderValue()));
    }
    return dst;
}

extern "C"
struct TensorWrapper resize(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dsize, double fx, double fy,
        int interpolation)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::resize(src.toMat(), retval, dsize, fx, fy, interpolation);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::resize(source, source, dsize, fx, fy, interpolation);
    } else {
        cv::resize(src.toMat(), dst.toMat(), dsize, fx, fy, interpolation);
    }
    return dst;
}

extern "C"
struct TensorWrapper warpAffine(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper M, struct SizeWrapper dsize,
        int flags, int borderMode, struct ScalarWrapper borderValue)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::warpAffine(src.toMat(), retval, M.toMat(), dsize, flags, borderMode, borderValue);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::warpAffine(source, source, M.toMat(), dsize, flags, borderMode, borderValue);
    } else {
        cv::warpAffine(src.toMat(), dst.toMat(), M.toMat(), dsize, flags, borderMode, borderValue);
    }
    return dst;
}

extern "C"
struct TensorWrapper warpPerspective(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper M, struct SizeWrapper dsize,
        int flags, int borderMode, struct ScalarWrapper borderValue)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::warpPerspective(src.toMat(), retval, M.toMat(), dsize, flags, borderMode, borderValue);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::warpPerspective(source, source, M.toMat(), dsize, flags, borderMode, borderValue);
    } else {
        cv::warpPerspective(src.toMat(), dst.toMat(), M.toMat(), dsize, flags, borderMode, borderValue);
    }
    return dst;
}

extern "C"
struct TensorWrapper remap(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper map1, struct TensorWrapper map2,
        int interpolation, int borderMode, struct ScalarWrapper borderValue)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::remap(src.toMat(), retval, map1.toMat(), map2.toMat(), interpolation, borderMode, borderValue);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::remap(source, source, map1.toMat(), map2.toMat(), interpolation, borderMode, borderValue);
    } else {
        cv::remap(src.toMat(), dst.toMat(), map1.toMat(), map2.toMat(), interpolation, borderMode, borderValue);
    }
    return dst;
}

extern "C"
struct TensorArray convertMaps(
        struct TensorWrapper map1, struct TensorWrapper map2,
        struct TensorWrapper dstmap1, struct TensorWrapper dstmap2,
        int dstmap1type, bool nninterpolation)
{
    if (dstmap1.isNull() and dstmap2.isNull()) {
        // output to retval
        std::vector<cv::Mat> retval(2);
        cv::convertMaps(map1.toMat(), map2.toMat(), retval[0], retval[1], dstmap1type, nninterpolation);
        return TensorArray(retval);
    }
    if (!dstmap1.isNull() and !dstmap2.isNull()) {
        // try to output to the given Tensors
        cv::convertMaps(map1.toMat(), map2.toMat(), dstmap1.toMat(), dstmap2.toMat(), dstmap1type, nninterpolation);
        return TensorArray();
    }
    THError("convertMaps: please specify either both or none of the dstmaps");
}

extern "C"
struct TensorWrapper getRotationMatrix2D(
        struct Point2fWrapper center, double angle, double scale)
{
    return TensorWrapper(cv::getRotationMatrix2D(center, angle, scale));
}

extern "C"
struct TensorWrapper invertAffineTransform(
        struct TensorWrapper M, struct TensorWrapper iM)
{
    if (iM.isNull()) {
        cv::Mat retval;
        cv::invertAffineTransform(M.toMat(), retval);
        return TensorWrapper(retval);
    } else if (iM.tensorPtr == M.tensorPtr) {
        // in-place
        cv::Mat source = M.toMat();
        cv::invertAffineTransform(source, source);
    } else {
        cv::invertAffineTransform(M.toMat(), iM.toMat());
    }
    return iM;
}

extern "C" struct TensorWrapper getPerspectiveTransform(
        struct TensorWrapper src, struct TensorWrapper dst)
{
    return TensorWrapper(cv::getPerspectiveTransform(src.toMat(), dst.toMat()));
}

extern "C" struct TensorWrapper getAffineTransform(
        struct TensorWrapper src, struct TensorWrapper dst)
{
    return TensorWrapper(cv::getAffineTransform(src.toMat(), dst.toMat()));
}

extern "C" struct TensorWrapper getRectSubPix(
        struct TensorWrapper image, struct SizeWrapper patchSize,
        struct Point2fWrapper center, struct TensorWrapper patch, int patchType)
{
    if (patch.isNull()) {
        cv::Mat retval;
        cv::getRectSubPix(image.toMat(), patchSize,
                          center, retval, patchType);
        return TensorWrapper(retval);
    } else if (image.tensorPtr == patch.tensorPtr) {
        // in-place
        THError("In-place isn't possible");
    } else {
        cv::getRectSubPix(image.toMat(), patchSize,
                          center, patch.toMat(), patchType);
    }
    return patch;
}

extern "C"
struct TensorWrapper logPolar(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct Point2fWrapper center, double M, int flags)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::logPolar(src.toMat(), retval, center, M, flags);
        return TensorWrapper(retval);
    } else if (src.tensorPtr == dst.tensorPtr) {
        // in-place
        THError("In-place isn't possible");
    } else {
        cv::logPolar(src.toMat(), dst.toMat(), center, M, flags);
    }
    return dst;
}

extern "C"
struct TensorWrapper linearPolar(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct Point2fWrapper center, double maxRadius, int flags)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::linearPolar(src.toMat(), retval, center, maxRadius, flags);
        return TensorWrapper(retval);
    } else if (src.tensorPtr == dst.tensorPtr) {
        // in-place
        THError("In-place isn't possible");
    } else {
        cv::linearPolar(src.toMat(), dst.toMat(), center, maxRadius, flags);
    }
    return dst;
}

extern "C"
struct TensorWrapper integral(
        struct TensorWrapper src, struct TensorWrapper sum, int sdepth)
{
    if (sum.isNull()) {
        cv::Mat retval;
        cv::integral(src.toMat(), retval, sdepth);
        return TensorWrapper(retval);
    } else if (sum.tensorPtr == src.tensorPtr) {
        // in-place 
        cv::Mat source = src.toMat();
        cv::integral(source, source, sdepth);
    } else {
        cv::integral(src.toMat(), sum.toMat(), sdepth);
    }
    return sum;
}

extern "C" struct TensorArray integralN(
        struct TensorWrapper src, struct TensorArray sums, int sdepth, int sqdepth)
{
    // sums.size == 2 or 3
    std::vector<cv::Mat> retval(sums.size);

    for (short i = 0; i < sums.size; ++i) {
        if (!sums.tensors[i].isNull()) {
            retval[i] = sums.tensors[i].toMat();
        }
    }
    cv::integral(src.toMat(), retval[0], retval[1], sdepth, sqdepth);

    return TensorArray(retval);
}

extern "C"
void accumulate(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper mask)
{
    cv::accumulate(src.toMat(), dst.toMat(), TO_MAT_OR_NOARRAY(mask));
}

extern "C"
void accumulateSquare(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper mask)
{
    cv::accumulateSquare(src.toMat(), dst.toMat(), TO_MAT_OR_NOARRAY(mask));
}

extern "C"
void accumulateProduct(
        struct TensorWrapper src1, struct TensorWrapper src2,
        struct TensorWrapper dst, struct TensorWrapper mask)
{
    cv::accumulateProduct(src1.toMat(), src2.toMat(), dst.toMat(), TO_MAT_OR_NOARRAY(mask));
}

extern "C"
void accumulateWeighted(
        struct TensorWrapper src, struct TensorWrapper dst,
        double alpha, struct TensorWrapper mask)
{
    cv::accumulateWeighted(src.toMat(), dst.toMat(), alpha, TO_MAT_OR_NOARRAY(mask));
}

extern "C"
struct Vec3dWrapper phaseCorrelate(
        struct TensorWrapper src1, struct TensorWrapper src2,
        struct TensorWrapper window)
{
    Vec3dWrapper retval;
    cv::Point2d result =
            cv::phaseCorrelate(src1.toMat(), src2.toMat(),
                               TO_MAT_OR_NOARRAY(window), &retval.v2);
    retval.v0 = result.x;
    retval.v1 = result.y;
    return retval;
}

extern "C"
struct TensorWrapper createHanningWindow(
        struct TensorWrapper dst, struct SizeWrapper winSize, int type)
{
    if (dst.isNull()) {
        // output to retval
        cv::Mat retval;
        cv::createHanningWindow(retval, winSize, type);
        return TensorWrapper(retval);
    } else {
        // try to output to dst
        cv::Mat dstMat = dst.toMat();
        cv::createHanningWindow(dstMat, winSize, type);
        return dst;
    }
}

extern "C"
struct TensorPlusDouble threshold(
        struct TensorWrapper src, struct TensorWrapper dst,
        double thresh, double maxval, int type)
{
    TensorPlusDouble retval;
    if (dst.isNull()) {
        // output to retval
        cv::Mat result;
        retval.val = cv::threshold(src.toMat(), result, thresh, maxval, type);
        new (&retval.tensor) TensorWrapper(result);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        retval.val = cv::threshold(source, source, thresh, maxval, type);
        retval.tensor = src;
    } else {
        // try to output to dst
        retval.val = cv::threshold(src.toMat(), dst.toMat(), thresh, maxval, type);
        retval.tensor = dst;
    }
    return retval;
}

extern "C"
struct TensorWrapper adaptiveThreshold(
        struct TensorWrapper src, struct TensorWrapper dst,
        double maxValue, int adaptiveMethod, int thresholdType,
        int blockSize, double C)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::adaptiveThreshold(src.toMat(), retval, maxValue, adaptiveMethod, thresholdType, blockSize, C);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place 
        cv::Mat source = src.toMat();
        cv::adaptiveThreshold(source, source, maxValue, adaptiveMethod, thresholdType, blockSize, C);
    } else {
        cv::adaptiveThreshold(src.toMat(), dst.toMat(), maxValue, adaptiveMethod, thresholdType, blockSize, C);
    }
    return dst;
}

extern "C"
struct TensorWrapper pyrDown(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dstSize, int borderType)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::pyrDown(src.toMat(), retval, dstSize, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::pyrDown(source, source, dstSize, borderType);
    } else {
        cv::pyrDown(src.toMat(), dst.toMat(), dstSize, borderType);
    }
    return dst;
}

extern "C"
struct TensorWrapper pyrUp(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dstSize, int borderType)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::pyrUp(src.toMat(), retval, dstSize, borderType);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::pyrUp(source, source, dstSize, borderType);
    } else {
        cv::pyrUp(src.toMat(), dst.toMat(), dstSize, borderType);
    }
    return dst;
}

extern "C" struct TensorArray buildPyramid(
        struct TensorWrapper src, struct TensorArray dst,
        int maxlevel, int borderType)
{
    if (dst.isNull()) {
        std::vector<cv::Mat> retval;
        cv::buildPyramid(src.toMat(), retval, maxlevel, borderType);
        return TensorArray(retval);
    } else {
        cv::buildPyramid(src.toMat(), dst.toMatList(), maxlevel, borderType);
        return dst;
    }
}

extern "C" struct TensorWrapper undistort(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper newCameraMatrix)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::undistort(src.toMat(), retval, cameraMatrix.toMat(),
                      TO_MAT_OR_NOARRAY(distCoeffs), TO_MAT_OR_NOARRAY(newCameraMatrix));
        return TensorWrapper(retval);
    } else {
        // output to dst
        cv::undistort(src.toMat(), dst.toMat(), cameraMatrix.toMat(),
                      TO_MAT_OR_NOARRAY(distCoeffs), TO_MAT_OR_NOARRAY(newCameraMatrix));
        return dst;
    }
}

extern "C" struct TensorArray initUndistortRectifyMap(
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper R, struct TensorWrapper newCameraMatrix,
        struct SizeWrapper size, int m1type,
        struct TensorArray maps)
{
    if (maps.isNull()) {
        // output to retval
        std::vector<cv::Mat> retval(2);
        cv::initUndistortRectifyMap(
                cameraMatrix.toMat(), TO_MAT_OR_NOARRAY(distCoeffs),
                TO_MAT_OR_NOARRAY(R), newCameraMatrix.toMat(),
                size, m1type, retval[0], retval[1]);
        return TensorArray(retval);
    } else {
        // oh. try to output to 'maps'...
        auto mapsVector = maps.toMatList();
        cv::initUndistortRectifyMap(
                cameraMatrix.toMat(), TO_MAT_OR_NOARRAY(distCoeffs),
                TO_MAT_OR_NOARRAY(R), newCameraMatrix.toMat(),
                size, m1type, mapsVector[0], mapsVector[1]);
        return maps;
    }
}

extern "C" struct TensorArrayPlusFloat initWideAngleProjMap(
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct SizeWrapper imageSize, int destImageWidth,
        int m1type, struct TensorArray maps,
        int projType, double alpha)
{
    if (maps.isNull()) {
        // output to retval
        TensorArrayPlusFloat retval;
        std::vector<cv::Mat> resultMats(2);
        retval.val = cv::initWideAngleProjMap(
                cameraMatrix.toMat(), TO_MAT_OR_NOARRAY(distCoeffs),
                imageSize, destImageWidth,
                m1type, resultMats[0], resultMats[1], projType, alpha);
        new (&retval.tensors) TensorArray(resultMats);
        return retval;
    } else {
        // oh. try to output to 'maps' and return only float...
        TensorArrayPlusFloat retval;
        retval.tensors.tensors = nullptr;
        auto mapsVec = maps.toMatList();
        retval.val = cv::initWideAngleProjMap(
                cameraMatrix.toMat(), TO_MAT_OR_NOARRAY(distCoeffs),
                imageSize, destImageWidth,
                m1type, mapsVec[0], mapsVec[1], projType, alpha);
        return retval;
    }
}

extern "C" struct TensorWrapper getDefaultNewCameraMatrix(
        struct TensorWrapper cameraMatrix, struct SizeWrapper imgsize, bool centerPrincipalPoint)
{
    return TensorWrapper(
            cv::getDefaultNewCameraMatrix(
                    cameraMatrix.toMat(), imgsize, centerPrincipalPoint
            ));
}

extern "C" struct TensorWrapper undistortPoints(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper R, struct TensorWrapper P)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::undistortPoints(
                src.toMat(), retval, cameraMatrix.toMat(),
                TO_MAT_OR_NOARRAY(distCoeffs), TO_MAT_OR_NOARRAY(R),
                TO_MAT_OR_NOARRAY(P));
        return TensorWrapper(retval);
    } else {
        // output to dst
        cv::undistortPoints(
                src.toMat(), dst.toMat(), cameraMatrix.toMat(),
                TO_MAT_OR_NOARRAY(distCoeffs), TO_MAT_OR_NOARRAY(R),
                TO_MAT_OR_NOARRAY(P));
        return dst;
    }
}

extern "C" struct TensorWrapper calcHist(
        struct TensorArray images,
        struct IntArray channels, struct TensorWrapper mask,
        struct TensorWrapper hist, int dims, struct IntArray histSize,
        struct FloatArrayOfArrays ranges, bool uniform, bool accumulate)
{
    auto imagesVec = images.toMatList();
    if (hist.isNull()) {
        // output to retval
        cv::Mat retval;
        cv::calcHist(
                imagesVec.data(), imagesVec.size(), channels.data, TO_MAT_OR_NOARRAY(mask),
                retval, dims, histSize.data, const_cast<const float**>(ranges.pointers),
                uniform, false);
        return TensorWrapper(retval);
    } else {
        // output to given hist
        cv::calcHist(
                imagesVec.data(), imagesVec.size(), channels.data, TO_MAT_OR_NOARRAY(mask),
                hist.toMat(), dims, histSize.data, const_cast<const float**>(ranges.pointers),
                uniform, accumulate);
        return hist;
    }
}

extern "C" struct TensorWrapper calcBackProject(
        struct TensorArray images, int nimages,
        struct IntArray channels, struct TensorWrapper hist,
        struct TensorWrapper backProject, struct FloatArrayOfArrays ranges,
        double scale, bool uniform)
{
    auto imagesVec = images.toMatList();
    if (hist.isNull()) {
        // output to retval
        cv::Mat retval;
        cv::calcBackProject(
                imagesVec.data(), nimages, channels.data, hist.toMat(), retval,
                const_cast<const float **>(ranges.pointers), scale, uniform);
        return TensorWrapper(retval);
    } else {
        // output to given 'backProject'
        cv::calcBackProject(
                imagesVec.data(), nimages, channels.data, hist.toMat(), backProject.toMat(),
                const_cast<const float **>(ranges.pointers), scale, uniform);
        return backProject;
    }
}

extern "C" double compareHist(
        struct TensorWrapper H1, struct TensorWrapper H2, int method)
{
    return cv::compareHist(H1.toMat(), H2.toMat(), method);
}

extern "C" struct TensorWrapper equalizeHist(
        struct TensorWrapper src, struct TensorWrapper dst)
{
    if (dst.isNull()) {
        // output to retval
        cv::Mat retval;
        cv::equalizeHist(src.toMat(), retval);
        return TensorWrapper(retval);
    } else {
        // try to output to dst
        cv::equalizeHist(src.toMat(), dst.toMat());
        return dst;
    }
}

extern "C" float EMD(
        struct TensorWrapper signature1, struct TensorWrapper signature2,
        int distType, struct TensorWrapper cost,
        struct FloatArray lowerBound, struct TensorWrapper flow)
{
    return cv::EMD(
            signature1.toMat(), signature2.toMat(), distType,
            TO_MAT_OR_NOARRAY(cost), lowerBound.data, TO_MAT_OR_NOARRAY(flow));
}

extern "C" void watershed(
        struct TensorWrapper image, struct TensorWrapper markers)
{
    cv::watershed(image.toMat(), markers.toMat());
}

extern "C" struct TensorWrapper pyrMeanShiftFiltering(
        struct TensorWrapper src, struct TensorWrapper dst,
        double sp, double sr, int maxLevel, TermCriteriaWrapper termcrit)
{
    if (dst.isNull()) {
        // output to retval
        cv::Mat retval;
        cv::pyrMeanShiftFiltering(
                src.toMat(), retval,
                sp, sr, maxLevel,
                termcrit.orDefault(
                        cv::TermCriteria(
                                cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 5, 1)));
        return TensorWrapper(retval);
    } else {
        // output to dst
        cv::pyrMeanShiftFiltering(
                src.toMat(), dst.toMat(),
                sp, sr, maxLevel,
                termcrit.orDefault(
                        cv::TermCriteria(
                                cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 5, 1)));
        return dst;
    }
}

extern "C" void grabCut(
        struct TensorWrapper img, struct TensorWrapper mask,
        struct RectWrapper rect, struct TensorWrapper bgdModel,
        struct TensorWrapper fgdModel, int iterCount, int mode)
{
    cv::grabCut(
            img.toMat(), mask.toMat(), rect, bgdModel.toMat(),
            fgdModel.toMat(), iterCount, mode);
}

extern "C" struct TensorWrapper distanceTransform(
        struct TensorWrapper src, struct TensorWrapper dst,
        int distanceType, int maskSize, int dstType)
{
    if (dst.isNull()) {
        // output to retval
        cv::Mat retval;
        cv::distanceTransform(
                src.toMat(), retval, distanceType, maskSize, dstType);
        return TensorWrapper(retval);
    } else {
        // output to dst
        cv::distanceTransform(
                src.toMat(), dst.toMat(), distanceType, maskSize, dstType);
        return dst;
    }
}

extern "C" struct TensorArray distanceTransformWithLabels(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper labels, int distanceType, int maskSize,
        int labelType)
{
    char outputVecSize = dst.isNull() + labels.isNull();
    if (outputVecSize == 0) {
        cv::distanceTransform(
                src.toMat(), dst.toMat(), labels.toMat(),
                distanceType, maskSize, labelType);
        return TensorArray();
    } else {
        std::vector<cv::Mat> retval(outputVecSize);
        cv::distanceTransform(
                src.toMat(),
                (dst.isNull() ? retval[0] : dst.toMat()),
                (src.isNull() ? retval[dst.isNull()] : src.toMat()),
                distanceType, maskSize, labelType);
        return TensorArray(retval);
    }
}

extern "C" struct RectPlusInt floodFill(
        struct TensorWrapper image, struct TensorWrapper mask,
        struct PointWrapper seedPoint, struct ScalarWrapper newVal,
        struct ScalarWrapper loDiff, struct ScalarWrapper upDiff, int flags)
{
    RectPlusInt retval;
    cv::Rect funcResult;
    if (mask.isNull()) {
        retval.val = cv::floodFill(
                image.toMat(), seedPoint,
                newVal, &funcResult, loDiff, upDiff, flags);
    } else {
        retval.val = cv::floodFill(
                image.toMat(), mask.toMat(), seedPoint,
                newVal, &funcResult, loDiff, upDiff, flags);
    }
    retval.rect = funcResult;
    return retval;
}

extern "C" struct TensorWrapper cvtColor(
        struct TensorWrapper src, struct TensorWrapper dst, int code, int dstCn)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::cvtColor(src.toMat(), retval, code, dstCn);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::cvtColor(source, source, code, dstCn);
    } else {
        cv::cvtColor(src.toMat(), dst.toMat(), code, dstCn);
    }
    return dst;
}

extern "C"
struct TensorWrapper demosaicing(
        struct TensorWrapper _src, struct TensorWrapper _dst, int code, int dcn)
{
    if (_dst.isNull()) {
        cv::Mat retval;
        cv::demosaicing(_src.toMat(), retval, code, dcn);
        return TensorWrapper(retval);
    } else if (_src.tensorPtr == _dst.tensorPtr) {
        cv::Mat source = _src.toMat();
        cv::demosaicing(source, source, code, dcn);
    } else {
        cv::demosaicing(_src.toMat(), _dst.toMat(), code, dcn);
    }
    return _dst;
}

extern "C"
struct MomentsWrapper moments(
        struct TensorWrapper array, bool binaryImage)
{
    return cv::moments(array.toMat(), binaryImage);
}

extern "C"
struct DoubleArray HuMoments(struct MomentsWrapper m)
{
    DoubleArray retval;
    retval.size = 7;
    retval.data = static_cast<double *>(malloc(retval.size * sizeof(double)));
    cv::HuMoments(m, retval.data);
    return retval;
}

extern "C"
struct TensorWrapper matchTemplate(
        struct TensorWrapper image, struct TensorWrapper templ, struct TensorWrapper result, int method, struct TensorWrapper mask)
{
    if (result.isNull()) {
        cv::Mat retval;
        cv::matchTemplate(image.toMat(), templ.toMat(), retval, method, TO_MAT_OR_NOARRAY(mask));
        return TensorWrapper(retval);
    } else if (image.tensorPtr == result.tensorPtr) {
        cv::Mat source = image.toMat();
        cv::matchTemplate(source, templ.toMat(), source, method, TO_MAT_OR_NOARRAY(mask));
    } else {
        cv::matchTemplate(image.toMat(), templ.toMat(), result.toMat(), method, TO_MAT_OR_NOARRAY(mask));
    }
    return result;
}

extern "C"
struct TensorPlusInt connectedComponents(
        struct TensorWrapper image, struct TensorWrapper labels, int connectivity, int ltype)
{
    TensorPlusInt retval;
    if (labels.isNull()) {
        cv::Mat result;
        retval.val = cv::connectedComponents(image.toMat(), result, connectivity, ltype);
        new (&retval.tensor) TensorWrapper(result);
    } else if (image.tensorPtr == labels.tensorPtr) {
        cv::Mat source = image.toMat();
        retval.val = cv::connectedComponents(source, source, connectivity, ltype);
        retval.tensor = labels;
    } else {
        retval.val = cv::connectedComponents(image.toMat(), labels.toMat(), connectivity, ltype);
        retval.tensor = labels;
    }
    return retval;
}

extern "C"
struct TensorArrayPlusInt connectedComponentsWithStats(
        struct TensorWrapper image, struct TensorArray outputTensors, int connectivity, int ltype)
{
    std::vector<cv::Mat> output(outputTensors);
    TensorArrayPlusInt retval;
    retval.val = cv::connectedComponentsWithStats(
            image.toMat(), output[0], output[1], output[2], connectivity, ltype);
    retval.tensors = TensorArray(output);
    return retval;
}

extern "C"
struct TensorArray findContours(
        struct TensorWrapper image, bool withHierarchy, struct TensorWrapper hierarchy, int mode, int method, struct PointWrapper offset)
{
    std::vector<cv::Mat> retval;
    if (withHierarchy) {
        if (hierarchy.isNull()) {
            cv::Mat hierarchyMat;
            cv::findContours(image.toMat(), retval, hierarchyMat, mode, method, offset);
            retval.push_back(hierarchyMat);
        } else {
            cv::findContours(image.toMat(), retval, hierarchy.toMat(), mode, method, offset);
        }
    } else {
        cv::findContours(image.toMat(), retval, mode, method, offset);
    }
    return TensorArray(retval);
}

extern "C"
struct TensorWrapper approxPolyDP(
        struct TensorWrapper curve, struct TensorWrapper approxCurve, double epsilon, bool closed)
{
    if (approxCurve.isNull()) {
        cv::Mat retval;
        cv::approxPolyDP(curve.toMat(), retval, epsilon, closed);
        return TensorWrapper(retval);
    } else {
        cv::approxPolyDP(curve.toMat(), approxCurve.toMat(), epsilon, closed);
        return approxCurve;
    }
}

extern "C"
double arcLength(
        struct TensorWrapper curve, bool closed)
{
    return cv::arcLength(curve.toMat(), closed);
}

extern "C"
struct RectWrapper boundingRect(
        struct TensorWrapper points)
{
    return cv::boundingRect(points.toMat());
}

extern "C"
double contourArea(
        struct TensorWrapper contour, bool oriented)
{
    cv::contourArea(contour.toMat(), oriented);
}

extern "C"
struct RotatedRectWrapper minAreaRect(
        struct TensorWrapper points)
{
    return cv::minAreaRect(points.toMat());
}

extern "C"
struct TensorWrapper boxPoints(
        struct RotatedRectWrapper box, struct TensorWrapper points)
{
    if (points.isNull()) {
        cv::Mat retval;
        cv::boxPoints(box, retval);
        return retval;
    } else {
        cv::boxPoints(box, points.toMat());
        return points;
    }
}

extern "C"
struct Vec3fWrapper minEnclosingCircle(
        struct TensorWrapper points, struct Point2fWrapper center, float radius)
{
    Vec3fWrapper retval;
    cv::Point2f temp;

    cv::minEnclosingCircle(points.toMat(), temp, retval.v2);

    retval.v0 = temp.x;
    retval.v1 = temp.y;
    return retval;
}

extern "C"
struct TensorPlusDouble minEnclosingTriangle(
        struct TensorWrapper points, struct TensorWrapper triangle)
{
    TensorPlusDouble retval;
    if (triangle.isNull()) {
        cv::Mat result;
        retval.val = cv::minEnclosingTriangle(points.toMat(), result);
        new (&retval.tensor) TensorWrapper(result);
    } else {
        retval.val = cv::minEnclosingTriangle(points.toMat(), triangle.toMat());
        retval.tensor = triangle;
    }
    return retval;
}

extern "C"
double matchShapes(
        struct TensorWrapper contour1, struct TensorWrapper contour2, int method, double parameter)
{
    return cv::matchShapes(contour1.toMat(), contour2.toMat(), method, parameter);
}

extern "C"
struct TensorWrapper convexHull(
        struct TensorWrapper points, bool clockwise, bool returnPoints)
{
    cv::Mat retval;
    cv::convexHull(points.toMat(), retval, clockwise, returnPoints);
    return TensorWrapper(retval);
}

extern "C"
struct TensorWrapper convexityDefects(
        struct TensorWrapper contour, struct TensorWrapper convexhull)
{
    cv::Mat retval;
    cv::convexityDefects(contour.toMat(), convexhull.toMat(), retval);
    return TensorWrapper(retval);
}

extern "C"
bool isContourConvex(
        struct TensorWrapper contour)
{
    return cv::isContourConvex(contour.toMat());
}

extern "C"
struct TensorPlusFloat intersectConvexConvex(
        struct TensorWrapper _p1, struct TensorWrapper _p2, bool handleNested)
{
    TensorPlusFloat retval;
    cv::Mat result;
    retval.val = cv::intersectConvexConvex(_p1.toMat(), _p2.toMat(), result, handleNested);
    new (&retval.tensor) TensorWrapper(result);
    return retval;
}

extern "C"
struct RotatedRectWrapper fitEllipse(
        struct TensorWrapper points)
{
    return cv::fitEllipse(points.toMat());
}

extern "C"
struct TensorWrapper fitLine(
        struct TensorWrapper points, int distType, double param, double reps, double aeps)
{
    cv::Mat retval;
    cv::fitLine(points.toMat(), retval, distType, param, reps, aeps);
    return TensorWrapper(retval);
}

extern "C"
double pointPolygonTest(
        struct TensorWrapper contour, struct Point2fWrapper pt, bool measureDist)
{
    return cv::pointPolygonTest(contour.toMat(), pt, measureDist);
}

extern "C"
struct TensorWrapper rotatedRectangleIntersection(
        struct RotatedRectWrapper rect1, struct RotatedRectWrapper rect2)
{
    cv::Mat retval;
    cv::rotatedRectangleIntersection(rect1, rect2, retval);
    return TensorWrapper(retval);
}

extern "C"
struct TensorWrapper blendLinear(
        struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper weights1, struct TensorWrapper weights2, struct TensorWrapper dst)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::blendLinear(src1.toMat(), src2.toMat(), weights1.toMat(), weights2.toMat(), retval);
        return TensorWrapper(retval);
    } else {
        cv::blendLinear(src1.toMat(), src2.toMat(), weights1.toMat(), weights2.toMat(), dst.toMat());
        return dst;
    }
}

extern "C"
struct TensorWrapper applyColorMap(
        struct TensorWrapper src, struct TensorWrapper dst, int colormap)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::applyColorMap(src.toMat(), retval, colormap);
        return TensorWrapper(retval);
    } else {
        cv::applyColorMap(src.toMat(), dst.toMat(), colormap);
        return dst;
    }
}

extern "C"
void line(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2, struct ScalarWrapper color, int thickness, int lineType, int shift)
{
    cv::line(img.toMat(), pt1, pt2, color, thickness, lineType, shift);
}

extern "C"
void arrowedLine(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2, struct ScalarWrapper color, int thickness, int line_type, int shift, double tipLength)
{
    cv::arrowedLine(img.toMat(), pt1, pt2, color, thickness, line_type, shift, tipLength);
}

extern "C"
void rectangle(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2, struct ScalarWrapper color, int thickness, int lineType, int shift)
{
    cv::rectangle(img.toMat(), pt1, pt2, color, thickness, lineType, shift);
}

extern "C"
void rectanglePts(
        struct TensorWrapper img, struct RectWrapper rec, struct ScalarWrapper color, int thickness, int lineType, int shift)
{
    cv::Mat imgMat(img);
    cv::rectangle(imgMat, rec, color, thickness, lineType, shift);
}

extern "C"
void circle(
        struct TensorWrapper img, struct PointWrapper center, int radius, struct ScalarWrapper color, int thickness, int lineType, int shift)
{
    cv::circle(img.toMat(), center, radius, color, thickness, lineType, shift);
}

extern "C"
void ellipse(
        struct TensorWrapper img, struct PointWrapper center, struct SizeWrapper axes, double angle, double startAngle, double endAngle, struct ScalarWrapper color, int thickness, int lineType, int shift)
{
    cv::ellipse(img.toMat(), center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift);
}

extern "C"
void ellipseFromRect(
        struct TensorWrapper img, struct RotatedRectWrapper box, struct ScalarWrapper color, int thickness, int lineType)
{
    cv::ellipse(img.toMat(), box, color, thickness, lineType);
}

extern "C"
void fillConvexPoly(
        struct TensorWrapper img, struct TensorWrapper points, struct ScalarWrapper color, int lineType, int shift)
{
    cv::Mat imgMat(img);
    cv::fillConvexPoly(imgMat, points.toMat(), color, lineType, shift);
}

extern "C"
void fillPoly(
        struct TensorWrapper img, struct TensorArray pts, struct ScalarWrapper color, int lineType, int shift, struct PointWrapper offset)
{
    cv::fillPoly(img.toMat(), pts.toMatList(), color, lineType, shift, offset);
}

extern "C"
void polylines(
        struct TensorWrapper img, struct TensorArray pts, bool isClosed, struct ScalarWrapper color, int thickness, int lineType, int shift)
{
    cv::polylines(img.toMat(), pts.toMatList(), isClosed, color, thickness, lineType, shift);
}

extern "C"
void drawContours(
        struct TensorWrapper image, struct TensorArray contours, int contourIdx, struct ScalarWrapper color, int thickness, int lineType, struct TensorWrapper hierarchy, int maxLevel, struct PointWrapper offset)
{
    cv::drawContours(image.toMat(), contours.toMatList(), contourIdx, color, thickness, lineType, TO_MAT_OR_NOARRAY(hierarchy), maxLevel, offset);
}

extern "C"
struct ScalarPlusBool clipLineSize(
        struct SizeWrapper imgSize, struct PointWrapper pt1, struct PointWrapper pt2)
{
    ScalarPlusBool retval;
    cv::Point temp1(pt1), temp2(pt2);
    retval.val = cv::clipLine(imgSize, temp1, temp2);
    retval.scalar.v0 = temp1.x;
    retval.scalar.v1 = temp1.y;
    retval.scalar.v2 = temp2.x;
    retval.scalar.v3 = temp2.y;
    return retval;
}

extern "C"
struct ScalarPlusBool clipLineRect(
        struct RectWrapper imgRect, struct PointWrapper pt1, struct PointWrapper pt2)
{
    ScalarPlusBool retval;
    cv::Point temp1(pt1), temp2(pt2);
    retval.val = cv::clipLine(imgRect, temp1, temp2);
    retval.scalar.v0 = temp1.x;
    retval.scalar.v1 = temp1.y;
    retval.scalar.v2 = temp2.x;
    retval.scalar.v3 = temp2.y;
    return retval;
}

extern "C"
struct TensorWrapper ellipse2Poly(
        struct PointWrapper center, struct SizeWrapper axes, int angle, int arcStart, int arcEnd, int delta)
{
    std::vector<cv::Point> result;
    cv::ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta, result);
    return TensorWrapper(cv::Mat(result));
}

extern "C"
void putText(
        struct TensorWrapper img, const char *text, struct PointWrapper org, int fontFace, double fontScale, struct ScalarWrapper color, int thickness, int lineType, bool bottomLeftOrigin)
{
    cv::putText(img.toMat(), text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin);
}

extern "C"
struct SizePlusInt getTextSize(
        const char *text, int fontFace, double fontScale, int thickness)
{
    SizePlusInt retval;
    retval.size = cv::getTextSize(text, fontFace, fontScale, thickness, &retval.val);
    return retval;
}

/****************** Algorithms ******************/

// GeneralizedHough

struct GeneralizedHoughPtr {
    void *ptr;

    inline cv::GeneralizedHough * operator->() { return static_cast<cv::GeneralizedHough *>(ptr); }
    inline GeneralizedHoughPtr(cv::GeneralizedHough *ptr) { this->ptr = ptr; }
};

extern "C"
void GeneralizedHough_setTemplate(
        GeneralizedHoughPtr ptr, struct TensorWrapper templ, struct PointWrapper templCenter)
{
    ptr->setTemplate(templ.toMat(), templCenter);
}

extern "C"
void GeneralizedHough_setTemplate_edges(
        GeneralizedHoughPtr ptr, struct TensorWrapper edges, struct TensorWrapper dx,
        struct TensorWrapper dy, struct PointWrapper templCenter)
{
    ptr->setTemplate(edges.toMat(), dx.toMat(), dy.toMat(), templCenter);
}

extern "C"
struct TensorArray GeneralizedHough_detect(
        GeneralizedHoughPtr ptr, struct TensorWrapper image, struct TensorWrapper positions, bool votes)
{
    std::vector<cv::Mat> retval(1 + votes);
    if (!positions.isNull()) retval[0] = positions;
    ptr->detect(image.toMat(), retval[0], votes ? retval[1] : cv::noArray());
    return TensorArray(retval);
}

extern "C"
struct TensorArray GeneralizedHough_detect_edges(
        GeneralizedHoughPtr ptr, struct TensorWrapper edges, struct TensorWrapper dx,
        struct TensorWrapper dy, struct TensorWrapper positions, bool votes)
{
    std::vector<cv::Mat> retval(1 + votes);
    if (!positions.isNull()) retval[0] = positions;
    ptr->detect(edges.toMat(), dx.toMat(), dy.toMat(), retval[0], votes ? retval[1] : cv::noArray());
    return TensorArray(retval);
}

extern "C"
void GeneralizedHough_setCannyLowThresh(GeneralizedHoughPtr ptr, int cannyLowThresh)
{
    ptr->setCannyLowThresh(cannyLowThresh);
}

extern "C"
int GeneralizedHough_getCannyLowThresh(GeneralizedHoughPtr ptr)
{
    return ptr->getCannyLowThresh();
}

extern "C"
void GeneralizedHough_setCannyHighThresh(GeneralizedHoughPtr ptr, int cannyHighThresh)
{
    ptr->setCannyHighThresh(cannyHighThresh);
}

extern "C"
int GeneralizedHough_getCannyHighThresh(GeneralizedHoughPtr ptr)
{
    return ptr->getCannyHighThresh();
}

extern "C"
void GeneralizedHough_setMinDist(GeneralizedHoughPtr ptr, double MinDist)
{
    ptr->setMinDist(MinDist);
}

extern "C"
double GeneralizedHough_getMinDist(GeneralizedHoughPtr ptr)
{
    return ptr->getMinDist();
}

extern "C"
void GeneralizedHough_setDp(GeneralizedHoughPtr ptr, double Dp)
{
    ptr->setDp(Dp);
}

extern "C"
double GeneralizedHough_getDp(GeneralizedHoughPtr ptr)
{
    return ptr->getDp();
}

extern "C"
void GeneralizedHough_setMaxBufferSize(GeneralizedHoughPtr ptr, int MaxBufferSize)
{
    ptr->setMaxBufferSize(MaxBufferSize);
}

extern "C"
int GeneralizedHough_getMaxBufferSize(GeneralizedHoughPtr ptr)
{
    return ptr->getMaxBufferSize();
}

// GeneralizedHoughBallard

struct GeneralizedHoughBallardPtr {
    void *ptr;

    inline cv::GeneralizedHoughBallard * operator->() { return static_cast<cv::GeneralizedHoughBallard *>(ptr); }
    inline GeneralizedHoughBallardPtr(cv::GeneralizedHoughBallard *ptr) { this->ptr = ptr; }
};

extern "C"
struct GeneralizedHoughBallardPtr GeneralizedHoughBallard_ctor() {
    return cv::createGeneralizedHoughBallard().get();
}

extern "C"
void GeneralizedHoughBallard_setLevels(GeneralizedHoughBallardPtr ptr, double Levels)
{
    ptr->setLevels(Levels);
}

extern "C"
double GeneralizedHoughBallard_getLevels(GeneralizedHoughBallardPtr ptr)
{
    return ptr->getLevels();
}

extern "C"
void GeneralizedHoughBallard_setVotesThreshold(GeneralizedHoughBallardPtr ptr, double votesThreshold)
{
    ptr->setVotesThreshold(votesThreshold);
}

extern "C"
double GeneralizedHoughBallard_getVotesThreshold(GeneralizedHoughBallardPtr ptr)
{
    return ptr->getVotesThreshold();
}

// GeneralizedHoughGuil

struct GeneralizedHoughGuilPtr {
    void *ptr;

    inline cv::GeneralizedHoughGuil * operator->() { return static_cast<cv::GeneralizedHoughGuil *>(ptr); }
    inline GeneralizedHoughGuilPtr(cv::GeneralizedHoughGuil *ptr) { this->ptr = ptr; }
};

extern "C"
struct GeneralizedHoughGuilPtr GeneralizedHoughGuil_ctor() {
    return cv::createGeneralizedHoughGuil().get();
}

extern "C"
void GeneralizedHoughGuil_setLevels(GeneralizedHoughGuilPtr ptr, int levels)
{
    ptr->setLevels(levels);
}

extern "C"
int GeneralizedHoughGuil_getLevels(GeneralizedHoughGuilPtr ptr)
{
    return ptr->getLevels();
}

extern "C"
void GeneralizedHoughGuil_setAngleEpsilon(GeneralizedHoughGuilPtr ptr, double AngleEpsilon)
{
    ptr->setAngleEpsilon(AngleEpsilon);
}

extern "C"
double GeneralizedHoughGuil_getAngleEpsilon(GeneralizedHoughGuilPtr ptr)
{
    return ptr->getAngleEpsilon();
}

extern "C"
void GeneralizedHoughGuil_setMinAngle(GeneralizedHoughGuilPtr ptr, double MinAngle)
{
    ptr->setMinAngle(MinAngle);
}

extern "C"
double GeneralizedHoughGuil_getMinAngle(GeneralizedHoughGuilPtr ptr)
{
    return ptr->getMinAngle();
}

extern "C"
void GeneralizedHoughGuil_setMaxAngle(GeneralizedHoughGuilPtr ptr, double MaxAngle)
{
    ptr->setMaxAngle(MaxAngle);
}

extern "C"
double GeneralizedHoughGuil_getMaxAngle(GeneralizedHoughGuilPtr ptr)
{
    return ptr->getMaxAngle();
}

extern "C"
void GeneralizedHoughGuil_setAngleStep(GeneralizedHoughGuilPtr ptr, double AngleStep)
{
    ptr->setAngleStep(AngleStep);
}

extern "C"
double GeneralizedHoughGuil_getAngleStep(GeneralizedHoughGuilPtr ptr)
{
    return ptr->getAngleStep();
}

extern "C"
void GeneralizedHoughGuil_setAngleThresh(GeneralizedHoughGuilPtr ptr, int AngleThresh)
{
    ptr->setAngleThresh(AngleThresh);
}

extern "C"
int GeneralizedHoughGuil_getAngleThresh(GeneralizedHoughGuilPtr ptr)
{
    return ptr->getAngleThresh();
}

extern "C"
void GeneralizedHoughGuil_setMinScale(GeneralizedHoughGuilPtr ptr, double MinScale)
{
    ptr->setMinScale(MinScale);
}

extern "C"
double GeneralizedHoughGuil_getMinScale(GeneralizedHoughGuilPtr ptr)
{
    return ptr->getMinScale();
}

extern "C"
void GeneralizedHoughGuil_setMaxScale(GeneralizedHoughGuilPtr ptr, double MaxScale)
{
    ptr->setMaxScale(MaxScale);
}

extern "C"
double GeneralizedHoughGuil_getMaxScale(GeneralizedHoughGuilPtr ptr)
{
    return ptr->getMaxScale();
}

extern "C"
void GeneralizedHoughGuil_setScaleStep(GeneralizedHoughGuilPtr ptr, double ScaleStep)
{
    ptr->setScaleStep(ScaleStep);
}

extern "C"
double GeneralizedHoughGuil_getScaleStep(GeneralizedHoughGuilPtr ptr)
{
    return ptr->getScaleStep();
}

extern "C"
void GeneralizedHoughGuil_setScaleThresh(GeneralizedHoughGuilPtr ptr, int ScaleThresh)
{
    ptr->setScaleThresh(ScaleThresh);
}

extern "C"
int GeneralizedHoughGuil_getScaleThresh(GeneralizedHoughGuilPtr ptr)
{
    return ptr->getScaleThresh();
}

extern "C"
void GeneralizedHoughGuil_setPosThresh(GeneralizedHoughGuilPtr ptr, int PosThresh)
{
    ptr->setPosThresh(PosThresh);
}

extern "C"
int GeneralizedHoughGuil_getPosThresh(GeneralizedHoughGuilPtr ptr)
{
    return ptr->getPosThresh();
}

// CLAHE

struct CLAHEPtr {
    void *ptr;

    inline cv::CLAHE * operator->() { return static_cast<cv::CLAHE *>(ptr); }
    inline CLAHEPtr(cv::CLAHE *ptr) { this->ptr = ptr; }
};

extern "C"
struct CLAHEPtr CLAHE_ctor() {
    return cv::createCLAHE().get();
}

extern "C"
void CLAHE_setClipLimit(CLAHEPtr ptr, double ClipLimit)
{
    ptr->setClipLimit(ClipLimit);
}

extern "C"
double CLAHE_getClipLimit(CLAHEPtr ptr)
{
    return ptr->getClipLimit();
}

extern "C"
void CLAHE_setTilesGridSize(CLAHEPtr ptr, struct SizeWrapper TilesGridSize)
{
    ptr->setTilesGridSize(TilesGridSize);
}

extern "C"
struct SizeWrapper CLAHE_getTilesGridSize(CLAHEPtr ptr)
{
    return ptr->getTilesGridSize();
}

extern "C"
void CLAHE_collectGarbage(CLAHEPtr ptr)
{
    ptr->collectGarbage();
}

// LineSegmentDetector

struct LineSegmentDetectorPtr {
    void *ptr;

    inline cv::LineSegmentDetector * operator->() { return static_cast<cv::LineSegmentDetector *>(ptr); }
    inline LineSegmentDetectorPtr(cv::LineSegmentDetector *ptr) { this->ptr = ptr; }
};

extern "C"
struct LineSegmentDetectorPtr LineSegmentDetector_ctor(
        int refine, double scale, double sigma_scale, double quant,
        double ang_th, double log_eps, double density_th, int n_bins)
{
    return cv::createLineSegmentDetector(
            refine, scale, sigma_scale, quant, ang_th, log_eps, density_th, n_bins).get();
}

extern "C"
struct TensorArray LineSegmentDetector_detect(
        struct LineSegmentDetectorPtr ptr, struct TensorWrapper image,
        struct TensorWrapper lines, bool width, bool prec, bool nfa)
{
    std::vector<cv::Mat> retval(1 + width + prec + nfa);
    if (!lines.isNull()) retval[0] = lines;
    ptr->detect(
            image.toMat(), retval[0],
            width ? retval[1] : cv::noArray(),
            prec  ? retval[1 + width] : cv::noArray(),
            nfa   ? retval[1 + width + prec] : cv::noArray());
    return TensorArray(retval);
}

extern "C"
void LineSegmentDetector_drawSegments(
        struct LineSegmentDetectorPtr ptr, struct TensorWrapper image, struct TensorWrapper lines)
{
    ptr->drawSegments(image.toMat(), lines.toMat());
}

extern "C"
int compareSegments(struct LineSegmentDetectorPtr ptr, struct SizeWrapper size, struct TensorWrapper lines1,
                    struct TensorWrapper lines2, struct TensorWrapper image)
{
    return ptr->compareSegments(size, lines1.toMat(), lines2.toMat(), TO_MAT_OR_NOARRAY(image));
}

