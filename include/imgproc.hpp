#include <Common.hpp>
#include <opencv2/imgproc.hpp>

extern "C" struct TensorWrapper getGaussianKernel(
        int ksize, double sigma, int ktype);

extern "C" struct MultipleTensorWrapper getDerivKernels(
        int dx, int dy, int ksize, bool normalize, int ktype);

extern "C" struct TensorWrapper getGaborKernel(
        int ksize_rows, int ksize_cols, double sigma, double theta,
        double lambd, double gamma, double psi, int ktype);

extern "C" struct TensorWrapper medianBlur(
        struct TensorWrapper src, struct TensorWrapper dst, int ksize);

extern "C" struct TensorWrapper GaussianBlur(
        struct TensorWrapper src, struct TensorWrapper dst,
        int ksize_x, int ksize_y, double sigmaX,
        double sigmaY, int borderType);

extern "C" struct TensorWrapper bilateralFilter(
        struct TensorWrapper src, struct TensorWrapper dst, int d,
        double sigmaColor, double sigmaSpace,
        int borderType);

extern "C" struct TensorWrapper boxFilter(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int ksize_x, int ksize_y, int anchor_x, int anchor_y,
        bool normalize, int borderType);

extern "C" struct TensorWrapper sqrBoxFilter(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int ksize_x, int ksize_y, int anchor_x, int anchor_y,
        bool normalize, int borderType);

extern "C" struct TensorWrapper blur(
        struct TensorWrapper src, struct TensorWrapper dst,
        int ksize_x, int ksize_y, int anchor_x, int anchor_y, int borderType);

extern "C" struct TensorWrapper filter2D(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct TensorWrapper kernel, int anchor_x, int anchor_y,
        double delta, int borderType);

extern "C" struct TensorWrapper sepFilter2D(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct TensorWrapper kernelX,struct TensorWrapper kernelY,
        int anchor_x, int anchor_y, double delta, int borderType);

extern "C" struct TensorWrapper Sobel(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int dx, int dy, int ksize, double scale, double delta, int borderType);

extern "C" struct TensorWrapper Scharr(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int dx, int dy, double scale, double delta, int borderType);

extern "C" struct TensorWrapper Laplacian(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int ksize, double scale, double delta, int borderType);

extern "C" struct TensorWrapper Canny(
        struct TensorWrapper image, struct TensorWrapper edges,
        double threshold1, double threshold2, int apertureSize, bool L2gradient);

extern "C" struct TensorWrapper cornerMinEigenVal(
        struct TensorWrapper src, struct TensorWrapper dst,
        int blockSize, int ksize, int borderType);

extern "C" struct TensorWrapper cornerHarris(
        struct TensorWrapper src, struct TensorWrapper dst, int blockSize,
        int ksize, double k, int borderType);

extern "C" struct TensorWrapper cornerEigenValsAndVecs(
        struct TensorWrapper src, struct TensorWrapper dst,
        int blockSize, int ksize, int borderType);

extern "C" struct TensorWrapper preCornerDetect(
        struct TensorWrapper src, struct TensorWrapper dst, int ksize, int borderType);

extern "C" struct TensorWrapper HoughLines(
        struct TensorWrapper image,
        double rho, double theta, int threshold, double srn, double stn,
        double min_theta, double max_theta);

extern "C" struct TensorWrapper HoughLinesP(
        struct TensorWrapper image, double rho,
        double theta, int threshold, double minLineLength, double maxLineGap);

extern "C" struct TensorWrapper HoughCircles(
        struct TensorWrapper image,
        int method, double dp, double minDist, double param1, double param2,
        int minRadius, int maxRadius);

extern "C" void cornerSubPix(
        struct TensorWrapper image, struct TensorWrapper corners,
        int winSize_x, int winSize_y, int zeroZone_x, int zeroZone_y,
        struct TermCriteriaWrapper criteria);

extern "C" struct TensorWrapper goodFeaturesToTrack(
        struct TensorWrapper image,
        int maxCorners, double qualityLevel, double minDistance,
        struct TensorWrapper mask, int blockSize, bool useHarrisDetector, double k);

extern "C" struct TensorWrapper erode(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper kernel, int anchor_x, int anchor_y,
        int iterations, int borderType, struct ScalarWrapper borderValue);

extern "C" struct TensorWrapper dilate(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper kernel, int anchor_x, int anchor_y,
        int iterations, int borderType, struct ScalarWrapper borderValue);

extern "C" struct TensorWrapper morphologyEx(
        struct TensorWrapper src, struct TensorWrapper dst,
        int op, struct TensorWrapper kernel,
        int anchor_x, int anchor_y, int iterations,
        int borderType, ScalarWrapper borderValue);

extern "C" struct TensorWrapper resize(
        struct TensorWrapper src, struct TensorWrapper dst,
        int dsize_x, int dsize_y, double fx, double fy,
        int interpolation);