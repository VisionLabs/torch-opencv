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
        int borderType, struct ScalarWrapper borderValue);

extern "C" struct TensorWrapper resize(
        struct TensorWrapper src, struct TensorWrapper dst,
        int dsize_x, int dsize_y, double fx, double fy,
        int interpolation);

extern "C" struct TensorWrapper warpAffine(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper M, int dsize_x, int dsize_y,
        int flags, int borderMode, struct ScalarWrapper borderValue);

extern "C" struct TensorWrapper warpPerspective(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper M, int dsize_x, int dsize_y,
        int flags, int borderMode, struct ScalarWrapper borderValue);

extern "C" struct TensorWrapper remap(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper map1, struct TensorWrapper map2,
        int interpolation, int borderMode, struct ScalarWrapper borderValue);

extern "C" struct MultipleTensorWrapper convertMaps(
        struct TensorWrapper map1, struct TensorWrapper map2,
        struct TensorWrapper dstmap1, struct TensorWrapper dstmap2,
        int dstmap1type, bool nninterpolation);

extern "C" struct TensorWrapper getRotationMatrix2D(
        float center_x, float center_y, double angle, double scale);

extern "C" struct TensorWrapper getPerspectiveTransform(
        struct TensorWrapper src, struct TensorWrapper dst);

extern "C" struct TensorWrapper getAffineTransform(
        struct TensorWrapper src, struct TensorWrapper dst);

extern "C" struct TensorWrapper getRectSubPix(
        struct TensorWrapper image, int patchSize_x, int patchsize_y,
        float center_x, float center_y, struct TensorWrapper patch,
        int patchType);

extern "C" struct TensorWrapper logPolar(
        struct TensorWrapper src, struct TensorWrapper dst,
        float center_x, float center_y, double M, int flags);

extern "C" struct TensorWrapper linearPolar(
        struct TensorWrapper src, struct TensorWrapper dst,
        float center_x, float center_y, double maxRadius, int flags);

extern "C" struct TensorWrapper integral(
        struct TensorWrapper src, struct TensorWrapper sum, int sdepth);

extern "C" struct MultipleTensorWrapper integralN(
        struct TensorWrapper src, struct MultipleTensorWrapper sums, int sdepth, int sqdepth);

extern "C" void accumulate(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper mask);

extern "C" void accumulateSquare(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper mask);

extern "C" void accumulateProduct(
        struct TensorWrapper src1, struct TensorWrapper src2,
        struct TensorWrapper dst, struct TensorWrapper mask);

extern "C" void accumulateWeighted(
        struct TensorWrapper src, struct TensorWrapper dst,
        double alpha, struct TensorWrapper mask);

extern "C" struct Vec3dWrapper phaseCorrelate(
        struct TensorWrapper src1, struct TensorWrapper src2,
        struct TensorWrapper window);

extern "C" struct TensorWrapper createHanningWindow(
        struct TensorWrapper dst, int winSize_x, int winSize_y, int type);

extern "C" struct TWPlusDouble threshold(
        struct TensorWrapper src, struct TensorWrapper dst,
        double thresh, double maxval, int type);

extern "C" struct TensorWrapper adaptiveThreshold(
        struct TensorWrapper src, struct TensorWrapper dst,
        double maxValue, int adaptiveMethod, int thresholdType,
        int blockSize, double C);

extern "C" struct TensorWrapper pyrDown(
        struct TensorWrapper src, struct TensorWrapper dst,
        int dstSize_x, int dstSize_y, int borderType);

extern "C" struct TensorWrapper pyrUp(
        struct TensorWrapper src, struct TensorWrapper dst,
        int dstSize_x, int dstSize_y, int borderType);

extern "C" struct MultipleTensorWrapper buildPyramid(
        struct TensorWrapper src, struct MultipleTensorWrapper dst,
        int maxlevel, int borderType);

extern "C" struct TensorWrapper undistort(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper newCameraMatrix);

extern "C" struct MultipleTensorWrapper initUndistortRectifyMap(
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper R, struct TensorWrapper newCameraMatrix,
        int size_x, int size_y, int m1type,
        struct MultipleTensorWrapper maps);

extern "C" struct MTWPlusFloat initWideAngleProjMap(
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        int imageSize_x, int imageSize_y, int destImageWidth,
        int m1type, struct MultipleTensorWrapper maps,
        int projType, double alpha);

extern "C" struct TensorWrapper getDefaultNewCameraMatrix(
        struct TensorWrapper cameraMatrix, int imgsize_x, int imgsize_y, bool centerPrincipalPoint);

extern "C" struct TensorWrapper undistortPoints(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper R, struct TensorWrapper P);

extern "C" struct TensorWrapper calcHist(
        struct MultipleTensorWrapper images,
        struct IntArray channels, struct TensorWrapper mask,
        struct TensorWrapper hist, int dims, struct IntArray histSize,
        struct FloatArrayOfArrays ranges, bool uniform, bool accumulate);

extern "C" struct TensorWrapper calcBackProject(
        struct MultipleTensorWrapper images, int nimages,
        struct IntArray channels, struct TensorWrapper hist,
        struct TensorWrapper backProject, struct FloatArrayOfArrays ranges,
        double scale, bool uniform);

extern "C" double compareHist(
        struct TensorWrapper H1, struct TensorWrapper H2, int method);

extern "C" struct TensorWrapper equalizeHist(
        struct TensorWrapper src, struct TensorWrapper dst);

extern "C" float EMD(
        struct TensorWrapper signature1, struct TensorWrapper signature2,
        int distType, struct TensorWrapper cost,
        struct FloatArray lowerBound, struct TensorWrapper flow);

extern "C" void watershed(
        struct TensorWrapper image, struct TensorWrapper markers);

extern "C" struct TensorWrapper pyrMeanShiftFiltering(
        struct TensorWrapper src, struct TensorWrapper dst,
        double sp, double sr, int maxLevel, TermCriteriaWrapper termcrit);