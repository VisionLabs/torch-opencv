#include <imgproc.hpp>

// Ехали медведи, на велосипеде

extern "C"
struct TensorWrapper getGaussianKernel(int ksize, double sigma, int ktype)
{
    return TensorWrapper(cv::getGaussianKernel(ksize, sigma, ktype));
}

extern "C"
struct TensorArray getDerivKernels(
        int dx, int dy, int ksize, struct TensorWrapper kx,
        struct TensorWrapper ky, bool normalize, int ktype)
{
    std::vector<MatT> output(2);
    output[0] = kx.toMatT();
    output[1] = ky.toMatT();
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
struct TensorWrapper medianBlur(struct TensorWrapper src, int ksize, struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    cv::medianBlur(src.toMat(), dst_mat, ksize);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper GaussianBlur(struct TensorWrapper src, struct SizeWrapper ksize,
                                  double sigmaX, struct TensorWrapper dst,
                                  double sigmaY, int borderType)
{
    MatT dst_mat = dst.toMatT();
    cv::GaussianBlur(src.toMat(), dst_mat, ksize, sigmaX, sigmaY, borderType);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper bilateralFilter(struct TensorWrapper src, int d,
                                     double sigmaColor, double sigmaSpace,
                                     struct TensorWrapper dst, int borderType)
{
    MatT dst_mat = dst.toMatT();
    cv::bilateralFilter(
                src.toMat(), dst_mat, d, sigmaColor, sigmaSpace, borderType);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper boxFilter(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct SizeWrapper ksize, struct PointWrapper anchor,
        bool normalize, int borderType)
{
    MatT dst_mat = dst.toMatT();
        cv::boxFilter(
                src.toMat(), dst_mat, ddepth, ksize,
                anchor, normalize, borderType);
    return TensorWrapper(dst_mat);
}


extern "C"
struct TensorWrapper sqrBoxFilter(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct SizeWrapper ksize, struct PointWrapper anchor,
        bool normalize, int borderType)
{
    MatT dst_mat = dst.toMatT();
    cv::sqrBoxFilter(src.toMat(), dst_mat, ddepth, ksize, anchor,
                     normalize, borderType);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper blur(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper ksize, struct PointWrapper anchor, int borderType)
{
    MatT dst_mat = dst.toMatT();
    cv::blur(src.toMat(), dst_mat, ksize, anchor, borderType);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper filter2D(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct TensorWrapper kernel, struct PointWrapper anchor,
        double delta, int borderType)
{
    MatT dst_mat = dst.toMatT();
    cv::filter2D(src.toMat(), dst_mat, ddepth,
                 kernel.toMat(), anchor,
                 delta, borderType);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper sepFilter2D(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct TensorWrapper kernelX,struct TensorWrapper kernelY,
        struct PointWrapper anchor, double delta, int borderType)
{
    MatT dst_mat = dst.toMatT();
    cv::sepFilter2D(src.toMat(), dst.toMat(), ddepth,
                    kernelX.toMat(),kernelY.toMat(),
                    anchor, delta, borderType);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper Sobel(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int dx, int dy, int ksize, double scale, double delta, int borderType)
{
    MatT dst_mat = dst.toMatT();
    cv::Sobel(src.toMat(), dst_mat, ddepth,
              dx, dy, ksize, scale, delta, borderType);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper Scharr(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int dx, int dy, double scale, double delta, int borderType)
{
    MatT dst_mat = dst.toMatT();
    cv::Scharr(src.toMat(), dst_mat, ddepth,
               dx, dy, scale, delta, borderType);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper Laplacian(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int ksize, double scale, double delta, int borderType)
{
    MatT dst_mat = dst.toMatT();
    cv::Laplacian(src.toMat(), dst_mat, ddepth,
                  ksize, scale, delta, borderType);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper Canny(
        struct TensorWrapper image, struct TensorWrapper edges,
        double threshold1, double threshold2, int apertureSize, bool L2gradient)
{
    MatT edges_mat = edges.toMatT();
    cv::Canny(image.toMat(), edges_mat, threshold1, threshold2, apertureSize, L2gradient);
    return TensorWrapper(edges_mat);
}

extern "C"
struct TensorWrapper cornerMinEigenVal(
        struct TensorWrapper src, struct TensorWrapper dst,
        int blockSize, int ksize, int borderType)
{
    MatT dst_mat = dst.toMatT();
    cv::cornerMinEigenVal(src.toMat(), dst_mat, blockSize, ksize, borderType);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper cornerHarris(
        struct TensorWrapper src, struct TensorWrapper dst, int blockSize,
        int ksize, double k, int borderType)
{
    MatT dst_mat = dst.toMatT();
    cv::cornerHarris(src.toMat(), dst_mat, blockSize,
                     ksize, k, borderType);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper cornerEigenValsAndVecs(
        struct TensorWrapper src, struct TensorWrapper dst,
        int blockSize, int ksize, int borderType)
{
    MatT dst_mat = dst.toMatT();
    cv::cornerEigenValsAndVecs(src.toMat(), dst_mat, blockSize, ksize, borderType);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper preCornerDetect(
        struct TensorWrapper src, struct TensorWrapper dst, int ksize, int borderType)
{
    MatT dst_mat = dst.toMatT();
    cv::preCornerDetect(src.toMat(), dst_mat, ksize, borderType);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper HoughLines(
        struct TensorWrapper image,
        double rho, double theta, int threshold, double srn, double stn,
        double min_theta, double max_theta)
{
    MatT lines;
    cv::HoughLines(image.toMat(), lines, rho, theta, threshold, srn, stn,
                   min_theta, max_theta);
    return TensorWrapper(lines);
}

extern "C"
struct TensorWrapper HoughLinesP(
        struct TensorWrapper image, double rho,
        double theta, int threshold, double minLineLength, double maxLineGap)
{
    MatT lines;
    cv::HoughLinesP(image.toMat(), lines, rho,
                    theta, threshold, minLineLength, maxLineGap);
    return TensorWrapper(lines);
}

extern "C"
struct TensorWrapper HoughCircles(
        struct TensorWrapper image,
        int method, double dp, double minDist, double param1, double param2,
        int minRadius, int maxRadius)
{
    MatT circles;
    cv::HoughCircles(image.toMat(), circles, method, dp, minDist, param1, param2,
                     minRadius, maxRadius);
    return TensorWrapper(circles);
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
    MatT corners;
    cv::goodFeaturesToTrack(image.toMat(), corners, maxCorners, qualityLevel, minDistance,
                            TO_MAT_OR_NOARRAY(mask), blockSize, useHarrisDetector, k);
    return TensorWrapper(corners);
}

extern "C"
struct ScalarWrapper morphologyDefaultBorderValue() {
    return cv::morphologyDefaultBorderValue();
}

extern "C"
struct TensorWrapper erode(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper kernel, struct PointWrapper anchor,
        int iterations, int borderType, struct ScalarWrapper borderValue)
{
    MatT dst_mat = dst.toMatT();
    cv::erode(src.toMat(), dst_mat, kernel.toMat(), anchor, iterations,
              borderType, borderValue);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper dilate(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper kernel, struct PointWrapper anchor,
        int iterations, int borderType, struct ScalarWrapper borderValue)
{
    MatT dst_mat = dst.toMatT();
    cv::dilate(src.toMat(), dst_mat, kernel.toMat(), anchor, iterations,
               borderType, borderValue);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper morphologyEx(
        struct TensorWrapper src, struct TensorWrapper dst,
        int op, struct TensorWrapper kernel, struct PointWrapper anchor,
        int iterations, int borderType, struct ScalarWrapper borderValue)
{
    MatT dst_mat = dst.toMatT();
    cv::morphologyEx(src.toMat(), dst_mat, op, kernel.toMat(), anchor, iterations,
                     borderType, borderValue);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper resize(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dsize, double fx, double fy,
        int interpolation)
{
    MatT dst_mat = dst.toMatT();
    cv::resize(src.toMat(), dst_mat, dsize, fx, fy, interpolation);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper warpAffine(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper M, struct SizeWrapper dsize,
        int flags, int borderMode, struct ScalarWrapper borderValue)
{
    MatT dst_mat = dst.toMatT();
    cv::warpAffine(src.toMat(), dst_mat, M.toMat(), dsize, flags, borderMode, borderValue);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper warpPerspective(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper M, struct SizeWrapper dsize,
        int flags, int borderMode, struct ScalarWrapper borderValue)
{
    MatT dst_mat = dst.toMatT();
    cv::warpPerspective(src.toMat(), dst_mat, M.toMat(), dsize, flags, borderMode, borderValue);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper remap(
        struct TensorWrapper src, struct TensorWrapper map1,
        struct TensorWrapper map2, int interpolation, struct TensorWrapper dst,
        int borderMode, struct ScalarWrapper borderValue)
{
    MatT dst_mat = dst.toMatT();
    cv::remap(src.toMat(), dst_mat, map1.toMat(), map2.toMat(),
              interpolation, borderMode, borderValue);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorArray convertMaps(
        struct TensorWrapper map1, struct TensorWrapper map2,
        struct TensorWrapper dstmap1, struct TensorWrapper dstmap2,
        int dstmap1type, bool nninterpolation)
{
    if (dstmap1.isNull() and dstmap2.isNull()) {
        // output to retval
        std::vector<MatT> retval(2);
        cv::convertMaps(map1.toMat(), map2.toMat(), retval[0], retval[1], dstmap1type, nninterpolation);
        return TensorArray(retval);
    }
    if (!dstmap1.isNull() and !dstmap2.isNull()) {
        // try to output to the given Tensors
        cv::convertMaps(map1.toMat(), map2.toMat(), dstmap1.toMat(), dstmap2.toMat(), dstmap1type, nninterpolation);
        return TensorArray();
    }
    THError("convertMaps: please specify either both or none of the dstmaps");
    return TensorArray();
}

extern "C"
struct TensorWrapper getRotationMatrix2D(
        struct Point2fWrapper center, double angle, double scale)
{
    return TensorWrapper(MatT(cv::getRotationMatrix2D(center, angle, scale)));
}

extern "C"
struct TensorWrapper invertAffineTransform(
        struct TensorWrapper M, struct TensorWrapper iM)
{
    MatT iM_mat = iM.toMatT();
    cv::invertAffineTransform(M.toMat(), iM_mat);
    return TensorWrapper(iM_mat);
}

extern "C" struct TensorWrapper getPerspectiveTransform(
        struct TensorWrapper src, struct TensorWrapper dst)
{
    return TensorWrapper(MatT(cv::getPerspectiveTransform(src.toMat(), dst.toMat())));
}

extern "C" struct TensorWrapper getAffineTransform(
        struct TensorWrapper src, struct TensorWrapper dst)
{
    return TensorWrapper(MatT(cv::getAffineTransform(src.toMat(), dst.toMat())));
}

extern "C" struct TensorWrapper getRectSubPix(
        struct TensorWrapper image, struct SizeWrapper patchSize,
        struct Point2fWrapper center, struct TensorWrapper patch, int patchType)
{
    MatT patch_mat = patch.toMatT();
    cv::getRectSubPix(image.toMat(), patchSize,
                      center, patch_mat, patchType);
    return TensorWrapper(patch_mat);
}

extern "C"
struct TensorWrapper logPolar(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct Point2fWrapper center, double M, int flags)
{
    MatT dst_mat = dst.toMatT();
    cv::logPolar(src.toMat(), dst_mat, center, M, flags);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper linearPolar(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct Point2fWrapper center, double maxRadius, int flags)
{
    MatT dst_mat = dst.toMatT();
    cv::linearPolar(src.toMat(), dst_mat, center, maxRadius, flags);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper integral(
        struct TensorWrapper src, struct TensorWrapper sum, int sdepth)
{
    MatT sum_mat = sum.toMatT();
    cv::integral(src.toMat(), sum_mat, sdepth);
    return TensorWrapper(sum_mat);
}

extern "C"
struct TensorArray integralN(
        struct TensorWrapper src, struct TensorArray sums, int sdepth, int sqdepth)
{
    // sums.size == 2 or 3
    std::vector<MatT> retval(sums.size);

    for (short i = 0; i < sums.size; ++i) {
        if (!sums.tensors[i].isNull()) {
            retval[i] = sums.tensors[i].toMatT();
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
    MatT dst_mat = dst.toMatT();
    cv::createHanningWindow(dst_mat, winSize, type);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorPlusDouble threshold(
        struct TensorWrapper src, struct TensorWrapper dst,
        double thresh, double maxval, int type)
{
    TensorPlusDouble retval;
    MatT dst_mat = dst.toMatT();
    retval.val = cv::threshold(src.toMat(), dst_mat, thresh, maxval, type);
    new(&retval.tensor) TensorWrapper(dst_mat);
    return retval;
}

extern "C"
struct TensorWrapper adaptiveThreshold(
        struct TensorWrapper src, struct TensorWrapper dst,
        double maxValue, int adaptiveMethod, int thresholdType,
        int blockSize, double C)
{
    MatT dst_mat = dst.toMatT();
    cv::adaptiveThreshold(
		src.toMat(), dst_mat, maxValue, adaptiveMethod,
		thresholdType, blockSize, C);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper pyrDown(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dstSize, int borderType)
{
    MatT dst_mat = dst.toMatT();
    cv::pyrDown(src.toMat(), dst_mat, dstSize, borderType);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper pyrUp(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dstSize, int borderType)
{
    MatT dst_mat = dst.toMatT();
    cv::pyrUp(src.toMat(), dst_mat, dstSize, borderType);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorArray buildPyramid(
        struct TensorWrapper src, struct TensorArray dst,
        int maxlevel, int borderType)
{
    std::vector<MatT> dst_vec = dst.toMatTList();
    cv::buildPyramid(src.toMat(), dst_vec, maxlevel, borderType);
    return TensorArray(dst_vec);
}

extern "C"
struct TensorWrapper undistort(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper newCameraMatrix)
{
    MatT dst_mat = dst.toMatT();
    cv::undistort(src.toMat(), dst_mat, cameraMatrix.toMat(),
                  TO_MAT_OR_NOARRAY(distCoeffs), TO_MAT_OR_NOARRAY(newCameraMatrix));
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorArray initUndistortRectifyMap(
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper R, struct TensorWrapper newCameraMatrix,
        struct SizeWrapper size, int m1type,
        struct TensorArray maps)
{
    std::vector<MatT> maps_vec(2);
    if(!maps.isNull()) maps_vec = maps.toMatTList();
    cv::initUndistortRectifyMap(
                cameraMatrix.toMat(), TO_MAT_OR_NOARRAY(distCoeffs),
                TO_MAT_OR_NOARRAY(R), newCameraMatrix.toMat(),
                size, m1type, maps_vec[0], maps_vec[1]);
        return TensorArray(maps_vec);
}


extern "C"
struct TensorArrayPlusFloat initWideAngleProjMap(
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct SizeWrapper imageSize, int destImageWidth,
        int m1type, struct TensorArray maps,
        int projType, double alpha)
{
    TensorArrayPlusFloat retval;
    std::vector<MatT> maps_vec(2);
    if(!maps.isNull()) maps_vec = maps.toMatTList();
    retval.val = cv::initWideAngleProjMap(
                cameraMatrix.toMat(), TO_MAT_OR_NOARRAY(distCoeffs),
                imageSize, destImageWidth,
                m1type, maps_vec[0], maps_vec[1], projType, alpha);
     new (&retval.tensors) TensorArray(maps_vec);
     return retval;
}

extern "C"
struct TensorWrapper getDefaultNewCameraMatrix(
        struct TensorWrapper cameraMatrix, struct SizeWrapper imgsize, bool centerPrincipalPoint)
{
    return TensorWrapper(
            cv::getDefaultNewCameraMatrix(
                    cameraMatrix.toMat(), imgsize, centerPrincipalPoint
            ));
}

extern "C"
struct TensorWrapper undistortPoints(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper R, struct TensorWrapper P)
{
    MatT dst_mat = dst.toMatT();
    cv::undistortPoints(
                src.toMat(), dst_mat, cameraMatrix.toMat(),
                TO_MAT_OR_NOARRAY(distCoeffs), TO_MAT_OR_NOARRAY(R),
                TO_MAT_OR_NOARRAY(P));
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper calcHist(
        struct TensorArray images,
        struct TensorWrapper channels, struct TensorWrapper mask,
        struct TensorWrapper hist, int dims, struct TensorWrapper histSize,
        struct TensorWrapper ranges, bool uniform, bool accumulate)
{
    auto imagesVec = images.toMatList();
    MatT hist_mat = hist.toMatT();
    cv::Mat channelsMat = channels.toMat();
    cv::Mat histSizeMat = histSize.toMat();
    cv::Mat rangesMat = ranges.toMat();
    std::vector<float *> rangesVec(rangesMat.rows);
    for (int i = 0; i < rangesVec.size(); ++i) {
        rangesVec[i] = reinterpret_cast<float *>(rangesMat.row(i).data);
    }

    cv::calcHist(
            imagesVec.data(), imagesVec.size(), reinterpret_cast<int*>(channelsMat.data),
            TO_MAT_OR_NOARRAY(mask), hist_mat, dims, reinterpret_cast<int*>(histSizeMat.data),
            const_cast<const float**>(rangesVec.data()), uniform, accumulate);
    return TensorWrapper(hist_mat);
}

extern "C"
struct TensorWrapper calcBackProject(
        struct TensorArray images, int nimages,
        struct TensorWrapper channels, struct TensorWrapper hist,
        struct TensorWrapper backProject, struct TensorWrapper ranges,
        double scale, bool uniform)
{
    auto imagesVec = images.toMatList();
    MatT backProject_mat = backProject.toMatT();

    cv::Mat channelsMat = channels.toMat();
    cv::Mat rangesMat = ranges.toMat();
    std::vector<float *> rangesVec(rangesMat.rows);
    for (int i = 0; i < rangesVec.size(); ++i) {
        rangesVec[i] = reinterpret_cast<float *>(rangesMat.row(i).data);
    }

    cv::calcBackProject(
                imagesVec.data(), nimages, reinterpret_cast<int*>(channelsMat.data), hist.toMat(),
                backProject_mat, const_cast<const float **>(rangesVec.data()), scale, uniform);
    return TensorWrapper(backProject_mat);
}

extern "C"
double compareHist(
        struct TensorWrapper H1, struct TensorWrapper H2, int method)
{
    return cv::compareHist(H1.toMat(), H2.toMat(), method);
}

extern "C"
struct TensorWrapper equalizeHist(
        struct TensorWrapper src, struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    cv::equalizeHist(src.toMat(), dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorPlusFloat EMD(
        struct TensorWrapper signature1, struct TensorWrapper signature2,
        int distType, struct TensorWrapper cost,
        struct TensorWrapper lowerBound, struct TensorWrapper flow)
{
    TensorPlusFloat retval;
    MatT flow_mat = flow.toMatT();

    cv::Mat lowerBoundMat = lowerBound.toMat();

    retval.val = cv::EMD(
            signature1.toMat(), signature2.toMat(), distType, TO_MAT_OR_NOARRAY(cost),
            reinterpret_cast<float *>(lowerBoundMat.data), flow_mat);

    new(&retval.tensor) TensorWrapper(flow_mat);
    return retval;
}

extern "C"
void watershed(
        struct TensorWrapper image, struct TensorWrapper markers)
{
    cv::watershed(image.toMat(), markers.toMat());
}

extern "C"
struct TensorWrapper pyrMeanShiftFiltering(
        struct TensorWrapper src, struct TensorWrapper dst,
        double sp, double sr, int maxLevel, TermCriteriaWrapper termcrit)
{
    MatT dst_mat = dst.toMatT();
    cv::pyrMeanShiftFiltering(src.toMat(), dst_mat, sp, sr, maxLevel, termcrit);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorArray grabCut(
        struct TensorWrapper img, struct TensorWrapper mask,
        struct RectWrapper rect, struct TensorWrapper bgdModel,
        struct TensorWrapper fgdModel, int iterCount, int mode)
{
    std::vector<MatT> retval(3);
    retval[0] = mask.toMatT();
    retval[1] = bgdModel.toMatT();
    retval[2] = fgdModel.toMatT();

    cv::grabCut(
            img.toMat(), retval[0], rect, retval[1],
            retval[2], iterCount, mode);

    return TensorArray(retval);
}

extern "C"
struct TensorWrapper distanceTransform(
        struct TensorWrapper src, struct TensorWrapper dst,
        int distanceType, int maskSize, int dstType)
{
    MatT dst_mat = dst.toMatT();
    cv::distanceTransform(
                src.toMat(), dst_mat, distanceType, maskSize, dstType);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorArray distanceTransformWithLabels(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper labels, int distanceType, int maskSize,
        int labelType)
{
    std::vector<MatT> retval(2);
    retval[0] = dst.toMatT();
    retval[1] = labels.toMatT();
    cv::distanceTransform(
            src.toMat(), retval[0], retval[1],
            distanceType, maskSize, labelType);
    return TensorArray(retval);
}

extern "C"
struct RectPlusInt floodFill(
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

extern "C"
struct TensorWrapper cvtColor(
        struct TensorWrapper src, struct TensorWrapper dst, int code, int dstCn)
{
    MatT dst_mat = dst.toMatT();
    cv::cvtColor(src.toMat(), dst_mat, code, dstCn);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper demosaicing(
        struct TensorWrapper _src, struct TensorWrapper _dst, int code, int dcn)
{
    MatT dst_mat = _dst.toMatT();
        cv::demosaicing(_src.toMat(), dst_mat, code, dcn);
    return TensorWrapper(dst_mat);
}

extern "C"
struct MomentsWrapper moments(
        struct TensorWrapper array, bool binaryImage)
{
    return cv::moments(array.toMat(), binaryImage);
}

extern "C"
struct TensorWrapper HuMoments(struct MomentsWrapper m)
{
    cv::Mat retval;
    cv::HuMoments(m, retval);
    return TensorWrapper(retval);
}

extern "C"
struct TensorWrapper matchTemplate(
        struct TensorWrapper image, struct TensorWrapper templ, struct TensorWrapper result, int method, struct TensorWrapper mask)
{
    MatT result_mat = result.toMatT();
    cv::matchTemplate(image.toMat(), templ.toMat(), result_mat, method, TO_MAT_OR_NOARRAY(mask));
    return TensorWrapper(result_mat);
}

extern "C"
struct TensorPlusInt connectedComponents(
        struct TensorWrapper image, struct TensorWrapper labels, int connectivity, int ltype)
{
    TensorPlusInt retval;
    MatT labels_mat = labels.toMatT();
    retval.val = cv::connectedComponents(image.toMat(), labels_mat, connectivity, ltype);
    new(&retval.tensor) TensorWrapper(labels_mat);
    return retval;
}

extern "C"
struct TensorArrayPlusInt connectedComponentsWithStats(
        struct TensorWrapper image, struct TensorArray outputTensors, int connectivity, int ltype)
{
    TensorArrayPlusInt retval;
    std::vector<MatT> output(3);
    if(!outputTensors.isNull()) output = outputTensors.toMatTList();
    retval.val = cv::connectedComponentsWithStats(
                        image.toMat(), output[0], output[1], output[2], connectivity, ltype);
    new(&retval.tensors) TensorArray(output);
    return retval;
}

extern "C"
struct TensorArray findContours(
        struct TensorWrapper image, bool withHierarchy,
        int mode, int method, struct PointWrapper offset)
{
    std::vector<cv::Mat> retval;

    if (withHierarchy) {
        cv::Mat hierarchy;
        cv::findContours(image.toMat(), retval, hierarchy, mode, method, offset);

        retval.push_back(hierarchy);
    } else {
        cv::findContours(image.toMat(), retval, mode, method, offset);
    }
    return TensorArray(retval);
}

extern "C"
struct TensorWrapper approxPolyDP(
        struct TensorWrapper curve, struct TensorWrapper approxCurve, double epsilon, bool closed)
{
    MatT approxCurve_mat = approxCurve.toMatT();
    cv::approxPolyDP(curve.toMat(), approxCurve_mat, epsilon, closed);
    return TensorWrapper(approxCurve_mat);
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
    return cv::contourArea(contour.toMat(), oriented);
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
    MatT points_mat = points.toMatT();
    cv::boxPoints(box, points_mat);
    return TensorWrapper(points_mat);
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
    MatT triangle_mat = triangle.toMatT();
    retval.val = cv::minEnclosingTriangle(points.toMat(), triangle_mat);
    new(&retval.tensor) TensorWrapper(triangle_mat);
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
        struct TensorWrapper points, struct TensorWrapper hull,
        bool clockwise, bool returnPoints)
{
    MatT hull_mat = hull.toMatT();
    cv::convexHull(points.toMat(), hull_mat, clockwise, returnPoints);
    return TensorWrapper(hull_mat);
}

extern "C"
struct TensorWrapper convexityDefects(
        struct TensorWrapper contour, struct TensorWrapper convexhull,
        struct TensorWrapper convexityDefects)
{
    MatT convexityDefects_mat = convexityDefects.toMatT();
    cv::convexityDefects(contour.toMat(), convexhull.toMat(), convexityDefects_mat);
    return TensorWrapper(convexityDefects_mat);
}

extern "C"
bool isContourConvex(
        struct TensorWrapper contour)
{
    return cv::isContourConvex(contour.toMat());
}

extern "C"
struct TensorPlusFloat intersectConvexConvex(
        struct TensorWrapper _p1, struct TensorWrapper _p2,
        struct TensorWrapper _p12, bool handleNested)
{
    TensorPlusFloat retval;
    MatT _p12_mat = _p12.toMatT();
    retval.val = cv::intersectConvexConvex(_p1.toMat(), _p2.toMat(), _p12_mat, handleNested);
    new (&retval.tensor) TensorWrapper(_p12_mat);
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
        struct TensorWrapper points, struct TensorWrapper line, int distType,
        double param, double reps, double aeps)
{
    MatT line_mat = line.toMatT();
    cv::fitLine(points.toMat(), line_mat, distType, param, reps, aeps);
    return TensorWrapper(line_mat);
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
    MatT intersectingRegion;
    cv::rotatedRectangleIntersection(rect1, rect2, intersectingRegion);
    return TensorWrapper(intersectingRegion);
}

extern "C"
struct TensorWrapper blendLinear(
        struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper weights1, struct TensorWrapper weights2, struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    cv::blendLinear(src1.toMat(), src2.toMat(), weights1.toMat(), weights2.toMat(), dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper applyColorMap(
        struct TensorWrapper src, struct TensorWrapper dst, int colormap)
{
    MatT dst_mat = dst.toMatT();
    cv::applyColorMap(src.toMat(), dst_mat, colormap);
        return TensorWrapper(dst_mat);
}

extern "C"
void line(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2,
        struct ScalarWrapper color, int thickness, int lineType, int shift)
{
    cv::line(img.toMat(), pt1, pt2, color, thickness, lineType, shift);
}

extern "C"
void arrowedLine(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2,
        struct ScalarWrapper color, int thickness, int line_type, int shift, double tipLength)
{
    cv::arrowedLine(img.toMat(), pt1, pt2, color, thickness, line_type, shift, tipLength);
}

extern "C"
void rectangle(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2,
        struct ScalarWrapper color, int thickness, int lineType, int shift)
{
    cv::rectangle(img.toMat(), pt1, pt2, color, thickness, lineType, shift);
}

extern "C"
void rectangle2(
        struct TensorWrapper img, struct RectWrapper rec,
        struct ScalarWrapper color, int thickness, int lineType, int shift)
{
    cv::Mat img_mat(img);
    cv::rectangle(img_mat, rec, color, thickness, lineType, shift);
}

extern "C"
void circle(
        struct TensorWrapper img, struct PointWrapper center, int radius,
        struct ScalarWrapper color, int thickness, int lineType, int shift)
{
    cv::circle(img.toMat(), center, radius, color, thickness, lineType, shift);
}

extern "C"
void ellipse(
        struct TensorWrapper img, struct PointWrapper center, struct SizeWrapper axes,
        double angle, double startAngle, double endAngle, struct ScalarWrapper color,
        int thickness, int lineType, int shift)
{
    cv::ellipse(img.toMat(), center, axes, angle, startAngle,
                endAngle, color, thickness, lineType, shift);
}

extern "C"
void ellipseFromRect(
        struct TensorWrapper img, struct RotatedRectWrapper box, struct ScalarWrapper color,
        int thickness, int lineType)
{
    cv::ellipse(img.toMat(), box, color, thickness, lineType);
}

extern "C"
void fillConvexPoly(
        struct TensorWrapper img, struct TensorWrapper points,
        struct ScalarWrapper color, int lineType, int shift)
{
    cv::Mat img_mat(img);
    cv::fillConvexPoly(img_mat, points.toMat(), color, lineType, shift);
}

extern "C"
void fillPoly(
        struct TensorWrapper img, struct TensorArray pts, struct ScalarWrapper color,
        int lineType, int shift, struct PointWrapper offset)
{
    cv::fillPoly(img.toMat(), pts.toMatList(), color, lineType, shift, offset);
}

extern "C"
void polylines(
        struct TensorWrapper img, struct TensorArray pts, bool isClosed,
        struct ScalarWrapper color, int thickness, int lineType, int shift)
{
    cv::polylines(img.toMat(), pts.toMatList(), isClosed, color, thickness, lineType, shift);
}

extern "C"
void drawContours(
        struct TensorWrapper image, struct TensorArray contours, int contourIdx,
        struct ScalarWrapper color, int thickness, int lineType, struct TensorWrapper hierarchy,
        int maxLevel, struct PointWrapper offset)
{
    cv::drawContours(image.toMat(), contours.toMatList(), contourIdx, color, thickness,
                     lineType, TO_MAT_OR_NOARRAY(hierarchy), maxLevel, offset);
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
        struct PointWrapper center, struct SizeWrapper axes,
        int angle, int arcStart, int arcEnd, int delta)
{
    std::vector<cv::Point> result;
    cv::ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta, result);
    return TensorWrapper(cv::Mat(result, true));
}

extern "C"
void putText(
        struct TensorWrapper img, const char *text, struct PointWrapper org,
        int fontFace, double fontScale, struct ScalarWrapper color,
        int thickness, int lineType, bool bottomLeftOrigin)
{
    cv::putText(img.toMat(), text, org, fontFace, fontScale, color,
                thickness, lineType, bottomLeftOrigin);
}

extern "C"
struct SizePlusInt getTextSize(
        const char *text, int fontFace, double fontScale, int thickness)
{
    SizePlusInt retval;
    retval.size = cv::getTextSize(text, fontFace, fontScale, thickness, &retval.val);
    return retval;
}

/****************** Classes ******************/

// GeneralizedHough

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
    std::vector<MatT> retval(1 + votes);
    retval[0] = positions.toMatT();
    if(votes) ptr->detect(image.toMat(), retval[0], retval[1]);
    else ptr->detect(image.toMat(), retval[0], cv::noArray());
    return TensorArray(retval);
}

extern "C"
struct TensorArray GeneralizedHough_detect_edges(
        GeneralizedHoughPtr ptr, struct TensorWrapper edges, struct TensorWrapper dx,
        struct TensorWrapper dy, struct TensorWrapper positions, bool votes)
{
    std::vector<MatT> retval(1 + votes);
    retval[0] = positions.toMatT();
    if(votes) ptr->detect(edges.toMat(), dx.toMat(), dy.toMat(), retval[0], retval[1]);
    else ptr->detect(edges.toMat(), dx.toMat(), dy.toMat(), retval[0], cv::noArray());
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

extern "C"
struct GeneralizedHoughBallardPtr GeneralizedHoughBallard_ctor() {
    return rescueObjectFromPtr(cv::createGeneralizedHoughBallard());
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

extern "C"
struct GeneralizedHoughGuilPtr GeneralizedHoughGuil_ctor() {
    return rescueObjectFromPtr(cv::createGeneralizedHoughGuil());
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

extern "C"
struct CLAHEPtr CLAHE_ctor()
{
    return rescueObjectFromPtr(cv::createCLAHE());
}

extern "C"
struct TensorWrapper CLAHE_apply(CLAHEPtr ptr, struct TensorWrapper src, struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    ptr->apply(src.toMat(), dst_mat);
    return TensorWrapper(dst_mat);
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

extern "C"
struct LineSegmentDetectorPtr LineSegmentDetector_ctor(
        int refine, double scale, double sigma_scale, double quant,
        double ang_th, double log_eps, double density_th, int n_bins)
{
    return rescueObjectFromPtr(cv::createLineSegmentDetector(
            refine, scale, sigma_scale, quant, ang_th, log_eps, density_th, n_bins));
}

extern "C"
struct TensorArray LineSegmentDetector_detect(
        struct LineSegmentDetectorPtr ptr, struct TensorWrapper image,
        struct TensorWrapper lines, bool width, bool prec, bool nfa)
{
    std::vector<cv::Mat> retval(1 + width + prec + nfa);
    retval[0] = lines.toMat();
    ptr->detect(
            image.toMat(), retval[0],
            width ? retval[1] : cv::noArray(),
            prec  ? retval[1 + width] : cv::noArray(),
            nfa   ? retval[1 + width + prec] : cv::noArray());
    std::vector<MatT> result(1 + width + prec + nfa);
    result[0] = MatT(retval[0]);
    if(width) result[1] = MatT(retval[1]);
    if(prec) result[1 + width] = MatT(retval[1 + width]);
    if(nfa) result[1 + width + prec] = MatT(retval[1 + width + prec]);
    return TensorArray(result);
}

extern "C"
struct TensorWrapper LineSegmentDetector_drawSegments(
        struct LineSegmentDetectorPtr ptr, struct TensorWrapper image, struct TensorWrapper lines)
{
    // TODO are we able not to clone this?
    cv::Mat retval = image.toMat().clone();
    ptr->drawSegments(retval, lines.toMat());
    return TensorWrapper(retval);
}

extern "C"
int LineSegmentDetector_compareSegments(
	struct LineSegmentDetectorPtr ptr, struct SizeWrapper size,
	struct TensorWrapper lines1, struct TensorWrapper lines2,
	struct TensorWrapper image)
{
    return ptr->compareSegments(size, lines1.toMat(), lines2.toMat(), TO_MAT_OR_NOARRAY(image));
}

// Subdiv2D

extern "C"
struct Subdiv2DPtr Subdiv2D_ctor_default() {
    return new cv::Subdiv2D();
}

extern "C"
struct Subdiv2DPtr Subdiv2D_ctor(struct RectWrapper rect) {
    return new cv::Subdiv2D(rect);
}

extern "C"
void Subdiv2D_dtor(struct Subdiv2DPtr ptr) {
    delete static_cast<cv::Subdiv2D *>(ptr.ptr);
}

extern "C"
void Subdiv2D_initDelaunay(struct Subdiv2DPtr ptr, struct RectWrapper rect)
{
    ptr->initDelaunay(rect);
}

extern "C"
int Subdiv2D_insert(struct Subdiv2DPtr ptr, struct Point2fWrapper pt)
{
    return ptr->insert(pt);
}

extern "C"
void Subdiv2D_insert_vector(struct Subdiv2DPtr ptr, struct TensorWrapper ptvec)
{
    ptr->insert(ptvec.toMat());
}

extern "C"
struct Vec3iWrapper Subdiv2D_locate(struct Subdiv2DPtr ptr, struct Point2fWrapper pt)
{
    Vec3iWrapper retval;
    retval.v0 = ptr->locate(pt, retval.v1, retval.v2);
    return retval;
}

extern "C"
struct Point2fPlusInt Subdiv2D_findNearest(struct Subdiv2DPtr ptr, struct Point2fWrapper pt)
{
    Point2fPlusInt retval;
    cv::Point2f temp;
    retval.val = ptr->findNearest(pt, &temp);
    retval.point = temp;
    return retval;
}

extern "C"
struct TensorWrapper Subdiv2D_getEdgeList(struct Subdiv2DPtr ptr)
{
    std::vector<cv::Vec4f> result;
    ptr->getEdgeList(result);
    return TensorWrapper(cv::Mat(result, true));
}

extern "C"
struct TensorWrapper Subdiv2D_getTriangleList(struct Subdiv2DPtr ptr)
{
    std::vector<cv::Vec6f> result;
    ptr->getTriangleList(result);
    return TensorWrapper(cv::Mat(result, true));
}

extern "C"
struct TensorArray Subdiv2D_getVoronoiFacetList(struct Subdiv2DPtr ptr, struct TensorWrapper idx)
{
    std::vector<std::vector<cv::Point2f>> facetList;
    std::vector<cv::Point2f> facetCenters;
    ptr->getVoronoiFacetList(idx.toMat(), facetList, facetCenters);

    std::vector<cv::Mat> retval(facetList.size() + 1);
    for (int i = 0; i < facetList.size(); ++i) {
        new (&retval[i + 1]) cv::Mat(facetList[i]);
    }
    new (&retval[retval.size() - 1]) cv::Mat(facetCenters);

    return TensorArray(retval);
}

extern "C"
struct Point2fPlusInt Subdiv2D_getVertex(struct Subdiv2DPtr ptr, int vertex)
{
    Point2fPlusInt retval;
    retval.point = ptr->getVertex(vertex, &retval.val);
    return retval;
}

extern "C"
int Subdiv2D_getEdge(struct Subdiv2DPtr ptr, int edge, int nextEdgeType)
{
    return ptr->getEdge(edge, nextEdgeType);
}

extern "C"
int Subdiv2D_nextEdge(struct Subdiv2DPtr ptr, int edge)
{
    return ptr->nextEdge(edge);
}

extern "C"
int Subdiv2D_rotateEdge(struct Subdiv2DPtr ptr, int edge, int rotate)
{
    return ptr->rotateEdge(edge, rotate);
}

extern "C"
int Subdiv2D_symEdge(struct Subdiv2DPtr ptr, int edge)
{
    return ptr->symEdge(edge);
}

extern "C"
struct Point2fPlusInt Subdiv2D_edgeOrg(struct Subdiv2DPtr ptr, int edge)
{
    Point2fPlusInt retval;
    cv::Point2f temp;
    retval.val = ptr->edgeOrg(edge, &temp);
    retval.point = temp;
    return retval;
}

extern "C"
struct Point2fPlusInt Subdiv2D_edgeDst(struct Subdiv2DPtr ptr, int edge)
{
    Point2fPlusInt retval;
    cv::Point2f temp;
    retval.val = ptr->edgeDst(edge, &temp);
    retval.point = temp;
    return retval;
}

// LineIterator

extern "C"
struct LineIteratorPtr LineIterator_ctor(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2,
        int connectivity, bool leftToRight)
{
    return new cv::LineIterator(img.toMat(), pt1, pt2, connectivity, leftToRight);
}

extern "C"
void LineIterator_dtor(struct LineIteratorPtr ptr) {
    delete static_cast<cv::LineIterator *>(ptr.ptr);
}

extern "C"
int LineIterator_count(struct LineIteratorPtr ptr)
{
    return ptr->count;
}

extern "C"
struct PointWrapper LineIterator_pos(struct LineIteratorPtr ptr)
{
    return ptr->pos();
}

extern "C"
void LineIterator_incr(struct LineIteratorPtr ptr)
{
    ++(*static_cast<cv::LineIterator *>(ptr.ptr));
}

extern "C"
struct TensorWrapper addWeighted(
        struct TensorWrapper src1, double alpha, struct TensorWrapper src2, double beta,
        double gamma, struct TensorWrapper dst, int dtype)
{
    MatT dstMat = dst.toMatT();
    cv::addWeighted(src1.toMat(), alpha, src2.toMat(), beta, gamma, dstMat, dtype);
    return TensorWrapper(dstMat);
}

extern "C"
struct TensorWrapper flip(
        struct TensorWrapper src, struct TensorWrapper dst,
        int mode)
{
    MatT dst_mat = dst.toMatT();
    cv::flip(src.toMat(), dst_mat, mode);
    return TensorWrapper(dst_mat);
}

extern "C"
struct LandmarkDetectorPtr LandmarkDetector_ctor(const char *path)
{
    auto retval = new dlib::shape_predictor;
    dlib::deserialize(path) >> (*retval);
    return retval;
}

extern "C"
void LandmarkDetector_dtor(struct LandmarkDetectorPtr ptr)
{
    delete static_cast<dlib::shape_predictor*>(ptr.ptr);
}

extern "C"
struct TensorWrapper LandmarkDetector_detect(struct LandmarkDetectorPtr ptr,
    struct TensorWrapper img, struct RectWrapper rect)
{
    dlib::rectangle rectDlib(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height);
    auto landmarks = ptr->operator()(dlib::cv_image<float>(img.toMat()), rectDlib);

    cv::Mat retval(landmarks.num_parts(), 2, CV_32FC1);
    for (int i = 0; i < landmarks.num_parts(); ++i) {
        auto landmark = landmarks.part(i);
        retval.at<float>(i, 0) = landmark.x();
        retval.at<float>(i, 1) = landmark.y();
    }

    return TensorWrapper(retval);
}

extern "C"
struct FaceDetectorPtr FaceDetector_ctor()
{
    dlib::frontal_face_detector *detector = new dlib::frontal_face_detector;
    *detector = dlib::get_frontal_face_detector();
    return detector;
}

extern "C"
void FaceDetector_dtor(struct FaceDetectorPtr ptr)
{
    delete static_cast<dlib::frontal_face_detector*>(ptr.ptr);
}

extern "C"
struct RectArray FaceDetector_detect(struct FaceDetectorPtr ptr, struct TensorWrapper img)
{
    auto dets = ptr->operator()(dlib::cv_image<float>(img.toMat()));
    std::vector<cv::Rect> detsCV(dets.size());
    for (int i = 0; i < dets.size(); ++i) {
        detsCV[i].x = dets[i].left();
        detsCV[i].width = dets[i].right() - dets[i].left();
        detsCV[i].y = dets[i].top();
        detsCV[i].height = dets[i].bottom() - dets[i].top();
    }

    std::cout << dets.size() << std::endl;
    return RectArray(detsCV);
}