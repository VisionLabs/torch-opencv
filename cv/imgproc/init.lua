local cv = require 'cv._env'
local ffi = require 'ffi'

ffi.cdef[[

struct TensorWrapper getGaussianKernel(int ksize, double sigma, int ktype);

struct TensorArray getDerivKernels(
        int dx, int dy, int ksize, struct TensorWrapper kx,
        struct TensorWrapper ky, bool normalize, int ktype);

struct TensorWrapper getGaborKernel(struct SizeWrapper ksize, double sigma, double theta,
                                    double lambd, double gamma, double psi, int ktype);

struct TensorWrapper getStructuringElement(int shape, struct SizeWrapper ksize,
                                           struct PointWrapper anchor);

struct TensorWrapper medianBlur(struct TensorWrapper src, int ksize, struct TensorWrapper dst);

struct TensorWrapper GaussianBlur(struct TensorWrapper src, struct SizeWrapper ksize,
                                  double sigmaX, struct TensorWrapper dst,
                                  double sigmaY, int borderType);

struct TensorWrapper bilateralFilter(struct TensorWrapper src, int d,
                                     double sigmaColor, double sigmaSpace,
                                     struct TensorWrapper dst, int borderType);

struct TensorWrapper boxFilter(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct SizeWrapper ksize, struct PointWrapper anchor,
        bool normalize, int borderType);

struct TensorWrapper sqrBoxFilter(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct SizeWrapper ksize, struct PointWrapper anchor,
        bool normalize, int borderType);

struct TensorWrapper blur(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper ksize, struct PointWrapper anchor, int borderType);

struct TensorWrapper filter2D(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct TensorWrapper kernel, struct PointWrapper anchor,
        double delta, int borderType);

struct TensorWrapper sepFilter2D(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct TensorWrapper kernelX,struct TensorWrapper kernelY,
        struct PointWrapper anchor, double delta, int borderType);

struct TensorWrapper Sobel(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int dx, int dy, int ksize, double scale, double delta, int borderType);

struct TensorWrapper Scharr(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int dx, int dy, double scale, double delta, int borderType);

struct TensorWrapper Laplacian(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int ksize, double scale, double delta, int borderType);

struct TensorWrapper Canny(
        struct TensorWrapper image, struct TensorWrapper edges,
        double threshold1, double threshold2, int apertureSize, bool L2gradient);

struct TensorWrapper cornerMinEigenVal(
        struct TensorWrapper src, struct TensorWrapper dst,
        int blockSize, int ksize, int borderType);

struct TensorWrapper cornerHarris(
        struct TensorWrapper src, struct TensorWrapper dst, int blockSize,
        int ksize, double k, int borderType);

struct TensorWrapper cornerEigenValsAndVecs(
        struct TensorWrapper src, struct TensorWrapper dst,
        int blockSize, int ksize, int borderType);

struct TensorWrapper preCornerDetect(
        struct TensorWrapper src, struct TensorWrapper dst, int ksize, int borderType);

struct TensorWrapper HoughLines(
        struct TensorWrapper image,
        double rho, double theta, int threshold, double srn, double stn,
        double min_theta, double max_theta);

struct TensorWrapper HoughLinesP(
        struct TensorWrapper image, double rho,
        double theta, int threshold, double minLineLength, double maxLineGap);

struct TensorWrapper HoughCircles(
        struct TensorWrapper image,
        int method, double dp, double minDist, double param1, double param2,
        int minRadius, int maxRadius);

void cornerSubPix(
        struct TensorWrapper image, struct TensorWrapper corners,
        struct SizeWrapper winSize, struct SizeWrapper zeroZone,
        struct TermCriteriaWrapper criteria);

struct TensorWrapper goodFeaturesToTrack(
        struct TensorWrapper image,
        int maxCorners, double qualityLevel, double minDistance,
        struct TensorWrapper mask, int blockSize, bool useHarrisDetector, double k);

struct TensorWrapper erode(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper kernel, struct PointWrapper anchor,
        int iterations, int borderType, struct ScalarWrapper borderValue);

struct TensorWrapper dilate(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper kernel, struct PointWrapper anchor,
        int iterations, int borderType, struct ScalarWrapper borderValue);

struct TensorWrapper morphologyEx(
        struct TensorWrapper src, struct TensorWrapper dst,
        int op, struct TensorWrapper kernel, struct PointWrapper anchor,
        int iterations, int borderType, struct ScalarWrapper borderValue);

struct TensorWrapper resize(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dsize, double fx, double fy,
        int interpolation);

struct TensorWrapper warpAffine(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper M, struct SizeWrapper dsize,
        int flags, int borderMode, struct ScalarWrapper borderValue);

struct TensorWrapper warpPerspective(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper M, struct SizeWrapper dsize,
        int flags, int borderMode, struct ScalarWrapper borderValue);

struct TensorWrapper remap(
        struct TensorWrapper src, struct TensorWrapper map1, struct TensorWrapper map2,
        int interpolation, struct TensorWrapper dst,
        int borderMode, struct ScalarWrapper borderValue);

struct TensorArray convertMaps(
        struct TensorWrapper map1, struct TensorWrapper map2,
        struct TensorWrapper dstmap1, struct TensorWrapper dstmap2,
        int dstmap1type, bool nninterpolation);

struct TensorWrapper getRotationMatrix2D(
        struct Point2fWrapper center, double angle, double scale);

struct TensorWrapper invertAffineTransform(
        struct TensorWrapper M, struct TensorWrapper iM);

struct TensorWrapper getPerspectiveTransform(
        struct TensorWrapper src, struct TensorWrapper dst);

struct TensorWrapper getAffineTransform(
        struct TensorWrapper src, struct TensorWrapper dst);

struct TensorWrapper getRectSubPix(
        struct TensorWrapper image, struct SizeWrapper patchSize,
        struct Point2fWrapper center, struct TensorWrapper patch, int patchType);

struct TensorWrapper logPolar(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct Point2fWrapper center, double M, int flags);

struct TensorWrapper linearPolar(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct Point2fWrapper center, double maxRadius, int flags);

struct TensorWrapper integral(
        struct TensorWrapper src, struct TensorWrapper sum, int sdepth);

struct TensorArray integralN(
        struct TensorWrapper src, struct TensorArray sums, int sdepth, int sqdepth);

void accumulate(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper mask);

void accumulateSquare(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper mask);

void accumulateProduct(
        struct TensorWrapper src1, struct TensorWrapper src2,
        struct TensorWrapper dst, struct TensorWrapper mask);

void accumulateWeighted(
        struct TensorWrapper src, struct TensorWrapper dst,
        double alpha, struct TensorWrapper mask);

struct Vec3dWrapper phaseCorrelate(
        struct TensorWrapper src1, struct TensorWrapper src2,
        struct TensorWrapper window);

struct TensorWrapper createHanningWindow(
        struct TensorWrapper dst, struct SizeWrapper winSize, int type);

struct TensorPlusDouble threshold(
        struct TensorWrapper src, struct TensorWrapper dst,
        double thresh, double maxval, int type);

struct TensorWrapper adaptiveThreshold(
        struct TensorWrapper src, struct TensorWrapper dst,
        double maxValue, int adaptiveMethod, int thresholdType,
        int blockSize, double C);

struct TensorWrapper pyrDown(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dstSize, int borderType);

struct TensorWrapper pyrUp(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dstSize, int borderType);

struct TensorArray buildPyramid(
        struct TensorWrapper src, struct TensorArray dst,
        int maxlevel, int borderType);

struct TensorWrapper undistort(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper newCameraMatrix);

struct TensorArray initUndistortRectifyMap(
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper R, struct TensorWrapper newCameraMatrix,
        struct SizeWrapper size, int m1type,
        struct TensorArray maps);

struct TensorArrayPlusFloat initWideAngleProjMap(
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct SizeWrapper imageSize, int destImageWidth,
        int m1type, struct TensorArray maps,
        int projType, double alpha);

struct TensorWrapper getDefaultNewCameraMatrix(
        struct TensorWrapper cameraMatrix, struct SizeWrapper imgsize, bool centerPrincipalPoint);

struct TensorWrapper undistortPoints(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper R, struct TensorWrapper P);

struct TensorWrapper calcHist(
        struct TensorArray images,
        struct TensorWrapper channels, struct TensorWrapper mask,
        struct TensorWrapper hist, int dims, struct TensorWrapper histSize,
        struct TensorWrapper ranges, bool uniform, bool accumulate);

struct TensorWrapper calcBackProject(
        struct TensorArray images, int nimages,
        struct TensorWrapper channels, struct TensorWrapper hist,
        struct TensorWrapper backProject, struct TensorWrapper ranges,
        double scale, bool uniform);

double compareHist(
        struct TensorWrapper H1, struct TensorWrapper H2, int method);

struct TensorWrapper equalizeHist(
        struct TensorWrapper src, struct TensorWrapper dst);

struct TensorPlusFloat EMD(
        struct TensorWrapper signature1, struct TensorWrapper signature2,
        int distType, struct TensorWrapper cost,
        struct TensorWrapper lowerBound, struct TensorWrapper flow);

void watershed(
        struct TensorWrapper image, struct TensorWrapper markers);

struct TensorWrapper pyrMeanShiftFiltering(
        struct TensorWrapper src, struct TensorWrapper dst,
        double sp, double sr, int maxLevel, struct TermCriteriaWrapper termcrit);

void grabCut(
        struct TensorWrapper img, struct TensorWrapper mask,
        struct RectWrapper rect, struct TensorWrapper bgdModel,
        struct TensorWrapper fgdModel, int iterCount, int mode);

struct TensorWrapper distanceTransform(
        struct TensorWrapper src, struct TensorWrapper dst,
        int distanceType, int maskSize, int dstType);

struct TensorArray distanceTransformWithLabels(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper labels, int distanceType, int maskSize,
        int labelType);

struct RectPlusInt floodFill(
        struct TensorWrapper image, struct TensorWrapper mask,
        struct PointWrapper seedPoint, struct ScalarWrapper newVal,
        struct ScalarWrapper loDiff, struct ScalarWrapper upDiff, int flags);

struct TensorWrapper cvtColor(
        struct TensorWrapper src, struct TensorWrapper dst, int code, int dstCn);

struct TensorWrapper demosaicing(
        struct TensorWrapper _src, struct TensorWrapper _dst, int code, int dcn);

struct MomentsWrapper moments(
        struct TensorWrapper array, bool binaryImage);

struct TensorWrapper HuMoments(
        struct MomentsWrapper m);

struct TensorWrapper matchTemplate(
        struct TensorWrapper image, struct TensorWrapper templ, struct TensorWrapper result, int method, struct TensorWrapper mask);

struct TensorPlusInt connectedComponents(
        struct TensorWrapper image, struct TensorWrapper labels, int connectivity, int ltype);

struct TensorArrayPlusInt connectedComponentsWithStats(
        struct TensorWrapper image, struct TensorArray outputTensors, int connectivity, int ltype);

struct TensorArray findContours(
        struct TensorWrapper image, bool withHierarchy, struct TensorWrapper hierarchy, int mode, int method, struct PointWrapper offset);

struct TensorWrapper approxPolyDP(
        struct TensorWrapper curve, struct TensorWrapper approxCurve, double epsilon, bool closed);

double arcLength(
        struct TensorWrapper curve, bool closed);

struct RectWrapper boundingRect(
        struct TensorWrapper points);

double contourArea(
        struct TensorWrapper contour, bool oriented);

struct RotatedRectWrapper minAreaRect(
        struct TensorWrapper points);

struct TensorWrapper boxPoints(
        struct RotatedRectWrapper box, struct TensorWrapper points);

struct Vec3fWrapper minEnclosingCircle(
        struct TensorWrapper points, struct Point2fWrapper center, float radius);

struct TensorPlusDouble minEnclosingTriangle(
        struct TensorWrapper points, struct TensorWrapper triangle);

double matchShapes(
        struct TensorWrapper contour1, struct TensorWrapper contour2, int method, double parameter);

struct TensorWrapper convexHull(
        struct TensorWrapper points, struct TensorWrapper hull,
        bool clockwise, bool returnPoints);

struct TensorWrapper convexityDefects(
        struct TensorWrapper contour, struct TensorWrapper convexhull,
        struct TensorWrapper convexityDefects);

bool isContourConvex(
        struct TensorWrapper contour);

struct TensorPlusFloat intersectConvexConvex(
        struct TensorWrapper _p1, struct TensorWrapper _p2,
        struct TensorWrapper _p12, bool handleNested);

struct RotatedRectWrapper fitEllipse(
        struct TensorWrapper points);

struct TensorWrapper fitLine(
        struct TensorWrapper points, struct TensorWrapper line, int distType,
        double param, double reps, double aeps);

double pointPolygonTest(
        struct TensorWrapper contour, struct Point2fWrapper pt, bool measureDist);

struct TensorWrapper rotatedRectangleIntersection(
        struct RotatedRectWrapper rect1, struct RotatedRectWrapper rect2);

struct TensorWrapper blendLinear(
        struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper weights1, struct TensorWrapper weights2, struct TensorWrapper dst);

struct TensorWrapper applyColorMap(
        struct TensorWrapper src, struct TensorWrapper dst, int colormap);

void line(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2, struct ScalarWrapper color, int thickness, int lineType, int shift);

void arrowedLine(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2, struct ScalarWrapper color, int thickness, int line_type, int shift, double tipLength);

void rectangle(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2, struct ScalarWrapper color, int thickness, int lineType, int shift);

void rectangle2(
        struct TensorWrapper img, struct RectWrapper rec, struct ScalarWrapper color, int thickness, int lineType, int shift);

void circle(
        struct TensorWrapper img, struct PointWrapper center, int radius, struct ScalarWrapper color, int thickness, int lineType, int shift);

void ellipse(
        struct TensorWrapper img, struct PointWrapper center, struct SizeWrapper axes, double angle, double startAngle, double endAngle, struct ScalarWrapper color, int thickness, int lineType, int shift);

void ellipseFromRect(
        struct TensorWrapper img, struct RotatedRectWrapper box, struct ScalarWrapper color, int thickness, int lineType);

void fillConvexPoly(
        struct TensorWrapper img, struct TensorWrapper points, struct ScalarWrapper color, int lineType, int shift);

void fillPoly(
        struct TensorWrapper img, struct TensorArray pts, struct ScalarWrapper color, int lineType, int shift, struct PointWrapper offset);

void polylines(
        struct TensorWrapper img, struct TensorArray pts, bool isClosed, struct ScalarWrapper color, int thickness, int lineType, int shift);

void drawContours(
        struct TensorWrapper image, struct TensorArray contours, int contourIdx, struct ScalarWrapper color, int thickness, int lineType, struct TensorWrapper hierarchy, int maxLevel, struct PointWrapper offset);

struct ScalarPlusBool clipLineSize(
        struct SizeWrapper imgSize, struct PointWrapper pt1, struct PointWrapper pt2);

struct ScalarPlusBool clipLineRect(
        struct RectWrapper imgRect, struct PointWrapper pt1, struct PointWrapper pt2);

void putText(
        struct TensorWrapper img, const char *text, struct PointWrapper org, int fontFace, double fontScale, struct ScalarWrapper color, int thickness, int lineType, bool bottomLeftOrigin);

struct SizePlusInt getTextSize(
        const char *text, int fontFace, double fontScale, int thickness);

struct TensorWrapper addWeighted(
        struct TensorWrapper src1, double alpha, struct TensorWrapper src2, double beta,
        double gamma, struct TensorWrapper dst, int dtype);
]]


local C = ffi.load(cv.libPath('imgproc'))


function cv.getGaussianKernel(t)
    local argRules = {
        {"ksize", required = true},
        {"sigma", required = true},
        {"ktype", default = cv.CV_64F}
    }
    local ksize, sigma, ktype = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(C.getGaussianKernel(ksize, sigma, ktype))
end


function cv.getDerivKernels(t)
    local argRules = {
        {"dx", required = true},
        {"dy", required = true},
        {"ksize", required = true},
        {"kx", default = nil},
        {"ky", default = nil},
        {"ktype", default = cv.CV_32F},
        {"normalize", default = false}
    }
    local dx, dy, ksize, kx, ky, ktype, normalize = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
		C.getDerivKernels(
			dx, dy, ksize, cv.wrap_tensor(kx),
			cv.wrap_tensor(ky), normalize, ktype))
end


function cv.getGaborKernel(t)
    local argRules = {
        {"ksize", required = true, operator = cv.Size},
        {"sigma", required = true},
        {"theta", required = true},
        {"lambd", required = true},
        {"gamma", required = true},
        {"psi", default = math.pi * 0.5},
        {"ktype", default = cv.CV_64F}
    }
    local ksize, sigma, theta, lambd, gamma, psi, ktype = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype))
end


function cv.getStructuringElement(t)
    local argRules = {
        {"shape", required = true},
        {"ksize", required = true, operator = cv.Size},
        {"anchor", default = {-1, -1}, operator = cv.Point}
    }
    local shape, ksize, anchor = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.getStructuringElement(shape, ksize, anchor))
end


function cv.medianBlur(t)
    local argRules = {
        {"src", required = true},
        {"ksize", required = true},
        {"dst", default = nil}}
        
    local src, ksize, dst = cv.argcheck(t, argRules)

    local srcChannels = src:size()[3]
    assert(srcChannels == 1 or srcChannels == 3 or srcChannels == 4)

    local srcType = cv.tensorType(src)
    if ksize == 3 or ksize == 5 then
        assert(srcType == cv.CV_8U or srcType == cv.CV_32F)
    else
        assert(srcType == cv.CV_8U)
    end

    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(C.medianBlur(cv.wrap_tensor(src), ksize, cv.wrap_tensor(dst)))
end


function cv.GaussianBlur(t)
    local argRules = {
        {"src", required = true},
        {"ksize", required = true, operator = cv.Size},
        {"sigmaX", required = true},
        {"dst", default = nil},
        {"sigmaY", default = 0},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, ksize, sigmaX, dst, sigmaY, borderType = cv.argcheck(t, argRules)

    assert(cv.tensorType(src) ~= cv.CV_8S and
           cv.tensorType(src) ~= cv.CV_32S)
    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.GaussianBlur(
            cv.wrap_tensor(src), ksize, sigmaX, cv.wrap_tensor(dst), sigmaY, borderType))
end


function cv.bilateralFilter(t)
    local argRules = {
        {"src", required = true},
        {"d", required = true},
        {"sigmaColor", required = true},
        {"sigmaSpace", required = true},
        {"dst", default = nil},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, d, sigmaColor, sigmaSpace, dst, borderType = cv.argcheck(t, argRules)

    assert(src:nDimension() == 2 or src:size()[3] == 3)

    local srcType = cv.tensorType(src)
    assert(srcType == cv.CV_8U or srcType == cv.CV_32F)

    if dst then
        assert(src ~= dst and dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.bilateralFilter(
            cv.wrap_tensor(src), d, sigmaColor, sigmaSpace, cv.wrap_tensor(dst), borderType))
end


function cv.boxFilter(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"ddepth", required = true},
        {"ksize", required = true, operator = cv.Size},
        {"anchor", default = {-1, -1}, operator = cv.Point},
        {"normalize", default = nil},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, dst, ddepth, ksize, anchor, normalize, borderType = cv.argcheck(t, argRules)
    if normalize == nil then normalize = true end

    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.boxFilter(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), ddepth, ksize[1],
            ksize[2], anchor, normalize, borderType))
end

function cv.sqrBoxFilter(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"ddepth", required = true},
        {"ksize", required = true, operator = cv.Size},
        {"anchor", default = {-1,-1}},
        {"normalize", default = nil},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, dst, ddepth, ksize, anchor, normalize, borderType = cv.argcheck(t, argRules)
    if normalize == nil then normalize = true end

    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.sqrBoxFilter(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), ddepth, ksize, anchor, normalize, borderType))
end


function cv.blur(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"ksize", required = true, operator = cv.Size},
        {"anchor", default = {-1,-1}, operator = cv.Point},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, dst, ksize, anchor, borderType = cv.argcheck(t, argRules)

    assert(cv.tensorType(src) ~= cv.CV_8S and
           cv.tensorType(src) ~= cv.CV_32S)
    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.blur(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), ksize, anchor, borderType))
end


function cv.filter2D(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"ddepth", required = true},
        {"kernel", required = true},
        {"anchor", default = {-1,-1}},
        {"delta", default = 0},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, dst, ddepth, kernel, anchor, delta, borderType = cv.argcheck(t, argRules)

    assert(cv.checkFilterCombination(src, ddepth))
    if dst then
        assert(src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.filter2D(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), ddepth, cv.wrap_tensor(kernel), anchor, delta, borderType))
end


function cv.sepFilter2D(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"ddepth", required = true},
        {"kernelX", required = true},
        {"kernelY", required = true},
        {"anchor", default = {-1,-1}},
        {"delta", default = 0},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, dst, ddepth, kernelX, kernelY, anchor, delta, borderType = cv.argcheck(t, argRules)

    assert(cv.checkFilterCombination(src, ddepth))
    if dst then
        assert(src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.sepFilter2D(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), ddepth, kernelX, kernelY, anchor, delta, borderType))
end


function cv.Sobel(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"ddepth", required = true},
        {"dx", required = true},
        {"dy", required = true},
        {"ksize", default = 3},
        {"scale", default = 1},
        {"delta", default = 0},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, dst, ddepth, dx, dy, ksize, scale, delta, borderType = cv.argcheck(t, argRules)

    assert(cv.checkFilterCombination(src, ddepth))
    if dst then
        assert(src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.Sobel(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), ddepth, dx, dy, ksize, scale, delta, borderType))
end


function cv.Scharr(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"ddepth", required = true},
        {"dx", required = true},
        {"dy", required = true},
        {"scale", default = 1},
        {"delta", default = 0},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, dst, ddepth, dx, dy, scale, delta, borderType = cv.argcheck(t, argRules)

    assert(cv.checkFilterCombination(src, ddepth))
    if dst then
        assert(src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.Scharr(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), ddepth, dx, dy, scale, delta, borderType))
end


function cv.Laplacian(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"ddepth", required = true},
        {"ksize", default = 1},
        {"scale", default = 1},
        {"delta", default = 0},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, dst, ddepth, ksize, scale, delta, borderType = cv.argcheck(t, argRules)

    assert(cv.checkFilterCombination(src, ddepth))
    if dst then
        assert(src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.Laplacian(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), ddepth, ksize, scale, delta, borderType))
end


function cv.Canny(t)
    local argRules = {
        {"image", required = true},
        {"threshold1", required = true},
        {"threshold2", required = true},
        {"edges", default = nil},
        {"apertureSize", default = 3},
        {"L2gradient", default = false}
    }
    local image, threshold1, threshold2, edges, apertureSize, L2gradient = cv.argcheck(t, argRules)

    assert(cv.tensorType(image) == cv.CV_8U)

    if edges then
        assert(edges:nDimension() == 2 and
               edges:size()[1] == image:size()[1] and
               edges:size()[2] == image:size()[2])
    end

    return cv.unwrap_tensors(
        C.Canny(
            cv.wrap_tensor(image), cv.wrap_tensor(edges), threshold1, threshold2, apertureSize, L2gradient))
end


function cv.cornerMinEigenVal(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"blockSize", required = true},
        {"ksize", default = 3},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, dst, blockSize, ksize, borderType = cv.argcheck(t, argRules)

    local srcType = cv.tensorType(src)
    assert(src:nDimension() == 2 and (srcType == cv.CV_8U or srcType == cv.CV_32F))
    if dst then
        assert(dst:isSameSizeAs(src) and cv.tensorType(dst) == cv.CV_32F)
    end

    return cv.unwrap_tensors(
        C.cornerMinEigenVal(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), blockSize, ksize, borderType))
end


function cv.cornerHarris(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"blockSize", required = true},
        {"ksize", required = true},
        {"k", required = true},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, dst, blockSize, ksize, k, borderType = cv.argcheck(t, argRules)

    local srcType = cv.tensorType(src)
    assert(src:nDimension() == 2 and (srcType == cv.CV_8U or srcType == cv.CV_32F))

    if dst then
        assert(dst:isSameSizeAs(src) and cv.tensorType(dst) == cv.CV_32F)
    end

    return cv.unwrap_tensors(
        C.cornerHarris(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), blockSize, ksize, k, borderType))
end


function cv.cornerEigenValsAndVecs(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"blockSize", required = true},
        {"ksize", required = true},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, dst, blockSize, ksize, borderType = cv.argcheck(t, argRules)

    local srcType = cv.tensorType(src)
    assert(src:nDimension() == 2 and (srcType == cv.CV_8U or srcType == cv.CV_32F))

    if dst then
        assert(dst:nDimension() == 3 and
               cv.tensorType(dst) == cv.CV_32F and
               dst:size()[1] == src:size()[1] and
               dst:size()[2] == src:size()[2] and
               dst:size()[3] == 6)
    end

    return cv.unwrap_tensors(
        C.cornerEigenValsAndVecs(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), blockSize, ksize, borderType))
end


function cv.preCornerDetect(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"ksize", required = true},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, dst, ksize, borderType = cv.argcheck(t, argRules)

    local srcType = cv.tensorType(src)
    assert(src:nDimension() == 2 and (srcType == cv.CV_8U or srcType == cv.CV_32F))

    if dst then
        assert(cv.tensorType(dst) == cv.CV_32F and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.preCornerDetect(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), ksize, borderType))
end


function cv.HoughLines(t)
    local argRules = {
        {"image", required = true},
        {"rho", required = true},
        {"theta", required = true},
        {"threshold", required = true},
        {"srn", default = 0},
        {"stn", default = 0},
        {"min_theta", default = 0},
        {"max_theta", default = cv.CV_PI}
    }
    local image, rho, theta, threshold, srn, stn, min_theta, max_theta = cv.argcheck(t, argRules)

    assert(image:nDimension() == 2 and cv.tensorType(image) == cv.CV_8U)

    return cv.unwrap_tensors(
        C.HoughLines(
            cv.wrap_tensor(image), rho, theta, threshold, srn, stn, min_theta, max_theta))
end


function cv.HoughLinesP(t)
    local argRules = {
        {"image", required = true},
        {"rho", required = true},
        {"theta", required = true},
        {"threshold", required = true},
        {"minLineLength", default = 0},
        {"maxLineGap", default = 0}
    }
    local image, rho, theta, threshold, minLineLength, maxLineGap = cv.argcheck(t, argRules)

    assert(image:nDimension() == 2 and cv.tensorType(image) == cv.CV_8U)

    return cv.unwrap_tensors(
        C.HoughLinesP(
            cv.wrap_tensor(image), rho, theta, threshold, minLineLength, maxLineGap))
end


function cv.HoughCircles(t)
    local argRules = {
        {"image", required = true},
        {"method", required = true},
        {"dp", required = true},
        {"minDist", required = true},
        {"param1", default = 100},
        {"param2", default = 100},
        {"minRadius", default = 0},
        {"maxRadius", default = 0}
    }
    local image, method, dp, minDist, param1, param2, minRadius, maxRadius = cv.argcheck(t, argRules)

    assert(image:nDimension() == 2 and cv.tensorType(image) == cv.CV_8U)

    return cv.unwrap_tensors(
        C.HoughCircles(
            cv.wrap_tensor(image), method, dp, minDist, param1, param2, minRadius, maxRadius))
end


function cv.cornerSubPix(t)
    local argRules = {
        {"image", required = true},
        {"corners", required = true},
        {"winSize", required = true, operator = cv.Size},
        {"zeroZone", required = true, operator = cv.Size},
        {"criteria", required = true, operator = cv.TermCriteria}
    }
    local image, corners, winSize, zeroZone, criteria = cv.argcheck(t, argRules)

    C.cornerSubPix(
        cv.wrap_tensor(image), cv.wrap_tensor(corners), winSize,
        zeroZone, criteria)
end


function cv.goodFeaturesToTrack(t)
    local argRules = {
        {"image", required = true},
        {"maxCorners", required = true},
        {"qualityLevel", required = true},
        {"minDistance", required = true},
        {"mask", default = nil},
        {"blockSize", default = 3},
        {"useHarrisDetector", default = false},
        {"k", default = 0.04}
    }
    local image, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k = cv.argcheck(t, argRules)

    local imgType = cv.tensorType(image)
    assert(image:nDimension() == 2 and (imgType == cv.CV_32F or imgType == cv.CV_8U))

    if mask then
        assert(cv.tensorType(mask) == cv.CV_8U and mask:isSameSizeAs(image))
    end

    return cv.unwrap_tensors(
        C.goodFeaturesToTrack(
            cv.wrap_tensor(image), maxCorners, qualityLevel, minDistance, cv.wrap_tensor(mask), blockSize, useHarrisDetector, k))
end


function cv.erode(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"kernel", required = true},
        {"anchor", default = {-1, -1}, operator = cv.Point},
        {"borderType", default = cv.BORDER_CONSTANT},
        {"borderValue", default = {0,0,0,0} , operator = cv.Scalar}
    }
    local src, dst, kernel, anchor, borderType, borderValue = cv.argcheck(t, argRules)
    local iterations = 1

    return cv.unwrap_tensors(
        C.erode(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), cv.wrap_tensor(kernel), anchor,
            iterations, borderType, borderValue))
end


function cv.dilate(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"kernel", required = true},
        {"anchor", default = {-1, -1}, operator = cv.Point},
        {"borderType", default = cv.BORDER_CONSTANT},
        {"borderValue", default = {0,0,0,0} , operator = cv.Scalar}
    }
    local src, dst, kernel, anchor, borderType, borderValue = cv.argcheck(t, argRules)
    local iterations = 1

    return cv.unwrap_tensors(
        C.dilate(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), cv.wrap_tensor(kernel), anchor,
            iterations, borderType, borderValue))
end


function cv.morphologyEx(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"op", required = true},
        {"kernel", required = true},
        {"anchor", default = {-1, -1}, operator = cv.Point},
        {"borderType", default = cv.BORDER_CONSTANT},
        {"borderValue", default = {0,0,0,0} , operator = cv.Scalar}
    }
    local src, dst, op, kernel, anchor, borderType, borderValue = cv.argcheck(t, argRules)
    local iterations = 1

    return cv.unwrap_tensors(
        C.morphologyEx(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), op, cv.wrap_tensor(kernel), anchor,
            iterations, borderType, borderValue))
end


function cv.resize(t)
    local argRules = {
        {"src", required = true},
        {"dsize", default = {0, 0}, operator = cv.Size},
        {"dst", default = nil},
        {"fx", default = 0},
        {"fy", default = 0},
        {"interpolation", default = cv.INTER_LINEAR}
    }
    local src, dsize, dst, fx, fy, interpolation = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.resize(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), dsize, fx, fy, interpolation))
end


function cv.warpAffine(t)
    local argRules = {
        {"src", required = true},
        {"M", required = true},
        {"dsize", default = {0, 0}, operator = cv.Size},
        {"dst", default = nil},
        {"flags", default = cv.INTER_LINEAR},
        {"borderMode", default = cv.BORDER_CONSTANT},
        {"borderValue", default = {0,0,0,0} , operator = cv.Scalar}
    }
    local src, M, dsize, dst, flags, borderMode, borderValue = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.warpAffine(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), cv.wrap_tensor(M), dsize,
            flags, borderMode, borderValue))
end


function cv.warpPerspective(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"M", required = true},
        {"dsize", default = {0, 0}, operator = cv.Size},
        {"flags", default = cv.INTER_LINEAR},
        {"borderMode", default = cv.BORDER_CONSTANT},
        {"borderValue", default = {0,0,0,0} , operator = cv.Scalar}
    }
    local src, dst, M, dsize, flags, borderMode, borderValue = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.warpPerspective(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), cv.wrap_tensor(M), dsize,
            flags, borderMode, borderValue))
end


function cv.remap(t)
    local argRules = {
        {"src", required = true},
        {"map1", required = true},
        {"map2", required = true},
        {"interpolation", required = true},
        {"dst", default = nil},
        {"borderMode", default = cv.BORDER_CONSTANT},
        {"borderValue", default = {0, 0, 0, 0}, operator = cv.Scalar}
    }
    local src, map1, map2, interpolation, dst, borderMode, borderValue = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.remap(
            cv.wrap_tensor(src), cv.wrap_tensor(map1), cv.wrap_tensor(map2),
            interpolation, cv.wrap_tensor(dst), borderMode, borderValue))
end


function cv.convertMaps(t)
    local argRules = {
        {"map1", required = true},
        {"map2", default = nil},
        {"dstmap1", required = true},
        {"dstmap2", required = true},
        {"dstmap1type", required = true},
        {"nninterpolation", default = false}
    }
    local map1, map2, dstmap1, dstmap2, dstmap1type, nninterpolation = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.convertMaps(
            cv.wrap_tensor(map1), cv.wrap_tensor(map2), cv.wrap_tensor(dstmap1),
            cv.wrap_tensor(dstmap2), dstmap1type, nninterpolation))
end


function cv.getRotationMatrix2D(t)
    local argRules = {
        {"center", required = true, operator = cv.Point2f},
        {"angle", required = true},
        {"scale", required = true}
    }
    local center, angle, scale = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.getRotationMatrix2D(center, angle, scale))
end


function cv.invertAffineTransform(t)
    local argRules = {
        {"M", required = true},
        {"iM", default = nil}
    }
    local M, iM = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.invertAffineTransform(cv.wrap_tensor(M), cv.wrap_tensor(iM)))
end


function cv.getPerspectiveTransform(t)
    local argRules = {
        {"src", required = true},
        {"dst", required = true}
    }
    local src, dst = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.getPerspectiveTransform(cv.wrap_tensor(src), cv.wrap_tensor(dst)))
end


function cv.getAffineTransform(t)
    local argRules = {
        {"src", required = true},
        {"dst", required = true}
    }
    local src, dst = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.getAffineTransform(cv.wrap_tensor(src), cv.wrap_tensor(dst)))
end


function cv.getRectSubPix(t)
    local argRules = {
        {"image", required = true},
        {"patchSize", required = true, operator = cv.Size},
        {"center", required = true, operator = cv.Point2f},
        {"patch", default = nil},
        {"patchType", default = -1}
    }
    local image, patchSize, center, patch, patchType = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.getRectSubPix(cv.wrap_tensor(image), patchSize, center,
                        cv.wrap_tensor(patch), patchType))
end


function cv.logPolar(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"center", required = true, operator = cv.Point2f},
        {"M", required = true},
        {"flags", required = true}
    }
    local src, dst, center, M, flags = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.logPolar(cv.wrap_tensor(src), cv.wrap_tensor(dst), center, M, flags))
end


function cv.linearPolar(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"center", required = true, operator = cv.Point2f},
        {"maxRadius", required = true},
        {"flags", required = true}
    }
    local src, dst, center, maxRadius, flags = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.linearPolar(cv.wrap_tensor(src), cv.wrap_tensor(dst), center, maxRadius, flags))
end


function cv.integral(t)
    local argRules = {
        {"src", required = true},
        {"sum", default = nil},
        {"sdepth", default = -1}
    }
    local src, sum, sdepth = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.integral(cv.wrap_tensor(src), cv.wrap_tensor(sum), sdepth))
end


function cv.integral2(t)
    local argRules = {
        {"src", required = true},
        {"sum", default = nil},
        {"sqsum", default = nil},
        {"sdepth", default = -1},
        {"sqdepth", default = -1}
    }
    local src, sum, sqsum, sdepth, sqdepth = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.integralN(cv.wrap_tensor(src), cv.wrap_tensors(sum, sqsum), sdepth, sqdepth))
end


function cv.integral3(t)
    local argRules = {
        {"src", required = true},
        {"sum", default = nil},
        {"sqsum", default = nil},
        {"tilted", default = nil},
        {"sdepth", default = -1},
        {"sqdepth", default = -1}
    }
    local src, sum, sqsum, tilted, sdepth, sqdepth = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.integralN(cv.wrap_tensor(src), cv.wrap_tensors(sum, sqsum, tilted), sdepth, sqdepth))
end


function cv.accumulate(t)
    local argRules = {
        {"src", required = true},
        {"dst", required = true},
        {"mask", default = nil}
    }
    local src, dst, mask = cv.argcheck(t, argRules)

    C.accumulate(cv.wrap_tensor(src), cv.wrap_tensor(dst), cv.wrap_tensor(mask))
end


function cv.accumulateSquare(t)
    local argRules = {
        {"src", required = true},
        {"dst", required = true},
        {"mask", default = nil}
    }
    local src, dst, mask = cv.argcheck(t, argRules)

    C.accumulateSquare(cv.wrap_tensor(src), cv.wrap_tensor(dst), cv.wrap_tensor(mask))
end


function cv.accumulateProduct(t)
    local argRules = {
        {"src1", required = true},
        {"src2", required = true},
        {"dst", required = true},
        {"mask", default = nil}
    }
    local src1, src2, dst, mask = cv.argcheck(t, argRules)

    C.accumulateSquare(cv.wrap_tensor(src1), cv.wrap_tensor(src2), cv.wrap_tensor(dst), cv.wrap_tensor(mask))
end


function cv.accumulateWeighted(t)
    local argRules = {
        {"src", required = true},
        {"dst", required = true},
        {"alpha", required = true},
        {"mask", default = nil}
    }
    local src, dst, alpha, mask = cv.argcheck(t, argRules)

    C.accumulateWeighted(cv.wrap_tensor(src), cv.wrap_tensor(dst), alpha, cv.wrap_tensor(mask))
end


-- point, response = cv.phaseCorrelate{...}
-- point    -> Point
-- response -> number
function cv.phaseCorrelate(t)
    local argRules = {
        {"src1", required = true},
        {"src2", required = true},
        {"window", default = nil}
    }
    local src1, src2, window = cv.argcheck(t, argRules)

    local result = C.phaseCorrelate(cv.wrap_tensor(src1), cv.wrap_tensor(src2), cv.wrap_tensor(window))
    return {x=result.v0, y=result.v1}, result.v2
end


function cv.createHanningWindow(t)
    local argRules = {
        {"dst", default = nil},
        {"winSize", required = true, operator = cv.Size},
        {"type", required = true}
    }
    local dst, winSize, type = cv.argcheck(t, argRules)

    if dst then
        assert(cv.tensorType(dst) == type)
        assert(dst:size()[1] == winSize[1] and
               dst:size()[2] == winSize[2])
    end

    return cv.unwrap_tensors(
        C.createHanningWindow(cv.wrap_tensor(dst), winSize, type))
end


-- value, binarized = cv.threshold{...}
-- value     -> number
-- binarized -> Tensor
function cv.threshold(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"thresh", required = true},
        {"maxval", required = true},
        {"type", required = true}
    }
    local src, dst, thresh, maxval, type = cv.argcheck(t, argRules)

    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    local result = C.threshold(
			cv.wrap_tensor(src), cv.wrap_tensor(dst),
			thresh, maxval, type)
    return result.val, cv.unwrap_tensors(result.tensor)
end


function cv.adaptiveThreshold(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"maxValue", required = true},
        {"adaptiveMethod", required = true},
        {"thresholdType", required = true},
        {"blockSize", required = true},
        {"C", required = true}
    }
    local src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, c = cv.argcheck(t, argRules)

    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.adaptiveThreshold(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), maxValue,
                            adaptiveMethod, thresholdType, blockSize, c))
end


function cv.pyrDown(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"dstSize", default = {0,0}, operator = cv.Size},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, dst, dstSize, borderType = cv.argcheck(t, argRules)

    if dst then
        assert(dst:type() == src:type())
        assert(dst:size()[1] == dstSize[1] and
               dst:size()[2] == dstSize[2])
    end

    return cv.unwrap_tensors(
        C.pyrDown(
            cv.wrap_tensor(src), cv.wrap_tensor(dst),
            dstSize, borderType))
end


function cv.pyrUp(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"dstSize", default = {0,0}, operator = cv.Size},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, dst, dstSize, borderType = cv.argcheck(t, argRules)

    if dst then
        assert(dst:type() == src:type())
        assert(dst:size()[1] == dstSize[1] and
                dst:size()[2] == dstSize[2])
    end

    return cv.unwrap_tensors(
        C.pyrUp(
            cv.wrap_tensor(src), cv.wrap_tensor(dst),
            dstSize, borderType))
end


function cv.buildPyramid(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"maxlevel", required = true},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, dst, maxlevel, borderType = cv.argcheck(t, argRules)

    if dst then
        assert(#dst == maxlevel + 1)
        for i, tensor in ipairs(dst) do
            assert(tensor:type() == src:type())
        end
    end

    return cv.unwrap_tensors(
        C.buildPyramid(cv.wrap_tensor(src), cv.wrap_tensors(dst), maxlevel, borderType),
        true)
end


function cv.undistort(t)
    local argRules = {
        {"src", required = true},
        {"cameraMatrix", required = true},
        {"distCoeffs", default = nil},
        {"dst", default = nil},
        {"newCameraMatrix", default = nil}
    }
    local src, cameraMatrix, distCoeffs, dst, newCameraMatrix = cv.argcheck(t, argRules)
    if type(cameraMatrix) == "table" then
        cameraMatrix = torch.FloatTensor(cameraMatrix)
    end
    if type(distCoeffs) == "table" then
        distCoeffs = torch.FloatTensor(distCoeffs)
    end
    if type(newCameraMatrix) == "table" then
        newCameraMatrix = torch.FloatTensor(newCameraMatrix)
    end

    if dst then
        assert(src:type() == dst:type())
        assert(src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.undistort(cv.wrap_tensor(src), cv.wrap_tensor(dst), cv.wrap_tensor(cameraMatrix),
                    cv.wrap_tensor(distCoeffs), cv.wrap_tensor(newCameraMatrix)))
end



function cv.initUndistortRectifyMap(t)
    local argRules = {
        {"cameraMatrix", required = true},
        {"distCoeffs", default = nil},
        {"R", default = nil},
        {"newCameraMatrix", default = nil},
        {"size", required = true, operator = cv.Size},
        {"m1type", required = true},
        {"maps", default = nil}
    }
    local cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type, maps = cv.argcheck(t, argRules)
    if type(cameraMatrix) == "table" then
        cameraMatrix = torch.FloatTensor(cameraMatrix)
    end
    if type(distCoeffs) == "table" then
        distCoeffs = torch.FloatTensor(distCoeffs)
    end
    if type(R) == "table" then
        R = torch.FloatTensor(R)
    end
    if type(newCameraMatrix) == "table" then
        newCameraMatrix = torch.FloatTensor(newCameraMatrix)
    end

    return cv.unwrap_tensors(
        C.initUndistortRectifyMap(
            cv.wrap_tensor(cameraMatrix), cv.wrap_tensor(distCoeffs),
            cv.wrap_tensor(R), cv.wrap_tensor(newCameraMatrix),
            size, m1type, cv.wrap_tensors(maps)))
end

-- value = cv.initWideAngleProjMap{maps={map1, map2}, ...}
-- OR
-- value, map1, map2 = cv.initWideAngleProjMap{...}
--
-- value      -> number
-- map1, map2 -> Tensor
function cv.initWideAngleProjMap(t)
    local argRules = {
        {"cameraMatrix", required = true},
        {"distCoeffs", default = nil},
        {"imageSize", required = true, operator = cv.Size},
        {"destImageWidth", required = true},
        {"m1type", required = true},
        {"maps", default = nil},
        {"projType", required = true},
        {"alpha", required = true}
    }
    local cameraMatrix, distCoeffs, imageSize, destImageWidth, m1type, maps, projType, alpha = cv.argcheck(t, argRules)
    if type(cameraMatrix) == "table" then
        cameraMatrix = torch.FloatTensor(cameraMatrix)
    end
    if type(distCoeffs) == "table" then
        distCoeffs = torch.FloatTensor(distCoeffs)
    end

    local result = C.initWideAngleProjMap(
        cv.wrap_tensor(cameraMatrix), cv.wrap_tensor(distCoeffs),
        imageSize, destImageWidth, m1type, cv.wrap_tensors(maps),
        projType, alpha)
    return result.val, cv.unwrap_tensors(result.tensors)
end


function cv.calcHist(t)
    local argRules = {
        {"images", required = true},
        {"channels", required = true},
        {"mask", default = nil},
        {"hist", default = nil},
        {"dims", required = true},
        {"histSize", required = true},
        {"ranges", required = true},
        {"uniform", default = true},
        {"accumulate", default = false}
    }
    local images, channels, mask, hist, dims, histSize, ranges, uniform, accumulate = cv.argcheck(t, argRules)

    assert(type(images) == 'table')
    assert(torch.isTensor(channels) and channels:type() == 'torch.IntTensor')
    assert(torch.isTensor(histSize) and histSize:type() == 'torch.IntTensor')
    assert(torch.isTensor(ranges) and ranges:type() == 'torch.FloatTensor')
    assert(hist or accumulate == false)

    return cv.unwrap_tensors(
        C.calcHist(
            cv.wrap_tensors(images), cv.wrap_tensor(channels), cv.wrap_tensor(mask),
            cv.wrap_tensor(hist), dims, histSize, cv.wrap_tensor(ranges), uniform, accumulate))
end


function cv.calcBackProject(t)
    local argRules = {
        {"images", required = true},
        {"nimages", required = true},
        {"channels", required = true},
        {"hist", required = true},
        {"backProject", default = nil},
        {"ranges", required = true},
        {"scale", default = 1},
        {"uniform", default = nil}
    }
    local images, nimages, channels, hist, backProject, ranges, scale, uniform = cv.argcheck(t, argRules)

    assert(type(images) == "table")
    assert(torch.isTensor(channels) and channels:type() == 'torch.IntTensor')
    assert(torch.isTensor(ranges) and ranges:type() == 'torch.FloatTensor')

    return cv.unwrap_tensors(
        C.calcBackProject(
            cv.wrap_tensors(images), nimages, cv.wrap_tensor(channels), cv.wrap_tensor(hist),
            cv.wrap_tensor(backProject), cv.wrap_tensor(ranges), scale, uniform))
end


function cv.compareHist(t)
    local argRules = {
        {"H1", required = true},
        {"H2", required = true},
        {"method", required = true}
    }
    local H1, H2, method = cv.argcheck(t, argRules)

    return C.compareHist(cv.wrap_tensor(H1), cv.wrap_tensor(H2), method)
end


function cv.equalizeHist(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil}
    }
    local src, dst = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.equalizeHist(
            cv.wrap_tensor(src), cv.wrap_tensor(dst)))
end


function cv.EMD(t)
    local argRules = {
        {"signature1", required = true},
        {"signature2", required = true},
        {"distType", required = true},
        {"cost", default = nil},
        {"lowerBound", default = nil},
        {"flow", default = nil}
    }
    local signature1, signature2, distType, cost, lowerBound, flow = cv.argcheck(t, argRules)

    if torch.isTensor(lowerBound) then
        assert(lowerBound:type() == 'torch.FloatTensor')
    end

    local result = C.EMD(
        cv.wrap_tensor(signature1), cv.wrap_tensor(signature2), distType,
        cv.wrap_tensor(cost), cv.wrap_tensor(lowerBound), cv.wrap_tensor(flow))
    return result.val, cv.unwrap_tensors(result.tensor)
end


function cv.watershed(t)
    local argRules = {
        {"image", required = true},
        {"markers", required = true}
    }
    local image, markers = cv.argcheck(t, argRules)

    C.watershed(cv.wrap_tensor(image), cv.wrap_tensor(markers))
end


function cv.pyrMeanShiftFiltering(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"sp", required = true},
        {"sr", required = true},
        {"maxLevel", default = 1},
        {"termcrit", default = nil, operator = cv.TermCriteria}
    }
    local src, dst, sp, sr, maxLevel, termcrit = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.pyrMeanShiftFiltering(
        cv.wrap_tensor(src), cv.wrap_tensor(dst), sp, sr, maxlevel, termcrit))
end


function cv.grabCut(t)
    local argRules = {
        {"img", required = true},
        {"mask", required = true},
        {"rect", default = nil, operator = cv.Rect},
        {"bgdModel", required = true},
        {"fgdModel", required = true},
        {"iterCount", required = true},
        {"mode", default = cv.GC_EVAL}
    }
    local img, mask, rect, bgdModel, fgdModel, iterCount, mode = cv.argcheck(t, argRules)

    C.grabCut(
        cv.wrap_tensor(img), cv.wrap_tensor(mask), rect,
        cv.wrap_tensor(bgdModel), cv.wrap_tensor(fgdModel),
        iterCount, mode)
end


function cv.distanceTransform(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"distanceType", required = true},
        {"maskSize", required = true},
        {"dstType", default = cv.CV_32F}
    }
    local src, dst, distanceType, maskSize, dstType = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.distanceTransform(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), distanceType, maskSize, dstType))
end


function cv.distanceTransformWithLabels(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"labels", default = nil},
        {"distanceType", required = true},
        {"maskSize", required = true},
        {"labelType", default = cv.DIST_LABEL_CCOMP}
    }
    local src, dst, labels, distanceType, maskSize, labelType = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.distanceTransformWithLabels(
            cv.wrap_tensor(src), cv.wrap_tensor(dst),
            cv.wrap_tensor(labels), distanceType, maskSize, labelType))
end

-- area, boundingRect = cv.floodFill{...}
-- area         -> number
-- boundingRect -> RectWrapper
function cv.floodFill(t)
    local argRules = {
        {"image", required = true},
        {"mask", default = nil},
        {"seedPoint", required = true, operator = cv.Point},
        {"newVal", required = true, operator = cv.Scalar},
        {"loDiff", default = {0, 0, 0, 0}, operator = cv.Scalar},
        {"upDiff", default = {0, 0, 0, 0}, operator = cv.Scalar},
        {"flags", default = 4}
    }
    local image, mask, seedPoint, newVal, loDiff, upDiff, flags = cv.argcheck(t, argRules)

    local result = C.floodFill(
        cv.wrap_tensor(image), cv.wrap_tensor(mask), seedPoint[1],
        seedPoint[2], newVal, loDiff, upDiff, flags)
    return result.val, result.rect
end


function cv.cvtColor(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"code", required = true},
        {"dstCn", default = 0}
    }
    local src, dst, code, dstCn = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.cvtColor(
        cv.wrap_tensor(src), cv.wrap_tensor(dst), code, dstCn))
end


function cv.demosaicing(t)
    local argRules = {
        {"_src", required = true},
        {"_dst", required = true},
        {"code", required = true},
        {"dcn", default = 0}
    }
    local _src, _dst, code, dcn = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.demosaicing(
        cv.wrap_tensor(_src), cv.wrap_tensor(_dst), code, dcn))
end


function cv.moments(t)
    local argRules = {
        {"array", required = true},
        {"binaryImage", default = false}
    }
    local array, binaryImage = cv.argcheck(t, argRules)

    return C.moments(cv.wrap_tensor(array), binaryImage)
end


function cv.HuMoments(t)
    local argRules = {
        {"moments", required = true}
    }
    local moments = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(C.HuMoments(moments))
end


function cv.matchTemplate(t)
    local argRules = {
        {"image", required = true},
        {"templ", default = nil},
        {"result", required = true},
        {"method", required = true},
        {"mask", default = nil}
    }
    local image, templ, result, method, mask = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.matchTemplate(
        cv.wrap_tensor(image), cv.wrap_tensor(templ), cv.wrap_tensor(result), method, cv.wrap_tensor(mask)))
end


function cv.connectedComponents(t)
    local argRules = {
        {"image", required = true},
        {"labels", required = true},
        {"connectivity", default = 8},
        {"ltype", default = cv.CV_32S}
    }
    local image, labels, connectivity, ltype = cv.argcheck(t, argRules)

    local result = C.connectedComponents(cv.wrap_tensor(image), cv.wrap_tensor(labels), connectivity, ltype)
    return result.val, cv.unwrap_tensors(result.tensor)
end


function cv.connectedComponentsWithStats(t)
    local argRules = {
        {"image", required = true},
        {"labels", required = true},
        {"stats", required = true},
        {"centroids", required = true},
        {"connectivity", default = 8},
        {"ltype", default = cv.CV_32S}
    }
    local image, labels, stats, centroids, connectivity, ltype = cv.argcheck(t, argRules)

    local result = C.connectedComponentsWithStats(
        cv.wrap_tensor(image),
        cv.wrap_tensors(labels, stats, centroids),
        connectivity,
        ltype)
    return result.val, cv.unwrap_tensors(result.tensors)
end

-- image: Source Tensor
-- hierarchy: optional, an array to output hierarchy into
-- withHierarchy: boolean, to output hierarchy or not. Default: false
-- other params: see OpenCV docs for findContours
function cv.findContours(t)
    local argRules = {
        {"image", required = true},
        {"hierarchy", default = nil},
        {"withHierarchy", default = false},
        {"mode", required = true},
        {"method", required = true},
        {"offset", default = {0, 0}, operator = cv.Point}
    }
    local image, hierarchy, withHierarchy, mode, method, offset = cv.argcheck(t, argRules)

    contours = cv.unwrap_tensors(
        C.findContours(
            cv.wrap_tensor(image), withHierarchy, cv.wrap_tensor(hierarchy), mode, method, offset), true)

    if withHierarchy and not hierarchy then
        hierarchy = contours[#contours]
        contours[#contours] = nil
        return contours, hierarchy
    else
        return contours
    end
end


function cv.approxPolyDP(t)
    local argRules = {
        {"curve", required = true},
        {"approxCurve", default = nil},
        {"epsilon", required = true},
        {"closed", required = true}
    }
    local curve, approxCurve, epsilon, closed = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.approxPolyDP(cv.wrap_tensor(curve), cv.wrap_tensor(approxCurve), epsilon, closed))
end


function cv.arcLength(t)
    local argRules = {
        {"curve", required = true},
        {"closed", required = true}
    }
    local curve, closed = cv.argcheck(t, argRules)

    return C.arcLength(cv.wrap_tensor(curve), closed)
end


function cv.boundingRect(t)
    local argRules = {
        {"points", required = true}
    }
    local points = cv.argcheck(t, argRules)
    if type(points) == "table" then
        points = torch.FloatTensor(points)
    end

    return C.boundingRect(cv.wrap_tensor(points))
end


function cv.contourArea(t)
    local argRules = {
        {"contour", required = true},
        {"oriented", default = false}
    }
    local contour, oriented = cv.argcheck(t, argRules)
    if type(contour) == "table" then
        contour = torch.FloatTensor(contour)
    end


    return C.contourArea(cv.wrap_tensor(contour), oriented)
end


function cv.minAreaRect(t)
    local argRules = {
        {"points", required = true}
    }
    local points = cv.argcheck(t, argRules)
    if type(points) == "table" then
        points = torch.FloatTensor(points)
    end

    return C.minAreaRect(cv.wrap_tensor(points))
end


-- box: a RotatedRectWrapper
-- points: optional; a 4x2 Tensor to hold the return value
function cv.boxPoints(t)
    local argRules = {
        {"box", required = true},
        {"points", default = nil}
    }
    local box, points = cv.argcheck(t, argRules)
    -- check that points is a Tensor
    assert(not points or points.torch)

    return cv.unwrap_tensors(C.boxPoints(box, cv.wrap_tensor(points)))
end

-- points: a Tensor or a table of points
-- return value: center, radius
function cv.minEnclosingCircle(t)
    local argRules = {
        {"points", required = true}
    }
    local points = cv.argcheck(t, argRules)
    if type(points) == "table" then
        points = torch.FloatTensor(points)
    end

    local result = C.minEnclosingCircle(cv.wrap_tensor(points))
    return cv.Point2f(result.v0, result.v1), result.v2
end

-- points: a Tensor or a table of points
-- triangle: optional; a 3x2 Tensor to hold the return value
-- return value: triangle_points, area
function cv.minEnclosingTriangle(t)
    local argRules = {
        {"points", required = true},
        {"triangle", default = nil}
    }
    local points, triangle = cv.argcheck(t, argRules)
    if type(points) == "table" then
        points = torch.FloatTensor(points)
    end
    -- check that triangle is a Tensor
    assert(not triangle or triangle.torch)

    local result = C.minEnclosingTriangle(cv.wrap_tensor(points), cv.wrap_tensor(triangle))
    return cv.unwrap_tensors(result.tensor), result.val
end


function cv.matchShapes(t)
    local argRules = {
        {"contour1", required = true},
        {"contour2", required = true},
        {"method", required = true},
        {"parameter", default = 0}
    }
    local contour1, contour2, method, parameter = cv.argcheck(t, argRules)

    return C.matchShapes(cv.wrap_tensor(contour1), cv.wrap_tensor(contour2), method, parameter)
end


function cv.convexHull(t)
    local argRules = {
        {"points", required = true},
        {"hull", default = nil},
        {"clockwise", default = false},
        {"returnPoints", default = nil}
    }
    local points, hull, clockwise, returnPoints = cv.argcheck(t, argRules)
    if returnPoints == nil then
         returnPoints = true
    end

    retval = cv.unwrap_tensors(
			C.convexHull(
				cv.wrap_tensor(points), cv.wrap_tensor(hull),
				clockwise, returnPoints))
    if not returnPoints then
        -- correct the 0-based indexing
        for i = 1,#retval do
            retval[i] = retval[i] + 1
        end
    end
    return retval
end


function cv.convexityDefects(t)
    local argRules = {
        {"contour", required = true},
        {"convexhull", required = true},
        {"convexityDefects", default = nil}
    }
    local contour, convexhull, convexityDefects = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
		C.convexityDefects(
			cv.wrap_tensor(contour), cv.wrap_tensor(convexhull),
			cv.wrap_tensor(convexityDefects)))
end


function cv.isContourConvex(t)
    local argRules = {
        {"contour", required = true}
    }
    local contour = cv.argcheck(t, argRules)

    return C.isContourConvex(cv.wrap_tensor(contour))
end


-- contour1, contour2: Tensors containing points
-- handleNested: boolean
-- return value: intersection
function cv.intersectConvexConvex(t)
    local argRules = {
        {"_p1", required = true},
        {"_p2", required = true},
        {"_p12", default = nil},
        {"handleNested", default = nil}
    }
    local _p1, _p2, _p12, handleNested = cv.argcheck(t, argRules)
    if handleNested == nil then
        handleNested = true
    end

    return cv.unwrap_tensors(
        C.intersectConvexConvex(
		cv.wrap_tensor(_p1), cv.wrap_tensor(_p2), cv.wrap_tensor(_p12), handleNested))
end


function cv.fitEllipse(t)
    local argRules = {
        {"points", required = true}
    }
    local points = cv.argcheck(t, argRules)

    return C.fitEllipse(cv.wrap_tensor(points))
end


function cv.fitLine(t)
    local argRules = {
        {"points", required = true},
        {"line", default = nil},
        {"distType", required = true},
        {"param", required = true},
        {"reps", required = true},
        {"aeps", required = true}
    }
    local points, line, distType, param, reps, aeps = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.fitLine(cv.wrap_tensor(points), cv.wrap_tensor(line), distType, param, reps, aeps))
end


function cv.pointPolygonTest(t)
    local argRules = {
        {"contour", required = true},
        {"pt", required = true, operator = cv.Point2f},
        {"measureDist", required = true}
    }
    local contour, pt, measureDist = cv.argcheck(t, argRules)

    return C.pointPolygonTest(cv.wrap_tensor(contour), pt, measureDist)
end


function cv.rotatedRectangleIntersection(t)
    local argRules = {
        {"rect1", required = true, operator = cv.RotatedRect},
        {"rect2", required = true, operator = cv.RotatedRect}
    }
    local rect1, rect2 = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.rotatedRectangleIntersection(rect1, rect2))
end


function cv.blendLinear(t)
    local argRules = {
        {"src1", required = true},
        {"src2", required = true},
        {"weights1", required = true},
        {"weights2", required = true},
        {"dst", default = nil}
    }
    local src1, src2, weights1, weights2, dst = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.blendLinear(cv.wrap_tensor(src1), cv.wrap_tensor(src2), cv.wrap_tensor(weights1), cv.wrap_tensor(weights2), cv.wrap_tensor(dst)))
end


function cv.applyColorMap(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"colormap", required = true}
    }
    local src, dst, colormap = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.applyColorMap(cv.wrap_tensor(src), cv.wrap_tensor(dst), colormap))
end


function cv.line(t)
    local argRules = {
        {"img", required = true},
        {"pt1", required = true, operator = cv.Point},
        {"pt2", required = true, operator = cv.Point},
        {"color", required = true, operator = cv.Scalar},
        {"thickness", default = 1},
        {"lineType", default = cv.LINE_8},
        {"shift", default = 0}
    }
    local img, pt1, pt2, color, thickness, lineType, shift = cv.argcheck(t, argRules)

    C.line(cv.wrap_tensor(img), pt1, pt2, color, thickness, lineType, shift)
end


function cv.arrowedLine(t)
    local argRules = {
        {"img", required = true},
        {"pt1", required = true, operator = cv.Point},
        {"pt2", required = true, operator = cv.Point},
        {"color", required = true, operator = cv.Scalar},
        {"thickness", default = 1},
        {"line_type", default = 8},
        {"shift", default = 0},
        {"tipLength", default = 0.1}
    }
    local img, pt1, pt2, color, thickness, line_type, shift, tipLength = cv.argcheck(t, argRules)

    C.arrowedLine(cv.wrap_tensor(img), pt1, pt2, color, thickness, line_type, shift, tipLength)
end


function cv.rectangle(t)
    local argRules = {
        {"img", required = true},
        {"pt1", required = true, operator = cv.Point},
        {"pt2", required = true, operator = cv.Point},
        {"color", required = true, operator = cv.Scalar},
        {"thickness", default = 1},
        {"lineType", default = cv.LINE_8},
        {"shift", default = 0}
    }
    local img, pt1, pt2, color, thickness, lineType, shift = cv.argcheck(t, argRules)

    C.rectangle(cv.wrap_tensor(img), pt1, pt2, color, thickness, lineType, shift)
end

function cv.rectangle2(t)
    local argRules = {
        {"img", required = true},
        {"rec", required = true, operator = cv.Rect},
        {"color", required = true, operator = cv.Scalar},
        {"thickness", default = 1},
        {"lineType", default = cv.LINE_8},
        {"shift", default = 0}
    }
    local img, rec, color, thickness, lineType, shift = cv.argcheck(t, argRules)

    C.rectangle2(cv.wrap_tensor(img), rec, color, thickness, lineType, shift)
end




function cv.circle(t)
    local argRules = {
        {"img", required = true},
        {"center", required = true, operator = cv.Point},
        {"radius", required = true},
        {"color", required = true, operator = cv.Scalar},
        {"thickness", default = 1},
        {"lineType", default = cv.LINE_8},
        {"shift", default = 0}
    }
    local img, center, radius, color, thickness, lineType, shift = cv.argcheck(t, argRules)

    C.circle(cv.wrap_tensor(img), center, radius, color, thickness, lineType, shift)
end


function cv.ellipse(t)
    local argRules = {
        {"img", required = true},
        {"center", required = true, operator = cv.Point},
        {"axes", required = true, operator = cv.Size},
        {"angle", required = true},
        {"startAngle", required = true},
        {"endAngle", required = true},
        {"color", required = true, operator = cv.Scalar},
        {"thickness", default = 1},
        {"lineType", default = cv.LINE_8},
        {"shift", default = 0}
    }
    local img, center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift = cv.argcheck(t, argRules)

    C.ellipse(cv.wrap_tensor(img), center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift)
end


function cv.ellipseFromRect(t)
    local argRules = {
        {"img", required = true},
        {"box", required = true, operator = cv.RotatedRect},
        {"color", required = true, operator = cv.Scalar},
        {"thickness", default = 1},
        {"lineType", default = cv.LINE_8}
    }
    local img, box, color, thickness, lineType = cv.argcheck(t, argRules)

    C.ellipseFromRect(cv.wrap_tensor(img), box, color, thickness, lineType)
end


function cv.fillConvexPoly(t)
    local argRules = {
        {"img", required = true},
        {"points", required = true},
        {"color", required = true, operator = cv.Scalar},
        {"lineType", default = cv.LINE_8},
        {"shift", default = 0}
    }
    local img, points, color, lineType, shift = cv.argcheck(t, argRules)
    if type(points) == "table" then
        points = torch.FloatTensor(points)
    end

    C.fillConvexPoly(cv.wrap_tensor(img), cv.wrap_tensor(points), color, lineType, shift)
end


function cv.fillPoly(t)
    local argRules = {
        {"img", required = true},
        {"pts", required = true},
        {"color", required = true, operator = cv.Scalar},
        {"lineType", default = cv.LINE_8},
        {"shift", default = 0},
        {"offset", default = {0,0}, operator = cv.Point}
    }
    local img, pts, color, lineType, shift, offset = cv.argcheck(t, argRules)
    assert(type(pts) == 'table')

    C.fillPoly(cv.wrap_tensor(img), cv.wrap_tensors(pts), color, lineType, shift, offset)
end


function cv.polylines(t)
    local argRules = {
        {"img", required = true},
        {"pts", required = true},
        {"isClosed", required = true},
        {"color", required = true, operator = cv.Scalar},
        {"thickness", default = 1},
        {"lineType", default = cv.LINE_8},
        {"shift", default = 0}
    }
    local img, pts, isClosed, color, thickness, lineType, shift = cv.argcheck(t, argRules)

    assert(type(pts) == 'table')

    C.polylines(cv.wrap_tensor(img), cv.wrap_tensors(pts), isClosed, color, thickness, lineType, shift)
end


function cv.drawContours(t)
    local argRules = {
        {"image", required = true},
        {"contours", required = true},
        {"contourIdx", required = true},
        {"color", required = true, operator = cv.Scalar},
        {"thickness", default = 1},
        {"lineType", default = cv.LINE_8},
        {"hierarchy", default = nil},
        {"maxLevel", default = cv.INT_MAX},
        {"offset", default = cv.Point}
    }
    local image, contours, contourIdx, color, thickness, lineType, hierarchy, maxLevel, offset = cv.argcheck(t, argRules)

    assert(type(contours) == 'table')

    C.drawContours(cv.wrap_tensor(image), cv.wrap_tensors(contours), contourIdx, color, thickness, lineType, cv.wrap_tensor(hierarchy), maxLevel, offset)
end


function cv.clipLine(t)
    local argRules = {
        {"imgSize", default = nil},
        {"imgRect", default = nil},
        {"pt1", required = true, operator = cv.Point},
        {"pt2", required = true, operator = cv.Point}
    }
    local imgSize, imgRect, pt1, pt2 = cv.argcheck(t, argRules)
    assert(imgSize or imgRect)
    if imgSize then
        imgSize = cv.Size(imgSize)
    else
        imgRect = cv.Rect(imgRect)
    end


    local result
    if imgSize then
        result = C.clipLineSize(imgSize, pt1, pt2)
    else
        result = C.clipLineRect(imgRect, pt1, pt2)
    end

    return cv.Point{result.scalar.v0, result.scalar.v1}, cv.Point{result.scalar.v2, result.scalar.v3}, result.val
end


function cv.ellipse2Poly(t)
    local argRules = {
        {"center", required = true, operator = cv.Point},
        {"axes", required = true, operator = cv.Size},
        {"angle", required = true},
        {"arcStart", required = true},
        {"arcEnd", required = true},
        {"delta", required = true}
    }
    local center, axes, angle, arcStart, arcEnd, delta = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta, pts))
end


function cv.putText(t)
    local argRules = {
        {"img", required = true},
        {"text", required = true},
        {"org", required = true, operator = cv.Point},
        {"fontFace", required = true},
        {"fontScale", required = true},
        {"color", required = true, operator = cv.Scalar},
        {"thickness", default = 1},
        {"lineType", default = cv.LINE_8},
        {"bottomLeftOrigin", default = false}
    }
    local img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin = cv.argcheck(t, argRules)

    C.putText(cv.wrap_tensor(img), text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)
end


function cv.getTextSize(t)
    local argRules = {
        {"text", required = true},
        {"fontFace", required = true},
        {"fontScale", required = true},
        {"thickness", required = true}
    }
    local text, fontFace, fontScale, thickness = cv.argcheck(t, argRules)

    local result = C.getTextSize(text, fontFace, fontScale, thickness)
    return result.size, result.val
end

--- ***************** Classes *****************
require 'cv.Classes'

local Classes = ffi.load(cv.libPath('Classes'))

ffi.cdef[[
void Algorithm_dtor(struct PtrWrapper ptr);

void GeneralizedHough_setTemplate(
        struct PtrWrapper ptr, struct TensorWrapper templ, struct PointWrapper templCenter);

void GeneralizedHough_setTemplate_edges(
        struct PtrWrapper ptr, struct TensorWrapper edges, struct TensorWrapper dx,
        struct TensorWrapper dy, struct PointWrapper templCenter);

struct TensorArray GeneralizedHough_detect(
        struct PtrWrapper ptr, struct TensorWrapper image, struct TensorWrapper positions, bool votes);

struct TensorArray GeneralizedHough_detect_edges(
        struct PtrWrapper ptr, struct TensorWrapper edges, struct TensorWrapper dx,
        struct TensorWrapper dy, struct TensorWrapper positions, bool votes);

void GeneralizedHough_setCannyLowThresh(struct PtrWrapper ptr, int cannyLowThresh);

int GeneralizedHough_getCannyLowThresh(struct PtrWrapper ptr);

void GeneralizedHough_setCannyHighThresh(struct PtrWrapper ptr, int cannyHighThresh);

int GeneralizedHough_getCannyHighThresh(struct PtrWrapper ptr);

void GeneralizedHough_setMinDist(struct PtrWrapper ptr, double MinDist);

double GeneralizedHough_getMinDist(struct PtrWrapper ptr);

void GeneralizedHough_setDp(struct PtrWrapper ptr, double Dp);

double GeneralizedHough_getDp(struct PtrWrapper ptr);

void GeneralizedHough_setMaxBufferSize(struct PtrWrapper ptr, int MaxBufferSize);

int GeneralizedHough_getMaxBufferSize(struct PtrWrapper ptr);

struct PtrWrapper GeneralizedHoughBallard_ctor();

void GeneralizedHoughBallard_setLevels(struct PtrWrapper ptr, double Levels);

double GeneralizedHoughBallard_getLevels(struct PtrWrapper ptr);

void GeneralizedHoughBallard_setVotesThreshold(struct PtrWrapper ptr, double votesThreshold);

double GeneralizedHoughBallard_getVotesThreshold(struct PtrWrapper ptr);

struct PtrWrapper GeneralizedHoughGuil_ctor();

void GeneralizedHoughGuil_setLevels(struct PtrWrapper ptr, int levels);

int GeneralizedHoughGuil_getLevels(struct PtrWrapper ptr);

void GeneralizedHoughGuil_setAngleEpsilon(struct PtrWrapper ptr, double AngleEpsilon);

double GeneralizedHoughGuil_getAngleEpsilon(struct PtrWrapper ptr);

void GeneralizedHoughGuil_setMinAngle(struct PtrWrapper ptr, double MinAngle);

double GeneralizedHoughGuil_getMinAngle(struct PtrWrapper ptr);

void GeneralizedHoughGuil_setMaxAngle(struct PtrWrapper ptr, double MaxAngle);

double GeneralizedHoughGuil_getMaxAngle(struct PtrWrapper ptr);

void GeneralizedHoughGuil_setAngleStep(struct PtrWrapper ptr, double AngleStep);

double GeneralizedHoughGuil_getAngleStep(struct PtrWrapper ptr);

void GeneralizedHoughGuil_setAngleThresh(struct PtrWrapper ptr, int AngleThresh);

int GeneralizedHoughGuil_getAngleThresh(struct PtrWrapper ptr);

void GeneralizedHoughGuil_setMinScale(struct PtrWrapper ptr, double MinScale);

double GeneralizedHoughGuil_getMinScale(struct PtrWrapper ptr);

void GeneralizedHoughGuil_setMaxScale(struct PtrWrapper ptr, double MaxScale);

double GeneralizedHoughGuil_getMaxScale(struct PtrWrapper ptr);

void GeneralizedHoughGuil_setScaleStep(struct PtrWrapper ptr, double ScaleStep);

double GeneralizedHoughGuil_getScaleStep(struct PtrWrapper ptr);

void GeneralizedHoughGuil_setScaleThresh(struct PtrWrapper ptr, int ScaleThresh);

int GeneralizedHoughGuil_getScaleThresh(struct PtrWrapper ptr);

void GeneralizedHoughGuil_setPosThresh(struct PtrWrapper ptr, int PosThresh);

int GeneralizedHoughGuil_getPosThresh(struct PtrWrapper ptr);

struct PtrWrapper CLAHE_ctor();

void CLAHE_setClipLimit(struct PtrWrapper ptr, double ClipLimit);

double CLAHE_getClipLimit(struct PtrWrapper ptr);

void CLAHE_setTilesGridSize(struct PtrWrapper ptr, struct SizeWrapper TilesGridSize);

struct SizeWrapper CLAHE_getTilesGridSize(struct PtrWrapper ptr);

void CLAHE_collectGarbage(struct PtrWrapper ptr);

struct PtrWrapper LineSegmentDetector_ctor(
        int refine, double scale, double sigma_scale, double quant,
        double ang_th, double log_eps, double density_th, int n_bins);

struct TensorArray LineSegmentDetector_detect(
        struct PtrWrapper ptr, struct TensorWrapper image,
        struct TensorWrapper lines, bool width, bool prec, bool nfa);

struct TensorWrapper LineSegmentDetector_drawSegments(
        struct PtrWrapper ptr, struct TensorWrapper image, struct TensorWrapper lines);

int LineSegmentDetector_compareSegments(struct PtrWrapper ptr, struct SizeWrapper size, struct TensorWrapper lines1,
                    struct TensorWrapper lines2, struct TensorWrapper image);

struct PtrWrapper Subdiv2D_ctor_default();

struct PtrWrapper Subdiv2D_ctor(struct RectWrapper rect);

void Subdiv2D_dtor(struct PtrWrapper ptr);

void Subdiv2D_initDelaunay(struct PtrWrapper ptr, struct RectWrapper rect);

int Subdiv2D_insert(struct PtrWrapper ptr, struct Point2fWrapper pt);

void Subdiv2D_insert_vector(struct PtrWrapper ptr, struct TensorWrapper ptvec);

struct Vec3iWrapper Subdiv2D_locate(struct PtrWrapper ptr, struct Point2fWrapper pt);

struct Point2fPlusInt Subdiv2D_findNearest(struct PtrWrapper ptr, struct Point2fWrapper pt);

struct TensorWrapper Subdiv2D_getEdgeList(struct PtrWrapper ptr);

struct TensorWrapper Subdiv2D_getTriangleList(struct PtrWrapper ptr);

struct TensorArray Subdiv2D_getVoronoiFacetList(struct PtrWrapper ptr, struct TensorWrapper idx);

struct Point2fPlusInt Subdiv2D_getVertex(struct PtrWrapper ptr, int vertex);

int Subdiv2D_getEdge(struct PtrWrapper ptr, int edge, int nextEdgeType);

int Subdiv2D_nextEdge(struct PtrWrapper ptr, int edge);

int Subdiv2D_rotateEdge(struct PtrWrapper ptr, int edge, int rotate);

int Subdiv2D_symEdge(struct PtrWrapper ptr, int edge);

struct Point2fPlusInt Subdiv2D_edgeOrg(struct PtrWrapper ptr, int edge);

struct Point2fPlusInt Subdiv2D_edgeDst(struct PtrWrapper ptr, int edge);

struct PtrWrapper LineIterator_ctor(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2,
        int connectivity, bool leftToRight);

void LineIterator_dtor(struct PtrWrapper ptr);

int LineIterator_count(struct PtrWrapper ptr);

struct PointWrapper LineIterator_pos(struct PtrWrapper ptr);

void LineIterator_incr(struct PtrWrapper ptr);
]]

-- GeneralizedHough

do
    local GeneralizedHough = torch.class('cv.GeneralizedHough', 'cv.Algorithm', cv)

    function GeneralizedHough:setTemplate(t)
        if t.dy or #t > 2 then
            local argRules = {
                {"edges", required = true},
                {"dx", required = true},
                {"dy", required = true},
                {"templCenter", default = {-1, -1}, operator = cv.Point}
            }
            local edges, dx, dy, templCenter = cv.argcheck(t, argRules)

            C.GeneralizedHough_setTemplate_edges(
                    self.ptr, cv.wrap_tensor(edges), cv.wrap_tensor(dx),
                    cv.wrap_tensor(dy), templCenter)
        else
            local argRules = {
                {"templ", required = true},
                {"templCenter", default = {-1, -1}, operator = cv.Point}
            }
            local templ, templCenter = cv.argcheck(t, argRules)

            C.GeneralizedHough_setTemplate(self.ptr, cv.wrap_tensor(templ), templCenter)
        end
    end

    -- votes: boolean. To output votes or not
    function GeneralizedHough:detect(t)
        local argRules = {
            {"image", required = true},
            {"positions", default = nil},
            {"votes", default = false},
            {"image", required = true},
            {"positions", default = nil},
            {"votes", default = false}
        }
        local image, positions, votes, image, positions, votes = cv.argcheck(t, argRules)
        if t.image then

            return cv.unwrap_tensors(
                C.GeneralizedHough_detect(
                       self.ptr, cv.wrap_tensor(image), cv.wrap_tensor(positions), votes))
        else

            return cv.unwrap_tensors(
                C.GeneralizedHough_detect(
                    self.ptr, cv.wrap_tensor(image), cv.wrap_tensor(positions), votes))
        end
    end

    function GeneralizedHough:setCannyLowThresh(cannyLowThresh)
        C.GeneralizedHough_setCannyLowThresh(self.ptr, cannyLowThresh)
    end

    function GeneralizedHough:getCannyLowThresh()
        return C.GeneralizedHough_getCannyLowThresh(self.ptr)
    end

    function GeneralizedHough:setCannyHighThresh(cannyHighThresh)
        C.GeneralizedHough_setCannyHighThresh(self.ptr, cannyHighThresh)
    end

    function GeneralizedHough:getCannyHighThresh()
        return C.GeneralizedHough_getCannyHighThresh(self.ptr)
    end

    function GeneralizedHough:setMinDist(minDist)
        C.GeneralizedHough_setMinDist(self.ptr, MinDist)
    end

    function GeneralizedHough:getMinDist()
        return C.GeneralizedHough_getMinDist(self.ptr)
    end

    function GeneralizedHough:setDp(dp)
        C.GeneralizedHough_setDp(self.ptr, Dp)
    end

    function GeneralizedHough:getDp()
        return C.GeneralizedHough_getDp(self.ptr)
    end

    function GeneralizedHough:setMaxBufferSize(maxBufferSize)
        C.GeneralizedHough_setMaxBufferSize(self.ptr, MaxBufferSize)
    end

    function GeneralizedHough:getMaxBufferSize()
        return C.GeneralizedHough_getMaxBufferSize(self.ptr)
    end
end

-- GeneralizedHoughBallard

do
    local GeneralizedHoughBallard = torch.class('cv.GeneralizedHoughBallard', 'cv.GeneralizedHough', cv)

    function GeneralizedHoughBallard:__init()
        self.ptr = ffi.gc(C.GeneralizedHoughBallard_ctor(), Classes.Algorithm_dtor)
    end

    function GeneralizedHoughBallard:setLevels(levels)
        C.GeneralizedHoughBallard_setLevels(self.ptr, levels)
    end

    function GeneralizedHoughBallard:getLevels()
        return C.GeneralizedHoughBallard_getLevels(self.ptr)
    end

    function GeneralizedHoughBallard:setVotesThreshold(votesThreshold)
        C.GeneralizedHoughBallard_setVotesThreshold(self.ptr, votesThreshold)
    end

    function GeneralizedHoughBallard:getVotesThreshold()
        return C.GeneralizedHoughBallard_getVotesThreshold(self.ptr)
    end
end

-- GeneralizedHoughGuil

do
    local GeneralizedHoughGuil = torch.class('cv.GeneralizedHoughGuil', 'cv.GeneralizedHough', cv)

    function GeneralizedHoughGuil:__init()
        self.ptr = ffi.gc(C.GeneralizedHoughGuil_ctor(), Classes.Algorithm_dtor)
    end

    function GeneralizedHoughGuil:setXi(xi)
        C.GeneralizedHoughGuil_setXi(self.ptr, xi)
    end

    function GeneralizedHoughGuil:getXi()
        return C.GeneralizedHoughGuil_getXi(self.ptr)
    end

    function GeneralizedHoughGuil:setLevels(levels)
        C.GeneralizedHoughGuil_setLevels(self.ptr, levels)
    end

    function GeneralizedHoughGuil:getLevels()
        return C.GeneralizedHoughGuil_getLevels(self.ptr)
    end

    function GeneralizedHoughGuil:setAngleEpsilon(angleEpsilon)
        C.GeneralizedHoughGuil_setAngleEpsilon(self.ptr, angleEpsilon)
    end

    function GeneralizedHoughGuil:getAngleEpsilon()
        return C.GeneralizedHoughGuil_getAngleEpsilon(self.ptr)
    end

    function GeneralizedHoughGuil:setMinAngle(minAngle)
        C.GeneralizedHoughGuil_setMinAngle(self.ptr, minAngle)
    end

    function GeneralizedHoughGuil:getMinAngle()
        return C.GeneralizedHoughGuil_getMinAngle(self.ptr)
    end

    function GeneralizedHoughGuil:setMaxAngle(maxAngle)
        C.GeneralizedHoughGuil_setMaxAngle(self.ptr, maxAngle)
    end

    function GeneralizedHoughGuil:getMaxAngle()
        return C.GeneralizedHoughGuil_getMaxAngle(self.ptr)
    end

    function GeneralizedHoughGuil:setAngleStep(angleStep)
        C.GeneralizedHoughGuil_setAngleStep(self.ptr, angleStep)
    end

    function GeneralizedHoughGuil:getAngleStep()
        return C.GeneralizedHoughGuil_getAngleStep(self.ptr)
    end

    function GeneralizedHoughGuil:setAngleThresh(angleThresh)
        C.GeneralizedHoughGuil_setAngleThresh(self.ptr, angleThresh)
    end

    function GeneralizedHoughGuil:getAngleThresh()
        return C.GeneralizedHoughGuil_getAngleThresh(self.ptr)
    end

    function GeneralizedHoughGuil:setMinScale(minScale)
        C.GeneralizedHoughGuil_setMinScale(self.ptr, minScale)
    end

    function GeneralizedHoughGuil:getMinScale()
        return C.GeneralizedHoughGuil_getMinScale(self.ptr)
    end

    function GeneralizedHoughGuil:setMaxScale(maxScale)
        C.GeneralizedHoughGuil_setMaxScale(self.ptr, maxScale)
    end

    function GeneralizedHoughGuil:getMaxScale()
        return C.GeneralizedHoughGuil_getMaxScale(self.ptr)
    end

    function GeneralizedHoughGuil:setScaleStep(scaleStep)
        C.GeneralizedHoughGuil_setScaleStep(self.ptr, scaleStep)
    end

    function GeneralizedHoughGuil:getScaleStep()
        return C.GeneralizedHoughGuil_getScaleStep(self.ptr)
    end

    function GeneralizedHoughGuil:setScaleThresh(scaleThresh)
        C.GeneralizedHoughGuil_setScaleThresh(self.ptr, scaleThresh)
    end

    function GeneralizedHoughGuil:getScaleThresh()
        return C.GeneralizedHoughGuil_getScaleThresh(self.ptr)
    end

    function GeneralizedHoughGuil:setPosThresh(posThresh)
        C.GeneralizedHoughGuil_setPosThresh(self.ptr, posThresh)
    end

    function GeneralizedHoughGuil:getPosThresh()
        return C.GeneralizedHoughGuil_getPosThresh(self.ptr)
    end
end

-- CLAHE

do
    local CLAHE = torch.class('cv.CLAHE', 'cv.Algorithm', cv)

    function CLAHE:__init()
        self.ptr = ffi.gc(C.CLAHE_ctor(), Classes.Algorithm_dtor)
    end

    function CLAHE:setClipLimit(clipLimit)
        C.CLAHE_setClipLimit(self.ptr, clipLimit)
    end

    function CLAHE:getClipLimit()
        return C.CLAHE_getClipLimit(self.ptr)
    end

    function CLAHE:setTileGridSize(tileGridSize)
        C.CLAHE_setTileGridSize(self.ptr, TileGridSize)
    end

    function CLAHE:getTileGridSize()
        return C.CLAHE_getTileGridSize(self.ptr)
    end

    function CLAHE:collectGarbage()
        C.CLAHE_collectGarbage(self.ptr)
    end
end

-- LineSegmentDetector

do
    local LineSegmentDetector = torch.class('cv.LineSegmentDetector', 'cv.Algorithm', cv)

    function LineSegmentDetector:__init(t)
        local argRules = {
            {"refine", default = cv.LSD_REFINE_STD},
            {"scale", default = 0.8},
            {"sigma_scale", default = 0.6},
            {"quant", default = 2.0},
            {"ang_th", default = 22.5},
            {"log_eps", default = 0},
            {"density_th", default = 0.7},
            {"n_bins", default = 1024}
        }
        local refine, scale, sigma_scale, quant, ang_th, log_eps, density_th, n_bins = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(
            C.LineSegmentDetector_ctor(
                refine, scale, sigma_scale, quant, ang_th, log_eps, density_th, n_bins),
            Classes.Algorithm_dtor
        )
    end

    function LineSegmentDetector:detect(t)
        local argRules = {
            {"image", required = true},
            {"lines", default = nil},
            {"width", default = false},
            {"prec", default = false},
            {"nfa", default = false}
        }
        local image, lines, width, prec, nfa = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(
            C.LineSegmentDetector_detect(
                self.ptr, cv.wrap_tensor(image), cv.wrap_tensor(lines), width, prec, nfa))
    end

    function LineSegmentDetector:drawSegments(t)
        local argRules = {
            {"image", required = true},
            {"lines", required = true}
        }
        local image, lines = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.LineSegmentDetector_drawSegments(
            self.ptr, cv.wrap_tensor(image), cv.wrap_tensor(lines)))
    end

    function LineSegmentDetector:compareSegments(t)
        local argRules = {
            {"lines1", required = true},
            {"lines2", required = true},
            {"image", default = nil}
        }
        local lines1, lines2, image = cv.argcheck(t, argRules)

        return C.LineSegmentDetector_compareSegments(
            self.ptr, cv.wrap_tensor(lines1), cv.wrap_tensor(lines2), cv.wrap_tensor(image))
    end
end

-- Subdiv2D

do
    local Subdiv2D = torch.class('cv.Subdiv2D', cv)

    function Subdiv2D:__init(t)
        local argRules = {
            {"rect", default = -1, operator = cv.Rect}
        }
        local rect = cv.argcheck(t, argRules)

        if rect.x == -1 then
            self.ptr = ffi.gc(C.Subdiv2D_ctor(rect), C.Subdiv2D_dtor)
        else
            self.ptr = ffi.gc(C.Subdiv2D_ctor_default(), C.Subdiv2D_dtor)
        end
    end

    function Subdiv2D:initDelaunay(t)
        local argRules = {
            {"rect", required = true, operator = cv.Rect}
        }
        local rect = cv.argcheck(t, argRules)

        C.Subdiv2D_initDelaunay(self.ptr, rect)
    end

    function Subdiv2D:insert(t)
        local argRules = {
            {"pt", default = nil},
            {"ptvec", default = nil}
        }
        local pt, ptvec = cv.argcheck(t, argRules)

        if pt then
            return C.Subdiv2D_insert(cv.Point2f(pt))
        else
            if type(ptvec) == "table" then
                ptvec = torch.FloatTensor(ptvec)
            end
            C.Subdiv2D_insert_vector(self.ptr, cv.wrap_tensor(ptvec))
        end
    end

    function Subdiv2D:locate(t)
        local argRules = {
            {"pt", required = true, operator = cv.Point2f}
        }
        local pt = cv.argcheck(t, argRules)

        local result = C.Subdiv2D_locate(self.ptr, pt)
        return result.v0, result.v1, result.v2
    end

    function Subdiv2D:findNearest(t)
        local argRules = {
            {"pt", required = true, operator = cv.Point2f}
        }
        local pt = cv.argcheck(t, argRules)

        local result = C.Subdiv2D_findNearest(self.ptr, pt)
        return result.point, result.val
    end

    function Subdiv2D:getEdgeList(t)
        return cv.unwrap_tensors(C.Subdiv2D_getEdgeList(self.ptr))
    end

    function Subdiv2D:getTriangleList(t)
        return cv.unwrap_tensors(C.Subdiv2D_getTriangleList(self.ptr), true)
    end

    function Subdiv2D:getVoronoiFacetList(t)
        local facetList = cv.unwrap_tensors(C.Subdiv2D_getVoronoiFacetList(self.ptr, cv.unwrap_tensors(idx)), true)
        local facetCenters = facetList[#facetList]
        facetList[#facetList] = nil
        return facetCenters, facetList
    end

    function Subdiv2D:getVertex(t)
        local result = C.Subdiv2D_getVertex(self.ptr, vertex)
        return result.point, result.val
    end

    function Subdiv2D:getEdge(t)
        return C.Subdiv2D_getEdge(self.ptr, edge, nextEdgeType)
    end

    function Subdiv2D:nextEdge(t)
        return C.Subdiv2D_nextEdge(self.ptr, edge)
    end

    function Subdiv2D:rotateEdge(t)
        return C.Subdiv2D_rotateEdge(self.ptr, edge, rotate)
    end

    function Subdiv2D:symEdge(t)
        return C.Subdiv2D_symEdge(self.ptr, edge)
    end

    function Subdiv2D:edgeOrg(t)
        local result = C.Subdiv2D_edgeOrg(self.ptr, edge)
        return result.point, result.val
    end

    function Subdiv2D:edgeDst(t)
        local result = C.Subdiv2D_edgeDst(self.ptr, edge)
        return result.point, result.val
    end
end

-- LineIterator

function cv.LineIterator(t)
    local argRules = {
        {"img", required = true},
        {"pt1", required = true, operator = cv.Point},
        {"pt2", required = true, operator = cv.Point},
        {"connectivity", default = 8},
        {"leftToRight", default = false}
    }
    local img, pt1, pt2, connectivity, leftToRight = cv.argcheck(t, argRules)
    pt1.x = pt1.x - 1
    pt1.y = pt1.y - 1
    pt2.x = pt2.x - 1
    pt2.y = pt2.y - 1

    local ptr = ffi.gc(
        C.LineIterator_ctor(cv.wrap_tensor(img), pt1, pt2, connectivity, leftToRight),
        C.LineIterator_dtor)
    local count = C.LineIterator_count(ptr)

    local
    function lineIter(pos)
        if count > 0 then
            count = count - 1
            result = C.LineIterator_pos(ptr)
            C.LineIterator_incr(ptr)
            return result
        end
    end

    return lineIter, nil, nil
end

function cv.addWeighted(t)
    local argRules = {
        {"src1", required = true},
        {"alpha", required = true},
        {"src2", required = true},
        {"beta", required = true},
        {"gamma", required = true},
        {"dst", default = nil},
        {"dtype", default = -1}
    }
    local src1, alpha, src2, beta, gamma, dst, dtype = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.addWeighted(
        cv.wrap_tensor(src1), alpha, cv.wrap_tensor(src2), 
        beta, gamma, cv.wrap_tensor(dst), dtype))
end

return cv
