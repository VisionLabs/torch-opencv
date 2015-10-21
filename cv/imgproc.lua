require 'cv'

local ffi = require 'ffi'

ffi.cdef[[

struct TensorWrapper getGaussianKernel(int ksize, double sigma, int ktype);

struct TensorArray getDerivKernels(
        int dx, int dy, int ksize,
        bool normalize, int ktype);

struct TensorWrapper getGaborKernel(struct SizeWrapper ksize, double sigma, double theta,
                                    double lambd, double gamma, double psi, int ktype);

struct TensorWrapper getStructuringElement(int shape, struct SizeWrapper ksize,
                                           struct PointWrapper anchor);

struct TensorWrapper medianBlur(struct TensorWrapper src, struct TensorWrapper dst, int ksize);

struct TensorWrapper GaussianBlur(struct TensorWrapper src, struct TensorWrapper dst,
                                  struct SizeWrapper ksize, double sigmaX,
                                  double sigmaY, int borderType);

struct TensorWrapper bilateralFilter(struct TensorWrapper src, struct TensorWrapper dst, int d,
                                     double sigmaColor, double sigmaSpace,
                                     int borderType);

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
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper map1, struct TensorWrapper map2,
        int interpolation, int borderMode, struct ScalarWrapper borderValue);

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
        struct IntArray channels, struct TensorWrapper mask,
        struct TensorWrapper hist, int dims, struct IntArray histSize,
        struct FloatArrayOfArrays ranges, bool uniform, bool accumulate);

struct TensorWrapper calcBackProject(
        struct TensorArray images, int nimages,
        struct IntArray channels, struct TensorWrapper hist,
        struct TensorWrapper backProject, struct FloatArrayOfArrays ranges,
        double scale, bool uniform);

double compareHist(
        struct TensorWrapper H1, struct TensorWrapper H2, int method);

struct TensorWrapper equalizeHist(
        struct TensorWrapper src, struct TensorWrapper dst);

float EMD(
        struct TensorWrapper signature1, struct TensorWrapper signature2,
        int distType, struct TensorWrapper cost,
        struct FloatArray lowerBound, struct TensorWrapper flow);

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

struct DoubleArray HuMoments(
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
        struct TensorWrapper points, bool clockwise, bool returnPoints);

struct TensorWrapper convexityDefects(
        struct TensorWrapper contour, struct TensorWrapper convexhull);

bool isContourConvex(
        struct TensorWrapper contour);

struct TensorPlusFloat intersectConvexConvex(
        struct TensorWrapper _p1, struct TensorWrapper _p2, bool handleNested);

struct RotatedRectWrapper fitEllipse(
        struct TensorWrapper points);

struct TensorWrapper fitLine(
        struct TensorWrapper points, int distType, double param, double reps, double aeps);

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

void rectanglePts(
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
]]


local C = ffi.load(libPath('imgproc'))


function cv.getGaussianKernel(t)
    local ksize = assert(t.ksize)
    local sigma = assert(t.sigma)
    local ktype = t.ktype or cv.CV_64F
    return cv.unwrap_tensor(C.getGaussianKernel(ksize, sigma, ktype))
end


function cv.getDerivKernels(t)
    local dx        = assert(t.dx)
    local dy        = assert(t.dy)
    local ksize     = assert(t.ksize)
    local ktype     = t.ktype or cv.CV_32F
    local normalize = t.normalize or false

    return cv.unwrap_tensors(C.getDerivKernels(dx, dy, ksize, normalize, ktype))
end


function cv.getGaborKernel(t)
    local ksize = cv.Size(assert(t.ksize))
    local sigma = assert(t.sigma)
    local theta = assert(t.theta)
    local lambd = assert(t.lambd)
    local gamma = assert(t.gamma)
    local psi   = t.psi or math.pi * 0.5
    local ktype = t.ktype or cv.CV_64F

    return cv.unwrap_tensors(
        C.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype))
end


function cv.getStructuringElement(t)
    local shape =  assert(t.shape)
    local ksize =  cv.Size(assert(t.ksize))
    local anchor = cv.Point(t.anchor or {-1, -1})

    return cv.unwrap_tensors(C.getStructuringElement(shape, ksize, anchor))
end


function cv.medianBlur(t)
    -- cv.medianBlur{src=X, dst=X, ksize} -- in-place
    -- cv.medianBlur{src=X, dst=Y, ksize} -- output to dst (must be of same size & type)
    -- cv.medianBlur{src=X, ksize}        -- output to return value
    local src =   assert(t.src)
    local dst =   assert(t.dst)
    local ksize = assert(t.ksize)

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

    return cv.unwrap_tensors(C.medianBlur(cv.wrap_tensors(src), cv.wrap_tensors(dst), ksize))
end


function cv.GaussianBlur(t)
    local src =        assert(t.src)
    local dst =        t.dst
    local ksize =      cv.Size(assert(t.ksize))
    local sigmaX =     assert(t.sigmaX)
    local sigmaY =     t.sigmaY or 0
    local borderType = t.borderType or cv.BORDER_DEFAULT

    assert(cv.tensorType(src) ~= cv.CV_8S and
           cv.tensorType(src) ~= cv.CV_32S)
    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.GaussianBlur(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), ksize, sigmaX, sigmaY, borderType))
end


function cv.bilateralFilter(t)
    local src =        assert(t.src)
    local dst =        t.dst
    local d =          assert(t.d)
    local sigmaColor = assert(t.sigmaColor)
    local sigmaSpace = assert(t.sigmaSpace)
    local borderType = t.borderType or cv.BORDER_DEFAULT

    assert(src:nDimension() == 2 or src:size()[3] == 3)

    local srcType = cv.tensorType(src)
    assert(srcType == cv.CV_8U or srcType == cv.CV_32F)

    if dst then
        assert(src ~= dst and dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.bilateralFilter(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), d, sigmaColor, sigmaSpace, borderType))
end


function cv.boxFilter(t)
    local src =        assert(t.src)
    local dst =        t.dst
    local ddepth =     assert(t.ddepth)
    local ksize =      cv.Size(assert(t.ksize))
    local anchor = cv.Point(t.anchor or {-1, -1})
    local normalize =  t.normalize
    if normalize == nil then normalize = true end
    local borderType = t.borderType or cv.BORDER_DEFAULT

    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.boxFilter(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), ddepth, ksize[1], 
            ksize[2], anchor, normalize, borderType))
end

function cv.sqrBoxFilter(t)

    local src = assert(t.src)
    local dst = t.dst
    local ddepth = assert(t.ddepth)
    local ksize = cv.Size(assert(t.ksize))
    local anchor = t.anchor or {-1,-1}
    local normalize =  t.normalize
    if normalize == nil then normalize = true end
    local borderType = t.borderType or cv.BORDER_DEFAULT

    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.sqrBoxFilter(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), ddepth, ksize, anchor, normalize, borderType))
end


function cv.blur(t)
    local src = assert(t.src)
    local dst = t.dst
    local ksize = cv.Size(assert(t.ksize))
    local anchor = cv.Point(t.anchor or {-1,-1})
    local borderType = t.borderType or cv.BORDER_DEFAULT

    assert(cv.tensorType(src) ~= cv.CV_8S and
           cv.tensorType(src) ~= cv.CV_32S)
    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.blur(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), ksize, anchor, borderType))
end


function cv.filter2D(t)
    local src = assert(t.src)
    local dst = t.dst
    local ddepth = assert(t.ddepth)
    local kernel = assert(t.kernel)
    local anchor = t.anchor or {-1,-1}
    local delta = t.delta or 0
    local borderType = t.borderType or cv.BORDER_DEFAULT

    assert(cv.checkFilterCombination(src, ddepth))
    if dst then
        assert(src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.filter2D(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), ddepth, cv.wrap_tensors(kernel), anchor, delta, borderType))
end


function cv.sepFilter2D(t)
    local src = assert(t.src)
    local dst = t.dst
    local ddepth = assert(t.ddepth)
    local kernelX = assert(t.kernelX)
    local kernelY = assert(t.kernelY)
    local anchor = t.anchor or {-1,-1}
    local delta = t.delta or 0
    local borderType = t.borderType or cv.BORDER_DEFAULT

    assert(cv.checkFilterCombination(src, ddepth))
    if dst then
        assert(src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.sepFilter2D(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), ddepth, kernelX, kernelY, anchor, delta, borderType))
end


function cv.Sobel(t)
    local src = assert(t.src)
    local dst = t.dst
    local ddepth = assert(t.ddepth)
    local dx = assert(t.dx)
    local dy = assert(t.dy)
    local ksize = t.ksize or 3
    local scale = t.scale or 1
    local delta = t.delta or 0
    local borderType = t.borderType or cv.BORDER_DEFAULT

    assert(cv.checkFilterCombination(src, ddepth))
    if dst then
        assert(src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.Sobel(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), ddepth, dx, dy, ksize, scale, delta, borderType))
end


function cv.Scharr(t)
    local src = assert(t.src)
    local dst = t.dst
    local ddepth = assert(t.ddepth)
    local dx = assert(t.dx)
    local dy = assert(t.dy)
    local scale = t.scale or 1
    local delta = t.delta or 0
    local borderType = t.borderType or cv.BORDER_DEFAULT

    assert(cv.checkFilterCombination(src, ddepth))
    if dst then
        assert(src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.Scharr(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), ddepth, dx, dy, scale, delta, borderType))
end


function cv.Laplacian(t)
    local src = assert(t.src)
    local dst = t.dst
    local ddepth = assert(t.ddepth)
    local ksize = t.ksize or 1
    local scale = t.scale or 1
    local delta = t.delta or 0
    local borderType = t.borderType or cv.BORDER_DEFAULT

    assert(cv.checkFilterCombination(src, ddepth))
    if dst then
        assert(src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.Laplacian(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), ddepth, ksize, scale, delta, borderType))
end


function cv.Canny(t)
    local image = assert(t.image)
    local edges = t.edges
    local threshold1 = assert(t.threshold1)
    local threshold2 = assert(t.threshold2)
    local apertureSize = t.apertureSize or 3
    local L2gradient = t.L2gradient or false

    assert(cv.tensorType(image) == cv.CV_8U)

    if edges then
        assert(edges:nDimension() == 2 and
               edges:size()[1] == image:size()[1] and
               edges:size()[2] == image:size()[2])
    end

    return cv.unwrap_tensors(
        C.Canny(
            cv.wrap_tensors(image), cv.wrap_tensors(edges), threshold1, threshold2, apertureSize, L2gradient))
end


function cv.cornerMinEigenVal(t)
    local src = assert(t.src)
    local dst = t.dst
    local blockSize = assert(t.blockSize)
    local ksize = t.ksize or 3
    local borderType = t.borderType or cv.BORDER_DEFAULT

    local srcType = cv.tensorType(src)
    assert(src:nDimension() == 2 and (srcType == cv.CV_8U or srcType == cv.CV_32F))
    if dst then
        assert(dst:isSameSizeAs(src) and cv.tensorType(dst) == cv.CV_32F)
    end

    return cv.unwrap_tensors(
        C.cornerMinEigenVal(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), blockSize, ksize, borderType))
end


function cv.cornerHarris(t)
    local src = assert(t.src)
    local dst = t.dst
    local blockSize = assert(t.blockSize)
    local ksize = assert(t.ksize)
    local k = assert(t.k)
    local borderType = t.borderType or cv.BORDER_DEFAULT

    local srcType = cv.tensorType(src)
    assert(src:nDimension() == 2 and (srcType == cv.CV_8U or srcType == cv.CV_32F))

    if dst then
        assert(dst:isSameSizeAs(src) and cv.tensorType(dst) == cv.CV_32F)
    end

    return cv.unwrap_tensors(
        C.cornerHarris(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), blockSize, ksize, k, borderType))
end


function cv.cornerEigenValsAndVecs(t)
    local src = assert(t.src)
    local dst = t.dst
    local blockSize = assert(t.blockSize)
    local ksize = assert(t.ksize)
    local borderType = t.borderType or cv.BORDER_DEFAULT

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
            cv.wrap_tensors(src), cv.wrap_tensors(dst), blockSize, ksize, borderType))
end


function cv.preCornerDetect(t)
    local src = assert(t.src)
    local dst = t.dst
    local ksize = assert(t.ksize)
    local borderType = t.borderType or cv.BORDER_DEFAULT

    local srcType = cv.tensorType(src)
    assert(src:nDimension() == 2 and (srcType == cv.CV_8U or srcType == cv.CV_32F))

    if dst then
        assert(cv.tensorType(dst) == cv.CV_32F and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.preCornerDetect(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), ksize, borderType))
end


function cv.HoughLines(t)
    local image = assert(t.image)
    local rho = assert(t.rho)
    local theta = assert(t.theta)
    local threshold = assert(t.threshold)
    local srn = t.srn or 0
    local stn = t.stn or 0
    local min_theta = t.min_theta or 0
    local max_theta = t.max_theta or cv.CV_PI

    assert(image:nDimension() == 2 and cv.tensorType(image) == cv.CV_8U)

    return cv.unwrap_tensors(
        C.HoughLines(
            cv.wrap_tensors(image), rho, theta, threshold, srn, stn, min_theta, max_theta))
end


function cv.HoughLinesP(t)
    local image = assert(t.image)
    local rho = assert(t.rho)
    local theta = assert(t.theta)
    local threshold = assert(t.threshold)
    local minLineLength = t.minLineLength or 0
    local maxLineGap = t.maxLineGap or 0

    assert(image:nDimension() == 2 and cv.tensorType(image) == cv.CV_8U)

    return cv.unwrap_tensors(
        C.HoughLinesP(
            cv.wrap_tensors(image), rho, theta, threshold, minLineLength, maxLineGap))
end


function cv.HoughCircles(t)
    local image = assert(t.image)
    local method = assert(t.method)
    local dp = assert(t.dp)
    local minDist = assert(t.minDist)
    local param1 = t.param1 or 100
    local param2 = t.param2 or 100
    local minRadius = t.minRadius or 0
    local maxRadius = t.maxRadius or 0

    assert(image:nDimension() == 2 and cv.tensorType(image) == cv.CV_8U)

    return cv.unwrap_tensors(
        C.HoughCircles(
            cv.wrap_tensors(image), method, dp, minDist, param1, param2, minRadius, maxRadius))
end


function cv.cornerSubPix(t)
    local image = assert(t.image)
    local corners = assert(t.corners)
    local winSize = cv.Size(assert(t.winSize))
    local zeroZone = cv.Size(assert(t.zeroZone))
    local criteria = cv.TermCriteria(assert(t.criteria))

    assert(image:nDimension() == 2)
    assert(corners:size()[2] == 2 and cv.tensorType(corners) == cv.CV_32F)

    C.cornerSubPix(
        cv.wrap_tensors(image), cv.wrap_tensors(corners), winSize,
        zeroZone, criteria)
end


function cv.goodFeaturesToTrack(t)
    local image = assert(t.image)
    local maxCorners = assert(t.maxCorners)
    local qualityLevel = assert(t.qualityLevel)
    local minDistance = assert(t.minDistance)
    local mask = t.mask
    local blockSize = t.blockSize or 3
    local useHarrisDetector = t.useHarrisDetector or false
    local k = t.k or 0.04

    local imgType = cv.tensorType(image)
    assert(image:nDimension() == 2 and (imgType == cv.CV_32F or imgType == cv.CV_8U))

    if mask then
        assert(cv.tensorType(mask) == cv.CV_8U and mask:isSameSizeAs(image))
    end

    return cv.unwrap_tensors(
        C.goodFeaturesToTrack(
            cv.wrap_tensors(image), maxCorners, qualityLevel, minDistance, cv.wrap_tensors(mask), blockSize, useHarrisDetector, k))
end


function cv.erode(t)
    local src =        assert(t.src)
    local dst =        t.dst
    local kernel = assert(t.kernel)
    local iterations = 1
    local anchor = cv.Point(t.anchor or {-1, -1})
    local borderType = t.borderType or cv.BORDER_CONSTANT
    local borderValue = cv.Scalar(t.borderValue or {0/0}) -- pass nan to detect default value

    return cv.unwrap_tensors(
        C.erode(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), cv.wrap_tensors(kernel), anchor,
            iterations, borderType, borderValue))
end


function cv.dilate(t)
    local src =        assert(t.src)
    local dst =        t.dst
    local kernel = assert(t.kernel)
    local iterations = 1
    local anchor = cv.Point(t.anchor or {-1, -1})
    local borderType = t.borderType or cv.BORDER_CONSTANT
    local borderValue = cv.Scalar(t.borderValue or {0/0}) -- pass nan to detect default value

    return cv.unwrap_tensors(
        C.dilate(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), cv.wrap_tensors(kernel), anchor,
            iterations, borderType, borderValue))
end


function cv.morphologyEx(t)
    local src = assert(t.src)
    local dst = t.dst
    local op = assert(t.op)
    local kernel = assert(t.kernel)
    local iterations = 1
    local anchor = cv.Point(t.anchor or {-1, -1})
    local borderType = t.borderType or cv.BORDER_CONSTANT
    local borderValue = cv.Scalar(t.borderValue or {0/0}) -- pass nan to detect default value

    return cv.unwrap_tensors(
        C.morphologyEx(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), op, cv.wrap_tensors(kernel), anchor,
            iterations, borderType, borderValue))
end


function cv.resize(t)
    local src = assert(t.src)
    local dst = t.dst
    local dsize = cv.Size(t.dsize or {0, 0})
    local fx = t.fx or 0
    local fy = t.fy or 0
    local interpolation = t.interpolation or cv.INTER_LINEAR

    return cv.unwrap_tensors(
        C.resize(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), dsize, fx, fy, interpolation))
end


function cv.warpAffine(t)
    local src = assert(t.src)
    local dst = t.dst
    local M = assert(t.M)
    local dsize = cv.Point(t.dsize or {0, 0})
    local flags = t.flags or cv.INTER_LINEAR
    local borderMode = t.borderMode or cv.BORDER_CONSTANT
    local borderValue = cv.Scalar(t.borderValue or {0/0}) -- pass nan to detect default value

    return cv.unwrap_tensors(
        C.warpAffine(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), cv.wrap_tensors(M), dsize,
            flags, borderMode, borderValue))
end


function cv.warpPerspective(t)
    local src = assert(t.src)
    local dst = t.dst
    local M = assert(t.M)
    local dsize = cv.Size(t.dsize or {0, 0})
    local flags = t.flags or cv.INTER_LINEAR
    local borderMode = t.borderMode or cv.BORDER_CONSTANT
    local borderValue = cv.Scalar(t.borderValue or {0/0}) -- pass nan to detect default value

    return cv.unwrap_tensors(
        C.warpPerspective(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), cv.wrap_tensors(M), dsize,
            flags, borderMode, borderValue))
end


function cv.remap(t)
    local src = assert(t.src)
    local dst = t.dst
    local map1 = assert(t.map1)
    local map2 = assert(t.map2)
    local interpolation = assert(t.interpolation)
    local borderMode = t.borderMode or cv.BORDER_CONSTANT
    local borderValue = cv.Scalar(t.borderValue or {0, 0, 0, 0})

    return cv.unwrap_tensors(
        C.remap(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), cv.wrap_tensors(map1), cv.wrap_tensors(map2),
            interpolation, borderMode, borderValue))
end


function cv.convertMaps(t)
    local map1 = assert(t.map1)
    local map2 = t.map2
    local dstmap1 = assert(t.dstmap1)
    local dstmap2 = assert(t.dstmap2)
    local dstmap1type = assert(t.dstmap1type)
    local nninterpolation = t.nninterpolation or false

    return cv.unwrap_tensors(
        C.convertMaps(
            cv.wrap_tensors(map1), cv.wrap_tensors(map2), cv.wrap_tensors(dstmap1),
            cv.wrap_tensors(dstmap2), dstmap1type, nninterpolation))
end


function cv.getRotationMatrix2D(t)
    local center = cv.Point2f(assert(t.center))
    local angle = assert(t.angle)
    local scale = assert(t.scale)

    return cv.unwrap_tensors(
        C.getRotationMatrix2D(center, angle, scale))
end


function cv.invertAffineTransform(t)
    local M = assert(t.M)
    local iM = t.iM

    return cv.unwrap_tensors(
        C.invertAffineTransform(cv.wrap_tensors(M), cv.wrap_tensors(iM)))
end


function cv.getPerspectiveTransform(t)
    local src = assert(t.src)
    local dst = assert(t.dst)

    return cv.unwrap_tensors(
        C.getPerspectiveTransform(cv.wrap_tensors(src), cv.wrap_tensors(dst)))
end


function cv.getAffineTransform(t)
    local src = assert(t.src)
    local dst = assert(t.dst)

    return cv.unwrap_tensors(
        C.getAffineTransform(cv.wrap_tensors(src), cv.wrap_tensors(dst)))
end


function cv.getRectSubPix(t)
    local image = assert(t.src)
    local patchSize = cv.Size(assert(t.patchSize))
    local center = cv.Point2f(assert(t.center))
    local patch = t.patch
    local patchType = t.patchType or -1

    return cv.unwrap_tensors(
        C.getRectSubPix(cv.wrap_tensors(image), patchSize, center,
                        cv.wrap_tensors(patch), patchType))
end


function cv.logPolar(t)
    local src = assert(t.src)
    local dst = t.dst
    local center = cv.Point2f(assert(t.center))
    local M = assert(t.M)
    local flags = assert(t.flags)

    return cv.unwrap_tensors(
        C.logPolar(cv.wrap_tensors(src), cv.wrap_tensors(dst), center, M, flags))
end


function cv.linearPolar(t)
    local src = assert(t.src)
    local dst = t.dst
    local center = cv.Point2f(assert(t.center))
    local maxRadius = assert(t.maxRadius)
    local flags = assert(t.flags)

    return cv.unwrap_tensors(
        C.linearPolar(cv.wrap_tensors(src), cv.wrap_tensors(dst), center, maxRadius, flags))
end


function cv.integral(t)
    local src = assert(t.src)
    local sum = t.sum
    local sdepth = t.sdepth or -1

    return cv.unwrap_tensors(
        C.integral(cv.wrap_tensors(src), cv.wrap_tensors(sum), sdepth))
end


function cv.integral2(t)
    local src = assert(t.src)
    local sum = t.sum
    local sqsum = t.sqsum
    local sdepth = t.sdepth or -1
    local sqdepth = t.sqdepth or -1

    return cv.unwrap_tensors(
        C.integralN(cv.wrap_tensors(src), cv.wrap_tensors(sum, sqsum), sdepth, sqdepth))
end


function cv.integral3(t)
    local src = assert(t.src)
    local sum = t.sum
    local sqsum = t.sqsum
    local tilted = t.tilted
    local sdepth = t.sdepth or -1
    local sqdepth = t.sqdepth or -1

    return cv.unwrap_tensors(
        C.integralN(cv.wrap_tensors(src), cv.wrap_tensors(sum, sqsum, tilted), sdepth, sqdepth))
end


function cv.accumulate(t)
    local src = assert(t.src)
    local sum = assert(t.sum)
    local mask = t.mask

    C.accumulate(cv.wrap_tensors(src), cv.wrap_tensors(sum), cv.wrap_tensors(mask))
end


function cv.accumulateSquare(t)
    local src = assert(t.src)
    local sum = assert(t.sum)
    local mask = t.mask

    C.accumulateSquare(cv.wrap_tensors(src), cv.wrap_tensors(sum), cv.wrap_tensors(mask))
end


function cv.accumulateProduct(t)
    local src1 = assert(t.src1)
    local src2 = assert(t.src2)
    local sum = assert(t.sum)
    local mask = t.mask

    C.accumulateSquare(cv.wrap_tensors(src1), cv.wrap_tensors(src2), cv.wrap_tensors(sum), cv.wrap_tensors(mask))
end


function cv.accumulateWeighted(t)
    local src = assert(t.src)
    local sum = assert(t.sum)
    local alpha = assert(t.alpha)
    local mask = t.mask

    C.accumulateWeighted(cv.wrap_tensors(src), cv.wrap_tensors(sum), alpha, cv.wrap_tensors(mask))
end


-- point, response = cv.phaseCorrelate{...}
-- point    -> Point
-- response -> number
function cv.phaseCorrelate(t)
    local src1 = assert(t.src1)
    local src2 = assert(t.src2)
    local window = t.window

    local result = C.phaseCorrelate(cv.wrap_tensors(src1), cv.wrap_tensors(src2), cv.wrap_tensors(window))
    return {x=result.v0, y=result.v1}, result.v2
end


function cv.createHanningWindow(t)
    local dst = t.dst
    local winSize = cv.Size(assert(t.winSize))
    local type = assert(t.type)

    if dst then
        assert(cv.tensorType(dst) == type)
        assert(dst:size()[1] == winSize[1] and
               dst:size()[2] == winSize[2])
    end

    return cv.unwrap_tensors(
        C.createHanningWindow(cv.wrap_tensors(dst), winSize, type))
end


-- value, binarized = cv.threshold{...}
-- value     -> number
-- binarized -> Tensor
function cv.threshold(t)
    local src = assert(t.src)
    local dst = t.dst
    local tresh = assert(t.tresh)
    local maxval = assert(t.maxval)
    local type = assert(t.type)

    if dst then
        assert(cv.tensorType(dst) == type)
        assert(dst:isSameSizeAs(src))
    end

    local result = C.threshold(cv.wrap_tensors(src), cv.wrap_tensors(dst),
                    tresh, maxval, type)
    return result.val, cv.unwrap_tensors(result.tensor)
end


function cv.adaptiveThreshold(t)
    local src = assert(t.src)
    local dst = t.dst
    local maxValue = assert(t.maxValue)
    local adaptiveMethod = assert(t.adaptiveMethod)
    local thresholdType = assert(t.thresholdType)
    local blockSize = assert(t.blockSize)
    local C = assert(t.C)

    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.adaptiveThreshold(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), maxValue,
                            adaptiveMethod, thresholdType, blockSize))
end


function cv.pyrDown(t)
    local src = assert(t.src)
    local dst = t.dst
    local dstSize = cv.Size(t.dstSize or {0,0})
    local borderType = t.borderType or cv.BORDER_DEFAULT

    if dst then
        assert(dst:type() == src:type())
        assert(dst:size()[1] == dstSize[1] and
               dst:size()[2] == dstSize[2])
    end

    return cv.unwrap_tensors(
        C.pyrDown(
            cv.wrap_tensors(src), cv.wrap_tensors(dst),
            dstSize, borderType))
end


function cv.pyrUp(t)
    local src = assert(t.src)
    local dst = t.dst
    local dstSize = cv.Size(t.dstSize or {0,0})
    local borderType = t.borderType or cv.BORDER_DEFAULT

    if dst then
        assert(dst:type() == src:type())
        assert(dst:size()[1] == dstSize[1] and
                dst:size()[2] == dstSize[2])
    end

    return cv.unwrap_tensors(
        C.pyrUp(
            cv.wrap_tensors(src), cv.wrap_tensors(dst),
            dstSize, borderType))
end


function cv.buildPyramid(t)
    local src = assert(t.src)
    local dst = t.dst or cv.wrap_tensors(dst)
    local maxlevel = assert(t.maxlevel)
    local borderType = t.borderType or cv.BORDER_DEFAULT

    if dst then
        assert(#dst == maxlevel + 1)
        for i, tensor in ipairs(dst) do
            assert(tensor:type() == src:type())
        end
    end

    return cv.unwrap_tensors(
        C.buildPyramid(cv.wrap_tensors(src), dst, maxlevel, borderType), 
        true)
end


function cv.undistort(t)
    local src = assert(t.src)
    local dst = t.dst
    local cameraMatrix = assert(t.cameraMatrix)
    if type(cameraMatrix) == "table" then
        cameraMatrix = torch.FloatTensor(cameraMatrix)
    end
    local distCoeffs = t.distCoeffs
    if type(distCoeffs) == "table" then
        distCoeffs = torch.FloatTensor(distCoeffs)
    end
    local newCameraMatrix = t.newCameraMatrix
    if type(newCameraMatrix) == "table" then
        newCameraMatrix = torch.FloatTensor(newCameraMatrix)
    end

    if dst then
        assert(src:type() == dst:type())
        assert(src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.undistort(cv.wrap_tensors(src), cv.wrap_tensors(dst), cv.wrap_tensors(cameraMatrix), 
                    cv.wrap_tensors(distCoeffs), cv.wrap_tensors(newCameraMatrix)))
end



function cv.initUndistortRectifyMap(t)
    local cameraMatrix = assert(t.cameraMatrix)
    if type(cameraMatrix) == "table" then
        cameraMatrix = torch.FloatTensor(cameraMatrix)
    end
    local distCoeffs = t.distCoeffs
    if type(distCoeffs) == "table" then
        distCoeffs = torch.FloatTensor(distCoeffs)
    end
    local R = t.R
    if type(R) == "table" then
        R = torch.FloatTensor(R)
    end
    local newCameraMatrix = t.newCameraMatrix
    if type(newCameraMatrix) == "table" then
        newCameraMatrix = torch.FloatTensor(newCameraMatrix)
    end
    local size = cv.Size(assert(t.size))
    local m1type = assert(t.m1type)
    local maps = t.maps

    return cv.unwrap_tensors(
        C.initUndistortRectifyMap(
            cv.wrap_tensors(cameraMatrix), cv.wrap_tensors(distCoeffs), 
            cv.wrap_tensors(R), cv.wrap_tensors(newCameraMatrix),
            size, m1type, cv.wrap_tensors(maps)))
end

-- value = cv.initWideAngleProjMap{maps={map1, map2}, ...}
-- OR
-- value, map1, map2 = cv.initWideAngleProjMap{...}
--
-- value      -> number
-- map1, map2 -> Tensor
function cv.initWideAngleProjMap(t)
    local cameraMatrix = assert(t.cameraMatrix)
    if type(cameraMatrix) == "table" then
        cameraMatrix = torch.FloatTensor(cameraMatrix)
    end
    local distCoeffs = t.distCoeffs
    if type(distCoeffs) == "table" then
        distCoeffs = torch.FloatTensor(distCoeffs)
    end
    local imageSize = cv.Size(assert(t.imageSize))
    local destImageWidth = assert(t.destImageWidth)
    local m1type = assert(t.m1type)
    local maps = t.maps
    local projType = assert(t.projType)
    local alpha = assert(t.alpha)
    
    local result = C.initWideAngleProjMap(
        cv.wrap_tensors(cameraMatrix), cv.wrap_tensors(distCoeffs),
        imageSize, destImageWidth, m1type, cv.wrap_tensors(maps),
        projType, alpha)
    return result.val, cv.unwrap_tensors(result.tensors)
end


function cv.calcHist(t)
    local images = assert(t.images)
    assert(type(images) == "table")
    local channels = assert(t.channels)
    if type(channels) == "table" then
        channels = cv.newArray('Int', channels)
    end
    local mask = t.mask
    local hist = t.hist
    local dims = assert(t.dims)
    local histSize = assert(t.histSize)
    if type(histSize) == "table" then
        histSize = cv.newArray('Int', histSize)
    end
    local ranges = assert(t.ranges)
    if type(ranges) == "table" then
        ranges = cv.FloatArrayOfArrays(ranges)
    end
    local uniform = t.uniform
    if uniform == nil then 
        uniform = true 
    end
    local accumulate = t.accumulate or false
    assert(hist or accumulate == false)

    return cv.unwrap_tensors(
        C.calcHist(
            cv.wrap_tensors(images), channels, cv.wrap_tensors(mask),
            cv.wrap_tensors(hist), dims, histSize, ranges, uniform, accumulate))
end


function cv.calcBackProject(t)
    local images = assert(t.images)
    assert(type(images) == "table")
    local nimages = assert(t.nimages)
    local channels = assert(t.channels)
    if type(channels) == "table" then
        channels = cv.newArray('Int', channels)
    end
    local hist = assert(t.hist)
    local backProject = t.backProject
    local ranges = assert(t.ranges)
    if type(ranges) == "table" then
        ranges = cv.FloatArrayOfArrays(ranges)
    end
    local scale = t.scale or 1
    local uniform = t.uniform
    if uniform == nil then 
        uniform = true 
    end
    
    return cv.unwrap_tensors(
        C.calcBackProject(
            cv.wrap_tensors(images), nimages, channels, cv.wrap_tensors(hist),
            cv.wrap_tensors(backProject), ranges, scale, uniform))
end


function cv.compareHist(t)
    local H1 = assert(t.H1)
    local H2 = assert(t.H2)
    local method = assert(t.method)

    return C.compareHist(cv.wrap_tensors(H1), cv.wrap_tensors(H2), method)
end


function cv.equializeHist(t)
    local src = assert(t.src)
    local dst = t.dst

    return cv.unwrap_tensors(
        C.equializeHist(
            cv.wrap_tensors(src), cv.wrap_tensors(dst)))
end


function cv.EMD(t)
    local signature1 = assert(t.signature1)
    local signature2 = assert(t.signature2)
    local distType = assert(t.distType)
    local cost = t.cost
    local lowerBound = t.lowerBound or ffi.new('struct FloatArray', nil)
    if type(lowerBound) == "table" then
        lowerBound = cv.newArray(lowerBound)
    end
    local flow = t.flow

    return C.EMD(
        cv.wrap_tensors(signature1), cv.wrap_tensors(signature2), distType,
        cv.wrap_tensors(cost), lowerBound, cv.wrap_tensors(flow))
end


function cv.watershed(t)
    local image = assert(t.image)
    local markers = assert(t.markers)

    C.watershed(cv.wrap_tensors(image), cv.wrap_tensors(markers))
end


function cv.pyrMeanShiftFiltering(t)
    local src = assert(t.src)
    local dst = t.dst
    local sp = assert(t.sp)
    local sr = assert(t.sr)
    local maxLevel = t.maxLevel or 1
    local termcrit = cv.TermCriteria(t.termcrit)

    return cv.unwrap_tensors(C.pyrMeanShiftFiltering(
        cv.wrap_tensors(src), cv.wrap_tensors(dst), sp, sr, maxlevel, termcrit))
end


function cv.grabCut(t)
    local img = assert(t.img)
    local mask = assert(t.mask)
    local rect = cv.Rect(t.rect)
    local bgdModel = assert(t.bgdModel)
    local fgdModel = assert(t.fgdModel)
    local iterCount = assert(t.iterCount)
    local mode = t.mode or cv.GC_EVAL
    
    C.grabCut(
        cv.wrap_tensors(img), cv.wrap_tensors(mask), rect,
        cv.wrap_tensors(bgdModel), cv.wrap_tensors(fgdModel),
        iterCount, mode)
end


function cv.distanceTransform(t)
    local src = assert(t.src)
    local dst = t.dst
    local distanceType = assert(t.distanceType)
    local maskSize = assert(t.maskSize)
    local dstType = t.dstType or cv.CV_32F

    return cv.unwrap_tensors(
        C.distanceTransform(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), distanceType, maskSize, dstType))
end


function cv.distanceTransformWithLabels(t)
    local src = assert(t.src)
    local dst = t.dst
    local labels = t.labels
    local distanceType = assert(t.distanceType)
    local maskSize = assert(t.maskSize)
    local labelType = t.labelType or cv.DIST_LABEL_CCOMP

    return cv.unwrap_tensors(
        C.distanceTransformWithLabels(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), 
            cv.wrap_tensors(labels), distanceType, maskSize, labelType)) 
end

-- area, boundingRect = cv.floodFill{...}
-- area         -> number
-- boundingRect -> RectWrapper
function cv.floodFill(t)
    local image = assert(t.image)
    local mask = t.mask
    local seedPoint = cv.Point(assert(t.seedPoint))
    local newVal = cv.Scalar(assert(t.newVal))
    local loDiff = cv.Scalar(t.loDiff or {0, 0, 0, 0})
    local upDiff = cv.Scalar(t.upDiff or {0, 0, 0, 0})
    local flags = t.flags or 4

    local result = C.floodFill(
        cv.wrap_tensors(image), cv.wrap_tensors(mask), seedPoint[1], 
        seedPoint[2], newVal, loDiff, upDiff, flags)
    return result.val, result.rect
end


function cv.cvtColor(t)
    local src = assert(t.src)
    local dst = t.dst
    local code = assert(t.code)
    local dstCn = t.dstCn or 0

    return cv.unwrap_tensors(C.cvtColor(
        cv.wrap_tensors(src), cv.wrap_tensors(dst), code, dstCn))
end


function cv.demosaicing(t)
    local _src = assert(t.src)
    local _dst = assert(t.dst)
    local code = assert(t.code)
    local dcn = t.dcn or 0

    return cv.unwrap_tensors(C.demosaicing(
        cv.wrap_tensors(_src), cv.wrap_tensors(_dst), code, dcn))
end


function cv.moments(t)
    local array = assert(t.array)
    local binaryImage = t.binaryImage or false

    return C.moments(cv.wrap_tensors(array), binaryImage)
end


-- moments: Input moments computed with cv.moments()
-- toTable: Output to table if true, otherwise output to Tensor. Default: true
-- output : Optional. A Tensor of length 7 or a table; if provided, will output there
function cv.HuMoments(t)
    local moments = assert(t.moments)
    local outputType = t.outputType or 'table'
    local output = t.output

    return cv.arrayToLua(C.HuMoments(moments), outputType, output)
end


function cv.matchTemplate(t)
    local image = assert(t.image)
    local templ = t.templ
    local result = assert(t.result)
    local method = assert(t.method)
    local mask = t.mask

    return cv.unwrap_tensors(C.matchTemplate(
        cv.wrap_tensors(image), cv.wrap_tensors(templ), cv.wrap_tensors(result), method, cv.wrap_tensors(mask)))
end


function cv.connectedComponents(t)
    local image = assert(t.image)
    local labels = assert(t.labels)
    local connectivity = t.connectivity or 8
    local ltype = t.ltype or cv.CV_32S

    local result = C.connectedComponents(cv.wrap_tensors(image), cv.wrap_tensors(labels), connectivity, ltype)
    return result.val, cv.unwrap_tensors(result.tensor)
end


function cv.connectedComponentsWithStats(t)
    local image = assert(t.image)
    local labels = assert(t.labels)
    local stats = assert(t.stats)
    local centroids = assert(t.centroids)
    local connectivity = t.connectivity or 8
    local ltype = t.ltype or cv.CV_32S

    local result = C.connectedComponentsWithStats(
        cv.wrap_tensors(image), 
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
    local image = assert(t.image)
    local hierarchy = t.hierarchy
    local withHierarchy = t.withHierarchy or false
    local mode = assert(t.mode)
    local method = assert(t.method)
    local offset = cv.Point(t.offset or {0, 0})

    contours = cv.unwrap_tensors(
        C.findContours(
            cv.wrap_tensors(image), withHierarchy, cv.wrap_tensors(hierarchy), mode, method, offset), true)

    if withHierarchy and not hierarchy then
        hierarchy = contours[#contours]
        contours[#contours] = nil
        return contours, hierarchy
    else
        return contours
    end
end


function cv.approxPolyDP(t)
    local curve = assert(t.curve)
    local approxCurve = t.approxCurve
    local epsilon = assert(t.epsilon)
    local closed = assert(t.closed)

    return cv.unwrap_tensors(C.approxPolyDP(cv.wrap_tensors(curve), cv.wrap_tensors(approxCurve), epsilon, closed))
end


function cv.arcLength(t)
    local curve = assert(t.curve)
    local closed = assert(t.closed)
    
    return C.arcLength(cv.wrap_tensors(curve), closed)
end


function cv.boundingRect(t)
    local points = assert(t.points)
    if type(points) == "table" then
        points = torch.FloatTensor(points)
    end

    return C.boundingRect(cv.wrap_tensors(points))
end


function cv.contourArea(t)
    local contour = assert(t.contour)
    if type(contour) == "table" then
        contour = torch.FloatTensor(contour)
    end

    local oriented = t.oriented or false

    return C.contourArea(cv.wrap_tensors(contour), oriented)
end


function cv.minAreaRect(t)
    local points = assert(t.points)
    if type(points) == "table" then
        points = torch.FloatTensor(points)
    end

    return C.minAreaRect(cv.wrap_tensors(points))
end


-- box: a RotatedRectWrapper
-- points: optional; a 4x2 Tensor to hold the return value
function cv.boxPoints(t)
    local box = assert(t.box)
    local points = t.points
    -- check that points is a Tensor
    assert(not points or points.torch)

    return cv.unwrap_tensors(C.boxPoints(box, cv.wrap_tensors(points)))
end

-- points: a Tensor or a table of points
-- return value: center, radius
function cv.minEnclosingCircle(t)
    local points = assert(t.points)
    if type(points) == "table" then
        points = torch.FloatTensor(points)
    end

    local result = C.minEnclosingCircle(cv.wrap_tensors(points))
    return cv.Point2f(result.v0, result.v1), result.v2
end

-- points: a Tensor or a table of points
-- triangle: optional; a 3x2 Tensor to hold the return value
-- return value: triangle_points, area
function cv.minEnclosingTriangle(t)
    local points = assert(t.points)
    if type(points) == "table" then
        points = torch.FloatTensor(points)
    end
    local triangle = t.triangle
    -- check that triangle is a Tensor
    assert(not triangle or triangle.torch)

    local result = C.minEnclosingTriangle(cv.wrap_tensors(points), cv.wrap_tensors(triangle))
    return cv.unwrap_tensors(result.tensor), result.val
end


function cv.matchShapes(t)
    local contour1 = assert(t.contour1)
    local contour2 = assert(t.contour2)
    local method = assert(t.method)
    local parameter = t.parameter or 0

    return C.matchShapes(cv.wrap_tensors(contour1), cv.wrap_tensors(contour2), method, parameter)
end


function cv.convexHull(t)
    local points = assert(t.points)
    local clockwise = t.clockwise or false
    local returnPoints = t.returnPoints
    if returnPoints == nil then
         returnPoints = true
    end

    retval = cv.unwrap_tensors(C.convexHull(cv.wrap_tensors(points), clockwise, returnPoints))
    if not returnPoints then
        -- correct the 0-based indexing
        for i = 1,#retval do
            retval[i] = retval[i] + 1
        end
    end
    return retval
end


function cv.convexityDefects(t)
    local contour = assert(t.contour)
    local convexhull = assert(t.convexhull)

    return cv.unwrap_tensors(C.convexityDefects(cv.wrap_tensors(contour), cv.wrap_tensors(convexhull)))
end


function cv.isContourConvex(t)
    local contour = assert(t.contour)

    return C.isContourConvex(cv.wrap_tensors(contour))
end


-- contour1, contour2: Tensors containing points
-- handleNested: boolean
-- return value: intersection
function cv.intersectConvexConvex(t)
    local _p1 = assert(t.contour1)
    local _p2 = assert(t.contour2)
    local handleNested = t.handleNested
    if handleNested == nil then
        handleNested = true
    end

    return cv.unwrap_tensors(
        C.intersectConvexConvex(cv.wrap_tensors(_p1), cv.wrap_tensors(_p2), handleNested))
end


function cv.fitEllipse(t)
    local points = assert(t.points)

    return C.fitEllipse(cv.wrap_tensors(points))
end


function cv.fitLine(t)
    local points = assert(t.points)
    local distType = assert(t.distType)
    local param = assert(t.param)
    local reps = assert(t.reps)
    local aeps = assert(t.aeps)

    return cv.unwrap_tensors(
        C.fitLine(cv.wrap_tensors(points), distType, param, reps, aeps))
end


function cv.pointPolygonTest(t)
    local contour = assert(t.contour)
    local pt = cv.Point2f(assert(t.pt))
    local measureDist = assert(t.measureDist)

    return C.pointPolygonTest(cv.wrap_tensors(contour), pt, measureDist)
end


function cv.rotatedRectangleIntersection(t)
    local rect1 = cv.RotatedRect(assert(t.rect1))
    local rect2 = cv.RotatedRect(assert(t.rect2))

    return cv.unwrap_tensors(
        C.rotatedRectangleIntersection(rect1, rect2))
end


function cv.blendLinear(t)
    local src1 = assert(t.src1)
    local src2 = assert(t.src2)
    local weights1 = assert(t.weights1)
    local weights2 = assert(t.weights2)
    local dst = t.dst

    return cv.unwrap_tensors(
        C.blendLinear(cv.wrap_tensors(src1), cv.wrap_tensors(src2), cv.wrap_tensors(weights1), cv.wrap_tensors(weights2), cv.wrap_tensors(dst)))
end


function cv.applyColorMap(t)
    local src = assert(t.src)
    local dst = t.dst
    local colormap = assert(t.colormap)

    return cv.unwrap_tensors(
        C.applyColorMap(cv.wrap_tensors(src), cv.wrap_tensors(dst), colormap))
end


function cv.line(t)
    local img = assert(t.img)
    local pt1 = cv.Point(assert(t.pt1))
    local pt2 = cv.Point(assert(t.pt2))
    local color = cv.Scalar(assert(t.color))
    local thickness = t.thickness or 1
    local lineType = t.lineType or cv.LINE_8
    local shift = t.shift or 0

    C.line(cv.wrap_tensors(img), pt1, pt2, color, thickness, lineType, shift)
end


function cv.arrowedLine(t)
    local img = assert(t.img)
    local pt1 = cv.Point(assert(t.pt1))
    local pt2 = cv.Point(assert(t.pt2))
    local color = cv.Scalar(assert(t.color))
    local thickness = t.thickness or 1
    local line_type = t.line_type or 8
    local shift = t.shift or 0
    local tipLength = t.tipLength or 0.1

    C.arrowedLine(cv.wrap_tensors(img), pt1, pt2, color, thickness, line_type, shift, tipLength)
end


function cv.rectangle(t)
    local img = assert(t.img)
    local pt1
    local pt2
    local rec
    if t.rec then
        rec = cv.Rect(t.rec)
    else
        pt1 = cv.Point(t.pt1)
        pt2 = cv.Point(t.pt2)
    end
    assert((pt1 and pt2) or rec)
    local color = cv.Scalar(assert(t.color))
    local thickness = t.thickness or 1
    local lineType = t.lineType or cv.LINE_8
    local shift = t.shift or 0

    if rec then
        C.rectangle(cv.wrap_tensors(img), rec, color, thickness, lineType, shift)
    else
        C.rectangle(cv.wrap_tensors(img), pt1, pt2, color, thickness, lineType, shift)
    end
end


function cv.circle(t)
    local img = assert(t.img)
    local center = cv.Point(assert(t.center))
    local radius = assert(t.radius)
    local color = cv.Scalar(assert(t.color))
    local thickness = t.thickness or 1
    local lineType = t.lineType or cv.LINE_8
    local shift = t.shift or 0

    C.circle(cv.wrap_tensors(img), center, radius, color, thickness, lineType, shift)
end


function cv.ellipse(t)
    local img = assert(t.img)
    local center = cv.Point(assert(t.center))
    local axes = cv.Size(assert(t.axes))
    local angle = assert(t.angle)
    local startAngle = assert(t.startAngle)
    local endAngle = assert(t.endAngle)
    local color = cv.Scalar(assert(t.color))
    local thickness = t.thickness or 1
    local lineType = t.lineType or cv.LINE_8
    local shift = t.shift or 0

    C.ellipse(cv.wrap_tensors(img), center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift)
end


function cv.ellipseFromRect(t)
    local img = assert(t.img)
    local box = cv.RotatedRect(assert(t.box))
    local color = cv.Scalar(assert(t.color))
    local thickness = t.thickness or 1
    local lineType = t.lineType or cv.LINE_8

    C.ellipseFromRect(cv.wrap_tensors(img), box, color, thickness, lineType)
end


function cv.fillConvexPoly(t)
    local img = assert(t.img)
    local points = assert(t.points)
    if type(points) == "table" then
        points = torch.FloatTensor(points)
    end
    local color = cv.Scalar(assert(t.color))
    local lineType = t.lineType or cv.LINE_8
    local shift = t.shift or 0

    C.fillConvexPoly(cv.wrap_tensors(img), cv.wrap_tensors(points), color, lineType, shift)
end


function cv.fillPoly(t)
    local img = assert(t.img)
    local pts = assert(t.pts)
    assert(type(pts) == 'table')
    local color = cv.Scalar(assert(t.color))
    local lineType = t.lineType or cv.LINE_8
    local shift = t.shift or 0
    local offset = cv.Point(t.offset or {0,0})

    C.fillPoly(cv.wrap_tensors(img), cv.wrap_tensors(pts), color, lineType, shift, offset)
end


function cv.polylines(t)
    local img = assert(t.img)
    local pts = assert(t.pts)
    assert(type(pts) == 'table')
    local isClosed = assert(t.isClosed)
    local color = cv.Scalar(assert(t.color))
    local thickness = t.thickness or 1
    local lineType = t.lineType or cv.LINE_8
    local shift = t.shift or 0

    C.polylines(cv.wrap_tensors(img), cv.wrap_tensors(pts), isClosed, color, thickness, lineType, shift)
end


function cv.drawContours(t)
    local image = assert(t.image)
    local contours = assert(t.contours)
    assert(type(contours) == 'table')
    local contourIdx = assert(t.contourIdx)
    local color = cv.Scalar(assert(t.color))
    local thickness = t.thickness or 1
    local lineType = t.lineType or cv.LINE_8
    local hierarchy = t.hierarchy
    local maxLevel = t.maxLevel or cv.INT_MAX
    local offset = t.offset or cv.Point

    C.drawContours(cv.wrap_tensors(image), cv.wrap_tensors(contours), contourIdx, color, thickness, lineType, cv.wrap_tensors(hierarchy), maxLevel, offset)
end


function cv.clipLine(t)
    local imgSize = t.imgSize
    local imgRect = t.imgRect
    assert(imgSize or imgRect)
    if imgSize then
        imgSize = cv.Size(imgSize)
    else
        imgRect = cv.Rect(imgRect)
    end

    local pt1 = cv.Point(assert(t.pt1))
    local pt2 = cv.Point(assert(t.pt2))

    local result
    if imgSize then
        result = C.clipLineSize(imgSize, pt1, pt2)
    else
        result = C.clipLineRect(imgRect, pt1, pt2)
    end

    return cv.Point{result.scalar.v0, result.scalar.v1}, cv.Point{result.scalar.v2, result.scalar.v3}, result.val
end


function cv.ellipse2Poly(t)
    local center = cv.Point(assert(t.center))
    local axes = cv.Size(assert(t.axes))
    local angle = assert(t.angle)
    local arcStart = assert(t.arcStart)
    local arcEnd = assert(t.arcEnd)
    local delta = assert(t.delta)

    return cv.unwrap_tensors(
        C.ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta, pts))
end


function cv.putText(t)
    local img = assert(t.img)
    local text = assert(t.text)
    local org = cv.Point(assert(t.org))
    local fontFace = assert(t.fontFace)
    local fontScale = assert(t.fontScale)
    local color = cv.Scalar(assert(t.color))
    local thickness = t.thickness or 1
    local lineType = t.lineType or cv.LINE_8
    local bottomLeftOrigin = t.bottomLeftOrigin or false

    C.putText(cv.wrap_tensors(img), text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)
end


function cv.getTextSize(t)
    local text = assert(t.text)
    local fontFace = assert(t.fontFace)
    local fontScale = assert(t.fontScale)
    local thickness = assert(t.thickness)

    local result = C.getTextSize(text, fontFace, fontScale, thickness)
    return result.size, result.val
end

--- ***************** Classes *****************
require 'cv.Classes'

-- GeneralizedHough

do
    local GeneralizedHough = torch.class('cv.GeneralizedHough', 'cv.Algorithm')

    function GeneralizedHough:setTemplate(t)
        if t.templ then            
            local templ = assert(t.templ)
            local templCenter = cv.Point(t.templCenter or {-1, -1})

            C.GeneralizedHough_setTemplate(self.ptr, cv.wrap_tensors(templ), templCenter)
        else
            local edges = assert(t.edges)
            local dx = assert(t.dx)
            local dy = assert(t.dy)
            local templCenter = cv.Point(t.templCenter or {-1, -1})

            C.GeneralizedHough_setTemplate_edges(
                self.ptr, cv.wrap_tensors(edges), cv.wrap_tensors(dx),
                cv.wrap_tensors(dy), templCenter)
        end
    end

    -- votes: boolean. To output votes or not
    function GeneralizedHough:detect(t)
        if t.image then
            local image = assert(t.image)
            local positions = t.positions
            local votes = t.votes or false

            return cv.unwrap_tensors(
                C.GeneralizedHough_detect(
                    self.ptr, cv.wrap_tensors(image), cv.wrap_tensors(positions), votes))
        else
            local image = assert(t.image)
            local positions = t.positions
            local votes = t.votes or false

            return cv.unwrap_tensors(
                C.GeneralizedHough_detect(
                    self.ptr, cv.wrap_tensors(image), cv.wrap_tensors(positions), votes))
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
    local GeneralizedHoughBallard = torch.class('cv.GeneralizedHoughBallard', 'cv.GeneralizedHough')

    function GeneralizedHoughBallard:__init()
        self.ptr = ffi.gc(C.GeneralizedHoughBallard_ctor(), C.Algorithm_dtor)
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
    local GeneralizedHoughGuil = torch.class('cv.GeneralizedHoughGuil', 'cv.GeneralizedHough')

    function GeneralizedHoughGuil:__init()
        self.ptr = ffi.gc(C.GeneralizedHoughGuil_ctor(), C.Algorithm_dtor)
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
    local CLAHE = torch.class('cv.CLAHE', 'cv.Algorithm')

    function CLAHE:__init()
        self.ptr = ffi.gc(C.CLAHE_ctor(), C.Algorithm_dtor)
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
    local LineSegmentDetector = torch.class('cv.LineSegmentDetector', 'cv.Algorithm')

    function LineSegmentDetector:__init(t)
        local refine = t.refine or cv.LSD_REFINE_STD
        local scale = t.scale or 0.8
        local sigma_scale = t.sigma_scale or 0.6
        local quant = t.quant or 2.0
        local ang_th = t.ang_th or 22.5
        local log_eps = t.log_eps or 0
        local density_th = t.density_th or 0.7
        local n_bins = t.n_bins or 1024

        self.ptr = ffi.gc(
            C.LineSegmentDetector_ctor(
                refine, scale, sigma_scale, quant, ang_th, log_eps, density_th, n_bins), 
            C.Algorithm_dtor
        )
    end

    function LineSegmentDetector:detect(t)
        local image = assert(t.image)
        local lines = t.lines
        local width = t.width or false
        local prec  = t.prec or false
        local nfa   = t.nfa or false

        return cv.unwrap_tensors(
            C.LineSegmentDetector_detect(
                self.ptr, cv.wrap_tensors(image), cv.wrap_tensors(lines), width, prec, nfa))
    end

    function LineSegmentDetector:drawSegments(t)
        local image = assert(t.image)
        local lines = assert(t.lines)

        C.LineSegmentDetector_drawSegments(
            self.ptr, cv.wrap_tensors(image), cv.wrap_tensors(lines))
    end

    function LineSegmentDetector:compareSegments(t)
        local lines1 = assert(t.lines1)
        local lines2 = assert(t.lines2)
        local image = t.image

        return C.LineSegmentDetector_compareSegments(
            self.ptr, cv.wrap_tensors(lines1), cv.wrap_tensors(lines2), cv.wrap_tensors(image))
    end
end

-- Subdiv2D

do
    local Subdiv2D = torch.class('cv.Subdiv2D')

    function Subdiv2D:__init(t)
        local rect = cv.Rect(t.rect)

        if rect then
            self.ptr = ffi.gc(C.Subdiv2D_ctor(cv.Rect(rect)), C.Subdiv2D_dtor)
        else
            self.ptr = ffi.gc(C.Subdiv2D_ctor_default(), C.Subdiv2D_dtor)
        end
    end

    function Subdiv2D:initDelaunay(t)
        local rect = cv.Rect(assert(t.rect))

        C.Subdiv2D_initDelaunay(self.ptr, rect)
    end

    function Subdiv2D:insert(t)
        local pt = t.pt
        local ptvec = t.ptvec
        
        if pt then
            return C.Subdiv2D_insert(cv.Point2f(pt))
        else
            if type(ptvec) == "table" then
                ptvec = torch.FloatTensor(ptvec)
            end
            C.Subdiv2D_insert_vector(self.ptr, cv.wrap_tensors(ptvec))
        end
    end

    function Subdiv2D:locate(t)
        local pt = cv.Point2f(assert(t.pt))
        
        result = C.Subdiv2D_locate(self.ptr, pt)
        return result.v0, result.v1, result.v2
    end

    function Subdiv2D:findNearest(t)
        local pt = cv.Point2f(assert(t.pt))
        
        result = C.Subdiv2D_findNearest(self.ptr, pt)
        return result.point, result.val
    end

    function Subdiv2D:getEdgeList(t)        
        return cv.unwrap_tensors(C.Subdiv2D_getEdgeList(self.ptr))
    end

    function Subdiv2D:getTriangleList(t)
        return cv.unwrap_tensors(C.Subdiv2D_getTriangleList(self.ptr), true)
    end

    function Subdiv2D:getVoronoiFacetList(t)
        facetList = cv.unwrap_tensors(C.Subdiv2D_getVoronoiFacetList(self.ptr, cv.unwrap_tensors(idx)), true)
        facetCenters = facetList[#facetList]
        facetList[#facetList] = nil
        return facetCenters, facetList
    end

    function Subdiv2D:getVertex(t)
        result = C.Subdiv2D_getVertex(self.ptr, vertex)
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
        result = C.Subdiv2D_edgeOrg(self.ptr, edge)
        return result.point, result.val
    end

    function Subdiv2D:edgeDst(t)
        result = C.Subdiv2D_edgeDst(self.ptr, edge)
        return result.point, result.val
    end
end