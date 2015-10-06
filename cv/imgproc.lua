require 'cv'

local ffi = require 'ffi'

ffi.cdef[[

struct TensorWrapper getGaussianKernel(
        int ksize, double sigma, int ktype);

struct MultipleTensorWrapper getDerivKernels(
        int dx, int dy, int ksize, bool normalize, int ktype);

struct TensorWrapper getGaborKernel(
        int ksize_rows, int ksize_cols, double sigma, double theta,
        double lambd, double gamma, double psi, int ktype);

struct TensorWrapper getStructuringElement(
        int shape, int ksize_rows, int ksize_cols, 
        int anchor_x, int anchor_y);

struct TensorWrapper medianBlur(
        struct TensorWrapper src, struct TensorWrapper dst, int ksize);

struct TensorWrapper GaussianBlur(
        struct TensorWrapper src, struct TensorWrapper dst, 
        int ksize_x, int ksize_y, double sigmaX, double sigmaY, int borderType);

struct TensorWrapper bilateralFilter(
        struct TensorWrapper src, struct TensorWrapper dst, int d, 
        double sigmaColor, double sigmaSpace, int borderType);

struct TensorWrapper boxFilter(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth, 
        int ksize_x, int ksize_y, int anchor_x, int anchor_y, bool normalize, int borderType);

struct TensorWrapper sqrBoxFilter(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int ksize_x, int ksize_y, int anchor_x, int anchor_y,
        bool normalize, int borderType);

struct TensorWrapper blur(
        struct TensorWrapper src, struct TensorWrapper dst,
        int ksize_x, int ksize_y, int anchor_x, int anchor_y, int borderType);

struct TensorWrapper filter2D(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct TensorWrapper kernel, int anchor_x, int anchor_y,
        double delta, int borderType);

struct TensorWrapper sepFilter2D(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct TensorWrapper kernelX,struct TensorWrapper kernelY,
        int anchor_x, int anchor_y, double delta, int borderType);

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
        int winSize_x, int winSize_y, int zeroZone_x, int zeroZone_y,
        struct TermCriteriaWrapper criteria);

struct TensorWrapper goodFeaturesToTrack(
        struct TensorWrapper image,
        int maxCorners, double qualityLevel, double minDistance,
        struct TensorWrapper mask, int blockSize, bool useHarrisDetector, double k);
        
struct TensorWrapper erode(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper kernel, int anchor_x, int anchor_y,
        int iterations, int borderType, struct ScalarWrapper borderValue);

struct TensorWrapper dilate(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper kernel, int anchor_x, int anchor_y,
        int iterations, int borderType, struct ScalarWrapper borderValue);

struct TensorWrapper morphologyEx(
        struct TensorWrapper src, struct TensorWrapper dst,
        int op, struct TensorWrapper kernel,
        int anchor_x, int anchor_y, int iterations,
        int borderType, struct ScalarWrapper borderValue);

struct TensorWrapper resize(
        struct TensorWrapper src, struct TensorWrapper dst,
        int dsize_x, int dsize_y, double fx, double fy,
        int interpolation);
        
struct TensorWrapper resize(
        struct TensorWrapper src, struct TensorWrapper dst,
        int dsize_x, int dsize_y, double fx, double fy,
        int interpolation);

struct TensorWrapper warpAffine(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper M, int dsize_x, int dsize_y, 
        int flags, int borderMode, struct ScalarWrapper borderValue);

struct TensorWrapper warpPerspective(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper M, int dsize_x, int dsize_y,
        int flags, int borderMode, struct ScalarWrapper borderValue);

struct TensorWrapper remap(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper map1, struct TensorWrapper map2,
        int interpolation, int borderMode, struct ScalarWrapper borderValue);

struct MultipleTensorWrapper convertMaps(
        struct TensorWrapper map1, struct TensorWrapper map2,
        struct TensorWrapper dstmap1, struct TensorWrapper dstmap2,
        int dstmap1type, bool nninterpolation);

struct TensorWrapper getRotationMatrix2D(
        double center_x, double center_y, double angle, double scale);
        
struct TensorWrapper getPerspectiveTransform(
        struct TensorWrapper src, struct TensorWrapper dst);

struct TensorWrapper getAffineTransform(
        struct TensorWrapper src, struct TensorWrapper dst);

struct TensorWrapper getRectSubPix(
        struct TensorWrapper image, int patchSize_x, int patchsize_y,
        double center_x, double center_y, struct TensorWrapper patch,
        int patchType);
        
struct TensorWrapper logPolar(
        struct TensorWrapper src, struct TensorWrapper dst,
        float center_x, float center_y, double M, int flags);

struct TensorWrapper linearPolar(
        struct TensorWrapper src, struct TensorWrapper dst,
        float center_x, float center_y, double maxRadius, int flags);

struct TensorWrapper integral(
        struct TensorWrapper src, struct TensorWrapper sum, int sdepth);

struct MultipleTensorWrapper integralN(
        struct TensorWrapper src, struct TensorWrapper sum,
        struct TensorWrapper sqsum, struct TensorWrapper tilted,
        int sdepth, int sqdepth);

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
        struct TensorWrapper dst, int winSize_x, int winSize_y, int type);

struct TWPlusDouble threshold(
        struct TensorWrapper src, struct TensorWrapper dst,
        double thresh, double maxval, int type);

struct TensorWrapper adaptiveThreshold(
        struct TensorWrapper src, struct TensorWrapper dst,
        double maxValue, int adaptiveMethod, int thresholdType,
        int blockSize, double C);

struct TensorWrapper pyrDown(
        struct TensorWrapper src, struct TensorWrapper dst,
        int dstSize_x, int dstSize_y, int borderType);

struct TensorWrapper pyrUp(
        struct TensorWrapper src, struct TensorWrapper dst,
        int dstSize_x, int dstSize_y, int borderType);

struct MultipleTensorWrapper buildPyramid(
        struct TensorWrapper src, struct MultipleTensorWrapper dst,
        int maxlevel, int borderType);

struct TensorWrapper undistort(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper newCameraMatrix);

struct MultipleTensorWrapper initUndistortRectifyMap(
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper R, struct TensorWrapper newCameraMatrix,
        int size_x, int size_y, int m1type,
        struct MultipleTensorWrapper maps);

struct MTWPlusFloat initWideAngleProjMap(
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        int imageSize_x, int imageSize_y, int destImageWidth,
        int m1type, struct MultipleTensorWrapper maps,
        int projType, double alpha);

struct TensorWrapper getDefaultNewCameraMatrix(
        struct TensorWrapper cameraMatrix, int imgsize_x, int imgsize_y, bool centerPrincipalPoint);

struct TensorWrapper undistortPoints(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper R, struct TensorWrapper P);

struct TensorWrapper calcHist(
        struct MultipleTensorWrapper images,
        struct IntArray channels, struct TensorWrapper mask,
        struct TensorWrapper hist, int dims, struct IntArray histSize,
        struct FloatArrayOfArrays ranges, bool uniform, bool accumulate);

struct TensorWrapper calcBackProject(
        struct MultipleTensorWrapper images, int nimages,
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
]]


local C = ffi.load 'lib/libimgproc.so'


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
    local ksize = t.ksize
    assert(#ksize == 2)
    local sigma = assert(t.sigma)
    local theta = assert(t.theta)
    local lambd = assert(t.lambd)
    local gamma = assert(t.gamma)
    local psi   = t.psi or math.pi * 0.5
    local ktype = t.ktype or cv.CV_64F

    return cv.unwrap_tensors(
        C.getGaborKernel(ksize[1], ksize[2], sigma, theta, lambd, gamma, psi, ktype))
end


function cv.getStructuringElement(t)
    local shape =  assert(t.shape)
    local ksize =  t.ksize
    assert(#ksize == 2)
    local anchor = t.anchor or {-1, -1}
    assert(#anchor == 2)

    return cv.unwrap_tensors(C.getStructuringElement(shape, ksize[1], ksize[2], anchor[1], anchor[2]))
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
    local ksize =      t.ksize
    assert(#ksize == 2)
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
            cv.wrap_tensors(src), cv.wrap_tensors(dst), ksize[1], ksize[2], sigmaX, sigmaY, borderType))
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
    local ksize =      t.ksize
    assert(#ksize == 2)
    local anchor = t.anchor or {-1, -1}
    assert(#anchor == 2)
    local normalize =  t.normalize
    if normalize == nil then normalize = true end
    local borderType = t.borderType or cv.BORDER_DEFAULT

    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.boxFilter(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), ddepth, ksize[1], 
            ksize[2], anchor[1], anchor[2], normalize, borderType))
end

function cv.sqrBoxFilter(t)

    local src = assert(t.src)
    local dst = t.dst
    local ddepth = assert(t.ddepth)
    local ksize = t.ksize
    assert(#ksize == 2)
    local anchor = t.anchor or {-1,-1}
    local normalize =  t.normalize
    if normalize == nil then normalize = true end
    local borderType = t.borderType or cv.BORDER_DEFAULT

    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.sqrBoxFilter(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), ddepth, ksize[1], ksize[2], anchor[1], anchor[2], normalize, borderType))
end


function cv.blur(t)
    local src = assert(t.src)
    local dst = t.dst
    local ksize = t.ksize
    assert(#ksize == 2)
    local anchor = t.anchor or {-1,-1}
    local borderType = t.borderType or cv.BORDER_DEFAULT

    assert(cv.tensorType(src) ~= cv.CV_8S and
           cv.tensorType(src) ~= cv.CV_32S)
    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.blur(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), ksize[1], ksize[2], anchor[1], anchor[2], borderType))
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
            cv.wrap_tensors(src), cv.wrap_tensors(dst), ddepth, cv.wrap_tensors(kernel), anchor[1], anchor[2], delta, borderType))
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
            cv.wrap_tensors(src), cv.wrap_tensors(dst), ddepth, kernelX, kernelY, anchor[1], anchor[2], delta, borderType))
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
    local winSize = t.winSize
    assert(#winSize == 2)
    local zeroZone = t.zeroZone
    assert(#zeroZone == 2)
    local criteria = cv.TermCriteria(assert(t.criteria))

    assert(image:nDimension() == 2)
    assert(corners:size()[2] == 2 and cv.tensorType(corners) == cv.CV_32F)

    C.cornerSubPix(
        cv.wrap_tensors(image), cv.wrap_tensors(corners), winSize[1], winSize[2],
        zeroZone[1], zeroZone[2], criteria)
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
    local anchor = t.anchor or {-1, -1}
    assert(#anchor == 2)
    local borderType = t.borderType or cv.BORDER_CONSTANT
    local borderValue = cv.Scalar(t.borderValue or {0/0}) -- pass nan to detect default value

    return cv.unwrap_tensors(
        C.erode(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), cv.wrap_tensors(kernel), anchor[1], anchor[2],
            iterations, borderType, borderValue))
end


function cv.dilate(t)
    local src =        assert(t.src)
    local dst =        t.dst
    local kernel = assert(t.kernel)
    local iterations = 1
    local anchor = t.anchor or {-1, -1}
    assert(#anchor == 2)
    local borderType = t.borderType or cv.BORDER_CONSTANT
    local borderValue = cv.Scalar(t.borderValue or {0/0}) -- pass nan to detect default value

    return cv.unwrap_tensors(
        C.dilate(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), cv.wrap_tensors(kernel), anchor[1], anchor[2],
            iterations, borderType, borderValue))
end


function cv.morphologyEx(t)
    local src = assert(t.src)
    local dst = t.dst
    local op = assert(t.op)
    local kernel = assert(t.kernel)
    local iterations = 1
    local anchor = t.anchor or {-1, -1}
    assert(#anchor == 2)
    local borderType = t.borderType or cv.BORDER_CONSTANT
    local borderValue = cv.Scalar(t.borderValue or {0/0}) -- pass nan to detect default value

    return cv.unwrap_tensors(
        C.morphologyEx(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), op, cv.wrap_tensors(kernel), anchor[1], anchor[2],
            iterations, borderType, borderValue))
end


function cv.resize(t)
    local src = assert(t.src)
    local dst = t.dst
    local dsize = t.dsize or {0, 0}
    assert(#dsize == 2)
    local fx = t.fx or 0
    local fy = t.fy or 0
    local interpolation = t.interpolation or cv.INTER_LINEAR

    return cv.unwrap_tensors(
        C.resize(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), dsize[1], dsize[2], fx, fy, interpolation))
end


function cv.warpAffine(t)
    local src = assert(t.src)
    local dst = t.dst
    local M = assert(t.M)
    local dsize = t.dsize or {0, 0}
    assert(#dsize == 2)
    local flags = t.flags or cv.INTER_LINEAR
    local borderMode = t.borderMode or cv.BORDER_CONSTANT
    local borderValue = cv.Scalar(t.borderValue or {0/0}) -- pass nan to detect default value

    return cv.unwrap_tensors(
        C.warpAffine(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), cv.wrap_tensors(M), dsize[1], dsize[2],
            flags, borderMode, borderValue))
end


function cv.warpPerspective(t)
    local src = assert(t.src)
    local dst = t.dst
    local M = assert(t.M)
    local dsize = t.dsize or {0, 0}
    assert(#dsize == 2)
    local flags = t.flags or cv.INTER_LINEAR
    local borderMode = t.borderMode or cv.BORDER_CONSTANT
    local borderValue = cv.Scalar(t.borderValue or {0/0}) -- pass nan to detect default value

    return cv.unwrap_tensors(
        C.warpPerspective(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), cv.wrap_tensors(M), dsize[1], dsize[2],
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
    local center = assert(t.center)
    assert(#center == 2)
    local angle = assert(t.angle)
    local scale = assert(t.scale)

    return cv.unwrap_tensors(
        C.getRotationMatrix2D(center[1], center[2], angle, scale))
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
    local patchSize = assert(t.patchSize)
    assert(#patchSize == 2)
    local center = assert(t.center)
    assert(#center == 2)
    local patch = t.patch
    local patchType = t.patchType or -1

    return cv.unwrap_tensors(
        C.getRectSubPix(cv.wrap_tensors(image), patchSize[1], patchSize[2], center[1], center[2],
                        cv.wrap_tensors(patch), patchType))
end


function cv.getRectSubPix(t)
    local image = assert(t.src)
    local patchSize = assert(t.patchSize)
    assert(#patchSize == 2)
    local center = assert(t.center)
    assert(#center == 2)
    local patch = t.patch
    local patchType = t.patchType or -1

    return cv.unwrap_tensors(
        C.getRectSubPix(cv.wrap_tensors(image), patchSize[1], patchSize[2], center[1], center[2],
            cv.wrap_tensors(patch), patchType))
end


function cv.logPolar(t)
    local src = assert(t.src)
    local dst = t.dst
    local center = assert(t.center)
    assert(#center == 2)
    local M = assert(t.M)
    local flags = assert(t.flags)

    return cv.unwrap_tensors(
        C.logPolar(cv.wrap_tensors(src), cv.wrap_tensors(dst), center[1], center[2], M, flags))
end


function cv.linearPolar(t)
    local src = assert(t.src)
    local dst = t.dst
    local center = assert(t.center)
    assert(#center == 2)
    local maxRadius = assert(t.maxRadius)
    local flags = assert(t.flags)

    return cv.unwrap_tensors(
        C.linearPolar(cv.wrap_tensors(src), cv.wrap_tensors(dst), center[1], center[2], maxRadius, flags))
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
    local winSize = assert(t.winSize)
    assert(#winSize == 2)
    local type = assert(t.type)

    if dst then
        assert(cv.tensorType(dst) == type)
        assert(dst:size()[1] == winSize[1] and
               dst:size()[2] == winSize[2])
    end

    return cv.unwrap_tensors(
        C.createHanningWindow(cv.wrap_tensors(dst), winSize[1], winSize[2], type))
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

    result = C.threshold(cv.wrap_tensors(src), cv.wrap_tensors(dst),
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
    local dstSize = t.dstSize or {0,0 }
    assert(#dstSize == 2)
    local borderType = t.borderType or cv.BORDER_DEFAULT

    if dst then
        assert(dst:type() == src:type())
        assert(dst:size()[1] == dstSize[1] and
               dst:size()[2] == dstSize[2])
    end

    return cv.unwrap_tensors(
        C.pyrDown(
            cv.wrap_tensors(src), cv.wrap_tensors(dst),
            dstSize[1], dstSize[2], borderType))
end


function cv.pyrUp(t)
    local src = assert(t.src)
    local dst = t.dst
    local dstSize = t.dstSize or {0,0 }
    assert(#dstSize == 2)
    local borderType = t.borderType or cv.BORDER_DEFAULT

    if dst then
        assert(dst:type() == src:type())
        assert(dst:size()[1] == dstSize[1] and
                dst:size()[2] == dstSize[2])
    end

    return cv.unwrap_tensors(
        C.pyrUp(
            cv.wrap_tensors(src), cv.wrap_tensors(dst),
            dstSize[1], dstSize[2], borderType))
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
    local size = assert(t.size)
    assert(#size == 2)
    local m1type = assert(t.m1type)
    local maps = t.maps

    return cv.unwrap_tensors(
        C.initUndistortRectifyMap(
            cv.wrap_tensors(cameraMatrix), cv.wrap_tensors(distCoeffs), 
            cv.wrap_tensors(R), cv.wrap_tensors(newCameraMatrix),
            size[1], size[2], m1type, cv.wrap_tensors(maps)))
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
    local imageSize = assert(t.imageSize)
    assert(#imageSize == 2)
    local destImageWidth = assert(t.destImageWidth)
    local m1type = assert(t.m1type)
    local maps = t.maps
    local projType = assert(t.projType)
    local alpha = assert(t.alpha)
    
    result = C.initWideAngleProjMap(
        cv.wrap_tensors(cameraMatrix), cv.wrap_tensors(distCoeffs),
        imageSize[1], imageSize[2], destImageWidth, m1type, cv.wrap_tensors(maps),
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
    local seedPoint = assert(t.seedPoint)
    assert(#seedPoint == 2)
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