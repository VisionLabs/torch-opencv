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
]]


local C = ffi.load 'lib/libimgproc.so'


function cv.getGaussianKernel(ksize, sigma, ktype)
    local ksize = assert(t.ksize)
    local sigma = assert(t.sigma)
    local ktype = t.ktype or cv.CV_64F
    return cv.unwrap_tensor(C.getGaussianKernel(ksize, sigma, ktype))
end


function cv.getDerivKernels(dx, dy, ksize, normalize, ktype)
    local dx        = assert(t.dx)
    local dy        = assert(t.dy)
    local ksize     = assert(t.ksize)
    local ktype     = ktype or cv.CV_32F
    local normalize = normalize or false

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
    local normalize =  t.normalize or true
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
    local normalize = t.normalize or true
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
