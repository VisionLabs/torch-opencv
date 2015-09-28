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
        struct TensorWrapper image, struct TensorWrapper lines,
        double rho, double theta, int threshold, double srn, double stn,
        double min_theta, double max_theta);

struct TensorWrapper HoughLinesP(
        struct TensorWrapper image, struct TensorWrapper lines, double rho,
        double theta, int threshold, double minLineLength, double maxLineGap);


struct TensorWrapper HoughCircles(
        struct TensorWrapper image, struct TensorWrapper circles,
        int method, double dp, double minDist, double param1, double param2,
        int minRadius, int maxRadius);

struct TensorWrapper cornerSubPix(
        struct TensorWrapper image, struct TensorWrapper corners,
        int winSize_x, int winSize_y, int zeroZone_x, int zeroZone_y,
        int crit_type, int crit_max_iter, double crit_eps);

struct TensorWrapper goodFeaturesToTrack(
        struct TensorWrapper image, struct TensorWrapper corners,
        int maxCorners, double qualityLevel, double minDistance,
        struct TensorWrapper mask, int blockSize, bool useHarrisDetector, double k);
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

    if dst and src ~= dst then
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

    if dst and src ~= dst then
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

    if dst and src ~= dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
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

    if dst and src ~= dst then
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

    if dst and src ~= dst then
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

    if dst and src ~= dst then
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

    if dst and src ~= dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
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

    if dst and src ~= dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
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

    if dst and src ~= dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
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

    if dst and src ~= dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
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

    if dst and src ~= dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
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

    if edges and image ~= edges then
        assert(edges:type() == image:type() and image:isSameSizeAs(edges))
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

    if dst and src ~= dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
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

    if dst and src ~= dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
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

    if dst and src ~= dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
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

    if dst and src ~= dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.preCornerDetect(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), ksize, borderType))
end


function cv.HoughLines(t)
    local image = assert(t.image)
    local lines = t.lines
    local rho = assert(t.rho)
    local theta = assert(t.theta)
    local threshold = assert(t.threshold)
    local srn = t.srn or 0
    local stn = t.stn or 0
    local min_theta = t.min_theta or 0
    local max_theta = t.max_theta or cv.CV_PI

    if lines and image ~= lines then
        assert(lines:type() == image:type() and image:isSameSizeAs(lines))
    end

    return cv.unwrap_tensors(
        C.HoughLines(
            cv.wrap_tensors(image), cv.wrap_tensors(lines), rho, theta, threshold, srn, stn, min_theta, max_theta))
end


function cv.HoughLinesP(t)
    local image = assert(t.image)
    local lines = t.lines
    local rho = assert(t.rho)
    local theta = assert(t.theta)
    local threshold = assert(t.threshold)
    local minLineLength = t.minLineLength or 0
    local maxLineGap = t.maxLineGap or 0

    if lines and image ~= lines then
        assert(lines:type() == image:type() and image:isSameSizeAs(lines))
    end

    return cv.unwrap_tensors(
        C.HoughLinesP(
            cv.wrap_tensors(image), cv.wrap_tensors(lines), rho, theta, threshold, minLineLength, maxLineGap))
end


function cv.HoughCircles(t)
    local image = assert(t.image)
    local circles = t.circles
    local method = assert(t.method)
    local dp = assert(t.dp)
    local minDist = assert(t.minDist)
    local param1 = t.param1 or 100
    local param2 = t.param2 or 100
    local minRadius = t.minRadius or 0
    local maxRadius = t.maxRadius or 0

    if circles and image ~= circles then
        assert(circles:type() == image:type() and image:isSameSizeAs(circles))
    end

    return cv.unwrap_tensors(
        C.HoughCircles(
            cv.wrap_tensors(image), cv.wrap_tensors(circles), method, dp, minDist, param1, param2, minRadius, maxRadius))
end


function cv.cornerSubPix(t)
    local image = assert(t.image)
    local corners = t.corners
    local winSize = t.winSize
    assert(#winSize == 2)
    local zeroZone = t.zeroZone
    assert(#zeroZone == 2)
    local TermCriteria = assert(t.TermCriteria)
    local criteria = assert(t.criteria)

    if corners and image ~= corners then
        assert(corners:type() == image:type() and image:isSameSizeAs(corners))
    end

    return cv.unwrap_tensors(
        C.cornerSubPix(
            cv.wrap_tensors(image), cv.wrap_tensors(corners), winSize[1], winSize[2], zeroZone[1], zeroZone[2], TermCriteria, criteria))
end


function cv.goodFeaturesToTrack(t)
    local image = assert(t.image)
    local corners = t.corners
    local maxCorners = assert(t.maxCorners)
    local qualityLevel = assert(t.qualityLevel)
    local minDistance = assert(t.minDistance)
    local mask = t.mask or cv.EMPTY_WRAPPER
    local blockSize = t.blockSize or 3
    local useHarrisDetector = t.useHarrisDetector or false
    local k = t.k or 0.04

    if corners and image ~= corners then
        assert(corners:type() == image:type() and image:isSameSizeAs(corners))
    end

    return cv.unwrap_tensors(
        C.goodFeaturesToTrack(
            cv.wrap_tensors(image), cv.wrap_tensors(corners), maxCorners, qualityLevel, minDistance, cv.wrap_tensors(mask), blockSize, useHarrisDetector, k))
end
