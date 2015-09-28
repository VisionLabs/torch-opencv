require 'cv'

local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper getGaussianKernel(int ksize, double sigma, int ktype);

struct MultipleTensorWrapper getDerivKernels(int dx, int dy, int ksize, bool normalize, int ktype);

struct TensorWrapper getGaborKernel(int ksize_rows, int ksize_cols, double sigma, double theta,
                                    double lambd, double gamma, double psi, int ktype);

struct TensorWrapper getStructuringElement(int shape, int ksize_rows, int ksize_cols, 
                                           int anchor_x, int anchor_y);

struct TensorWrapper medianBlur(struct TensorWrapper src, struct TensorWrapper dst, int ksize);

struct TensorWrapper GaussianBlur(struct TensorWrapper src, struct TensorWrapper dst,
                                  int ksize_x, int ksize_y, double sigmaX,
                                  double sigmaY, int borderType);
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
    local ksize =      t.ksize;
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
