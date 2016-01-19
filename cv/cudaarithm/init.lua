local cv = require 'cv._env'
require 'cutorch'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or require 'cv._env_cuda'

local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper min(
        struct cutorchInfo state, struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper dst);

struct TensorWrapper max(
        struct cutorchInfo state, struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper dst);

struct TensorPlusDouble threshold(
        struct cutorchInfo state, struct TensorWrapper src,
        struct TensorWrapper dst, double thresh, double maxval, int type);

struct TensorWrapper magnitude(
        struct cutorchInfo state, struct TensorWrapper xy, struct TensorWrapper magnitude);

struct TensorWrapper magnitudeSqr(
        struct cutorchInfo state, struct TensorWrapper xy, struct TensorWrapper magnitude);

struct TensorWrapper magnitude2(
        struct cutorchInfo state, struct TensorWrapper x, struct TensorWrapper y, struct TensorWrapper magnitude);

struct TensorWrapper magnitudeSqr2(
        struct cutorchInfo state, struct TensorWrapper x, struct TensorWrapper y, struct TensorWrapper magnitudeSqr);

struct TensorWrapper phase(
        struct cutorchInfo state, struct TensorWrapper x, struct TensorWrapper y,
        struct TensorWrapper angle, bool angleInDegrees);

struct TensorArray cartToPolar(
        struct cutorchInfo state, struct TensorWrapper x, struct TensorWrapper y,
        struct TensorWrapper magnitude, struct TensorWrapper angle, bool angleInDegrees);

struct TensorArray polarToCart(
        struct cutorchInfo state, struct TensorWrapper magnitude, struct TensorWrapper angle,
        struct TensorWrapper x, struct TensorWrapper y, bool angleInDegrees);

struct PtrWrapper LookUpTable_ctor(
        struct cutorchInfo state, struct TensorWrapper lut);

struct TensorWrapper LookUpTable_transform(
        struct cutorchInfo state, struct PtrWrapper ptr,
        struct TensorWrapper src, struct TensorWrapper dst);

struct TensorWrapper rectStdDev(
        struct cutorchInfo state, struct TensorWrapper src, struct TensorWrapper sqr,
        struct TensorWrapper dst, struct RectWrapper rect);

struct TensorWrapper normalize(
        struct cutorchInfo state, struct TensorWrapper src, struct TensorWrapper dst,
        double alpha, double beta, int norm_type, int dtype);

struct TensorWrapper integral(
        struct cutorchInfo state, struct TensorWrapper src, struct TensorWrapper sum);

struct TensorWrapper sqrIntegral(
        struct cutorchInfo state, struct TensorWrapper src, struct TensorWrapper sum);

struct TensorWrapper mulSpectrums(
        struct cutorchInfo state, struct TensorWrapper src1, struct TensorWrapper src2,
        struct TensorWrapper dst, int flags, bool conjB);

struct TensorWrapper mulAndScaleSpectrums(
        struct cutorchInfo state, struct TensorWrapper src1, struct TensorWrapper src2,
        struct TensorWrapper dst, int flags, float scale, bool conjB);

struct TensorWrapper dft(
        struct cutorchInfo state, struct TensorWrapper src,
        struct TensorWrapper dst, struct SizeWrapper dft_size, int flags);

struct PtrWrapper Convolution_ctor(
        struct cutorchInfo state, struct SizeWrapper user_block_size);

struct TensorWrapper Convolution_convolve(
        struct cutorchInfo state, struct PtrWrapper ptr, struct TensorWrapper image,
        struct TensorWrapper templ, struct TensorWrapper result, bool ccor);
]]

local C = ffi.load(cv.libPath('cudaarithm'))

require 'cv.Classes'
local Classes = ffi.load(cv.libPath('Classes'))

function cv.cuda.min(t)
    local argRules = {
        {"src1", required = true},
        {"src2", required = true},
        {"dst", default = nil}
    }
    local src1, src2, dst = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.min(cv.cuda._info(), cv.wrap_tensor(src1), cv.wrap_tensor(src2), cv.wrap_tensor(dst)))
end

function cv.cuda.max(t)
    local argRules = {
        {"src1", required = true},
        {"src2", required = true},
        {"dst", default = nil}
    }
    local src1, src2, dst = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.max(cv.cuda._info(), cv.wrap_tensor(src1), cv.wrap_tensor(src2), cv.wrap_tensor(dst)))
end

function cv.cuda.threshold(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"thresh", required = true},
        {"maxval", required = true},
        {"type", required = true}
    }
    local src, dst, thresh, maxval, _type = cv.argcheck(t, argRules)

    local retval = C.threshold(cv.cuda._info(),
            cv.wrap_tensor(src), cv.wrap_tensor(dst), thresh, maxval, _type)
    return retval.val, cv.unwrap_tensors(retval.tensor)
end

function cv.cuda.magnitude(t)
    local argRules = {
        {"xy", required = true},
        {"magnitude", default = nil}
    }
    local xy, magnitude = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.magnitude(cv.cuda._info(),
        cv.wrap_tensor(xy), cv.wrap_tensor(magnitude)))
end

function cv.cuda.magnitudeSqr(t)
    local argRules = {
        {"xy", required = true},
        {"magnitude", default = nil}
    }
    local xy, magnitude = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.magnitudeSqr(cv.cuda._info(),
        cv.wrap_tensor(xy), cv.wrap_tensor(magnitude)))
end

function cv.cuda.magnitude2(t)
    local argRules = {
        {"x", required = true},
        {"y", required = true},
        {"magnitude", default = nil}
    }
    local x, y, magnitude = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.magnitude2(cv.cuda._info(),
        cv.wrap_tensor(x), cv.wrap_tensor(y), cv.wrap_tensor(magnitude)))
end

function cv.cuda.magnitudeSqr2(t)
    local argRules = {
        {"x", required = true},
        {"y", required = true},
        {"magnitude", default = nil}
    }
    local x, y, magnitude = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.magnitudeSqr2(cv.cuda._info(),
        cv.wrap_tensor(x), cv.wrap_tensor(y), cv.wrap_tensor(magnitudeSqr)))
end

function cv.cuda.phase(t)
    local argRules = {
        {"x", required = true},
        {"y", required = true},
        {"angle", default = nil},
        {"angleInDegrees", default = false},
    }
    local x, y, angle, angleInDegrees = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.phase(cv.cuda._info(),
        cv.wrap_tensor(x), cv.wrap_tensor(y), cv.wrap_tensor(angle), angleInDegrees))
end

function cv.cuda.cartToPolar(t)
    local argRules = {
        {"x", required = true},
        {"y", required = true},
        {"magnitude", default = nil},
        {"angle", default = nil},
        {"angleInDegrees", default = false},
    }
    local x, y, magnitude, angle, angleInDegrees = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.cartToPolar(cv.cuda._info(),
        cv.wrap_tensor(x), cv.wrap_tensor(y), cv.wrap_tensor(magnitude),
        cv.wrap_tensor(angle), angleInDegrees))
end

function cv.cuda.polarToCart(t)
    local argRules = {
        {"magnitude", required = true},
        {"angle", required = true},
        {"x", default = nil},
        {"y", default = nil},
        {"angleInDegrees", default = false},
    }
    local magnitude, angle, x, y, angleInDegrees = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.cartToPolar(cv.cuda._info(),
        cv.wrap_tensor(magnitude), cv.wrap_tensor(angle),
        cv.wrap_tensor(x), cv.wrap_tensor(y), angleInDegrees))
end

do
    local LookUpTable = torch.class('cuda.LookUpTable', cv.cuda)

    function LookUpTable:__init(t)
        local argRules = {
            {"lut", required = true}
        }
        local lut = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(
            C.LookUpTable_ctor(cv.cuda._info(), cv.wrap_tensor(lut)),
            Classes.Algorithm_dtor)
    end

    function LookUpTable:transform(t)
        local argRules = {
            {"src", required = true},
            {"dst", default = nil}
        }
        local src, dst = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.LookUpTable_transform(cv.cuda._info(), self.ptr,
            cv.wrap_tensor(src), cv.wrap_tensor(dst)))
    end
end

function cv.cuda.rectStdDev(t)
    local argRules = {
        {"src", required = true},
        {"sqr", required = true},
        {"dst", default = nil},
        {"rect", required = true, operator = cv.Rect}
    }
    local src, sqr, dst, rect = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.rectStdDev(cv.cuda._info(),
        cv.wrap_tensor(src), cv.wrap_tensor(sqr), cv.wrap_tensor(dst), rect))
end

function cv.cuda.normalize(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"alpha", required = true},
        {"beta", required = true},
        {"norm_type", required = true},
        {"dtype", required = true},
        {"mask", default = nil}
    }
    local src, dst, alpha, beta, norm_type, dtype, mask = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.normalize(cv.cuda._info(),
        cv.wrap_tensor(src), cv.wrap_tensor(dst), 
        alpha, beta, norm_type, dtype, cv.wrap_tensor(mask)))
end

function cv.cuda.integral(t)
    local argRules = {
        {"src", required = true},
        {"sum", default = nil}
    }
    local src, sum = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.integral(cv.cuda._info(),
        cv.wrap_tensor(src), cv.wrap_tensor(sum)))
end

function cv.cuda.sqrIntegral(t)
    local argRules = {
        {"src", required = true},
        {"sqsum", default = nil}
    }
    local src, sqsum = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.sqrIntegral(cv.cuda._info(),
        cv.wrap_tensor(src), cv.wrap_tensor(sqsum)))
end

function cv.cuda.mulSpectrums(t)
    local argRules = {
        {"src1", required = true},
        {"src2", required = true},
        {"dst", default = nil},
        {"flags", required = true},
        {"conjB", default = false}
    }
    local src1, src2, dst, flags, conjB = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.mulSpectrums(cv.cuda._info(),
            cv.wrap_tensor(src1), cv.wrap_tensor(src2), cv.wrap_tensor(dst), flags, conjB))
end

function cv.cuda.mulAndScaleSpectrums(t)
    local argRules = {
        {"src1", required = true},
        {"src2", required = true},
        {"dst", default = nil},
        {"flags", required = true},
        {"scale", required = true},
        {"conjB", default = false}
    }
    local src1, src2, dst, flags, scale, conjB = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.mulAndScaleSpectrums(cv.cuda._info(),
            cv.wrap_tensor(src1), cv.wrap_tensor(src2), cv.wrap_tensor(dst), flags, scale, conjB))
end

function cv.cuda.dft(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"dft_size", required = true, operator = cv.Size},
        {"flags", default = 0}
    }
    local src, dst, dft_size, flags = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.dft(cv.cuda._info(),
        cv.wrap_tensor(src), cv.wrap_tensor(dst), dft_size, flags))
end

do
    local Convolution = torch.class('cuda.Convolution', cv.cuda)

    function Convolution:__init(t)
        local argRules = {
            {"user_block_size", default = 0, operator = cv.Size}
        }
        local user_block_size = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(
            C.Convolution_ctor(cv.cuda._info(), user_block_size),
            Classes.Algorithm_dtor)
    end

    function Convolution:convolve(t)
        local argRules = {
            {"image", required = true},
            {"templ", required = true},
            {"result", default = nil},
            {"ccorr", default = false}
        }
        local image, templ, result, ccorr = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.Convolution_convolve(cv.cuda._info(), self.ptr,
            cv.wrap_tensor(image), cv.wrap_tensor(templ), cv.wrap_tensor(result), ccorr))
    end
end

return cv.cuda
