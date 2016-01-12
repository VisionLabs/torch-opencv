local cv = require 'cv._env'
require 'cutorch'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or {}

local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper min(
        void *state, struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper dst);

struct TensorWrapper max(
        void *state, struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper dst);

struct TensorPlusDouble threshold(
        void *state, struct TensorWrapper src, 
        struct TensorWrapper dst, double thresh, double maxval, int type);

struct TensorWrapper magnitude(
        void *state, struct TensorWrapper xy, struct TensorWrapper magnitude);

struct TensorWrapper magnitudeSqr(
        void *state, struct TensorWrapper xy, struct TensorWrapper magnitude);

struct TensorWrapper magnitude2(
        void *state, struct TensorWrapper x, struct TensorWrapper y, struct TensorWrapper magnitude);

struct TensorWrapper magnitudeSqr2(
        void *state, struct TensorWrapper x, struct TensorWrapper y, struct TensorWrapper magnitudeSqr);

struct TensorWrapper phase(
        void *state, struct TensorWrapper x, struct TensorWrapper y,
        struct TensorWrapper angle, bool angleInDegrees);

struct TensorArray cartToPolar(
        void *state, struct TensorWrapper x, struct TensorWrapper y,
        struct TensorWrapper magnitude, struct TensorWrapper angle, bool angleInDegrees);

struct TensorArray polarToCart(
        void *state, struct TensorWrapper magnitude, struct TensorWrapper angle,
        struct TensorWrapper x, struct TensorWrapper y, bool angleInDegrees);

struct PtrWrapper LookUpTable_ctor(
        void *state, struct TensorWrapper lut);

struct TensorWrapper LookUpTable_transform(
        void *state, struct PtrWrapper ptr,
        struct TensorWrapper src, struct TensorWrapper dst);

struct TensorWrapper rectStdDev(
        void *state, struct TensorWrapper src, struct TensorWrapper sqr,
        struct TensorWrapper dst, struct RectWrapper rect);

struct TensorWrapper normalize(
        void *state, struct TensorWrapper src, struct TensorWrapper dst,
        double alpha, double beta, int norm_type, int dtype);

struct TensorWrapper integral(
        void *state, struct TensorWrapper src, struct TensorWrapper sum);

struct TensorWrapper sqrIntegral(
        void *state, struct TensorWrapper src, struct TensorWrapper sum);

struct TensorWrapper mulSpectrums(
        void *state, struct TensorWrapper src1, struct TensorWrapper src2, 
        struct TensorWrapper dst, int flags, bool conjB);

struct TensorWrapper mulAndScaleSpectrums(
        void *state, struct TensorWrapper src1, struct TensorWrapper src2,
        struct TensorWrapper dst, int flags, float scale, bool conjB);

struct TensorWrapper dft(
        void *state, struct TensorWrapper src, 
        struct TensorWrapper dst, struct SizeWrapper dft_size, int flags);

struct PtrWrapper Convolution_ctor(
        void *state, struct SizeWrapper user_block_size);

struct TensorWrapper Convolution_convolve(
        void *state, struct PtrWrapper ptr, struct TensorWrapper image,
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
        C.min(cutorch._state, cv.wrap_tensor(src1), cv.wrap_tensor(src2), cv.wrap_tensor(dst)))
end

function cv.cuda.max(t)
    local argRules = {
        {"src1", required = true},
        {"src2", required = true},
        {"dst", default = nil}
    }
    local src1, src2, dst = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.max(cutorch._state, cv.wrap_tensor(src1), cv.wrap_tensor(src2), cv.wrap_tensor(dst)))
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

    local retval = C.threshold(cutorch._state, 
            cv.wrap_tensor(src), cv.wrap_tensor(dst), thresh, maxval, _type)
    return retval.val, cv.unwrap_tensors(retval.tensor)
end

function cv.cuda.magnitude(t)
    local argRules = {
        {"xy", required = true},
        {"magnitude", default = nil}
    }
    local xy, magnitude = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.magnitude(cutorch._state,
        cv.wrap_tensor(xy), cv.wrap_tensor(magnitude)))
end

function cv.cuda.magnitudeSqr(t)
    local argRules = {
        {"xy", required = true},
        {"magnitude", default = nil}
    }
    local xy, magnitude = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.magnitudeSqr(cutorch._state,
        cv.wrap_tensor(xy), cv.wrap_tensor(magnitude)))
end

function cv.cuda.magnitude2(t)
    local argRules = {
        {"x", required = true},
        {"y", required = true},
        {"magnitude", default = nil}
    }
    local x, y, magnitude = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.magnitude2(cutorch._state,
        cv.wrap_tensor(x), cv.wrap_tensor(y), cv.wrap_tensor(magnitude)))
end

function cv.cuda.magnitudeSqr2(t)
    local argRules = {
        {"x", required = true},
        {"y", required = true},
        {"magnitude", default = nil}
    }
    local x, y, magnitude = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.magnitudeSqr2(cutorch._state,
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

    return cv.unwrap_tensors(C.phase(cutorch._state,
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

    return cv.unwrap_tensors(C.cartToPolar(cutorch._state, 
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

    return cv.unwrap_tensors(C.cartToPolar(cutorch._state, 
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
            C.LookUpTable_ctor(cutorch._state, cv.wrap_tensor(lut)),
            Classes.Algorithm_dtor)
    end

    function LookUpTable:transform(t)
        local argRules = {
            {"src", required = true},
            {"dst", default = nil}
        }
        local src, dst = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.LookUpTable_transform(cutorch._state, self.ptr,
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

    return cv.unwrap_tensors(C.rectStdDev(cutorch._state,
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
    }
    local src, dst, alpha, beta, norm_type, dtype = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.normalize(cutorch._state,
        cv.wrap_tensor(src), cv.wrap_tensor(dst), alpha, beta, norm_type, dtype))
end

function cv.cuda.integral(t)
    local argRules = {
        {"src", required = true},
        {"sum", default = nil}
    }
    local src, sum = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.integral(cutorch._state,
        cv.wrap_tensor(src), cv.wrap_tensor(sum)))
end

function cv.cuda.sqrIntegral(t)
    local argRules = {
        {"src", required = true},
        {"sqsum", default = nil}
    }
    local src, sqsum = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.sqrIntegral(cutorch._state,
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
        C.mulSpectrums(cutorch._state, 
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
        C.mulAndScaleSpectrums(cutorch._state, 
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

    return cv.unwrap_tensors(C.dft(cutorch._state,
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
            C.Convolution_ctor(cutorch._state, user_block_size),
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

        return cv.unwrap_tensors(C.Convolution_convolve(cutorch._state, self.ptr, 
            cv.wrap_tensor(image), cv.wrap_tensor(templ), cv.wrap_tensor(result), ccorr))
    end
end

return cv.cuda
