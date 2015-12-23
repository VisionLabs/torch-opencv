local cv = require 'cv._env'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or {}

local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper min(
        void *state, struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper dst);

struct TensorWrapper max(
        void *state, struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper dst);

struct TensorWrapper threshold(
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

struct PtrWrapper Convolution_ctor(struct SizeWrapper user_block_size);

struct TensorWrapper Convolution_convolve(
        void *state, struct PtrWrapper ptr, struct TensorWrapper image,
        struct TensorWrapper templ, struct TensorWrapper result, bool ccor);
]]

local C = ffi.load(cv.libPath('cudaarithm'))

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

return cv
