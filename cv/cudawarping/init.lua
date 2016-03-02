local cv = require 'cv._env'
require 'cutorch'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or require 'cv._env_cuda'

local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper remapCuda(struct cutorchInfo info,
                           struct TensorWrapper src, struct TensorWrapper map1,
                           struct TensorWrapper map2, int interpolation, struct TensorWrapper dst,
                           int borderMode, struct ScalarWrapper borderValue);

struct TensorWrapper resizeCuda(struct cutorchInfo info,
                            struct TensorWrapper src, struct TensorWrapper dst,
                            struct SizeWrapper dsize, double fx, double fy,
                            int interpolation);

struct TensorWrapper warpAffineCuda(struct cutorchInfo info,
                                struct TensorWrapper src, struct TensorWrapper dst,
                                struct TensorWrapper M, struct SizeWrapper dsize,
                                int flags, int borderMode, struct ScalarWrapper borderValue);

struct TensorArray buildWarpAffineMapsCuda(
        struct cutorchInfo info, struct TensorWrapper M, bool inverse,
        struct SizeWrapper dsize, struct TensorWrapper xmap, struct TensorWrapper ymap);

struct TensorWrapper warpPerspectiveCuda(struct cutorchInfo info,
                                     struct TensorWrapper src, struct TensorWrapper dst,
                                     struct TensorWrapper M, struct SizeWrapper dsize,
                                     int flags, int borderMode, struct ScalarWrapper borderValue);

struct TensorArray buildWarpPerspectiveMapsCuda(
        struct cutorchInfo info, struct TensorWrapper M, bool inverse,
        struct SizeWrapper dsize, struct TensorWrapper xmap, struct TensorWrapper ymap);

struct TensorWrapper rotateCuda(
        struct cutorchInfo info, struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dsize, double angle, double xShift, double yShift, int interpolation);

struct TensorWrapper pyrDownCuda(struct cutorchInfo info,
                             struct TensorWrapper src, struct TensorWrapper dst);

struct TensorWrapper pyrUpCuda(struct cutorchInfo info,
                           struct TensorWrapper src, struct TensorWrapper dst);
]]

local C = ffi.load(cv.libPath('cudawarping'))

function cv.cuda.remap(t)
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
        C.remapCuda(cv.cuda._info(),
            cv.wrap_tensor(src), cv.wrap_tensor(map1), cv.wrap_tensor(map2),
            interpolation, cv.wrap_tensor(dst), borderMode, borderValue))
end

function cv.cuda.resize(t)
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
        C.resizeCuda(cv.cuda._info(),
            cv.wrap_tensor(src), cv.wrap_tensor(dst), dsize, fx, fy, interpolation))
end

function cv.cuda.warpAffine(t)
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
        C.warpAffineCuda(cv.cuda._info(),
            cv.wrap_tensor(src), cv.wrap_tensor(dst), cv.wrap_tensor(M), dsize,
            flags, borderMode, borderValue))
end

function cv.cuda.buildWarpAffineMaps(t)
    local argRules = {
        {"M", required = true},
        {"inverse", required = true},
        {"dsize", required = true, operator = cv.Size},
        {"xmap", default = nil},
        {"ymap", default = nil}
    }
    local M, inverse, dsize, xmap, ymap = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.buildWarpAffineMapsCuda(cv.cuda._info(),
        cv.wrap_tensor(M), inverse, dsize, cv.wrap_tensor(xmap), cv.wrap_tensor(ymap)))
end

function cv.cuda.warpPerspective(t)
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
        C.warpPerspectiveCuda(cv.cuda._info(),
            cv.wrap_tensor(src), cv.wrap_tensor(dst), cv.wrap_tensor(M), dsize,
            flags, borderMode, borderValue))
end

function cv.cuda.buildWarpPerspectiveMaps(t)
    local argRules = {
        {"M", required = true},
        {"inverse", required = true},
        {"dsize", required = true, operator = cv.Size},
        {"xmap", default = nil},
        {"ymap", default = nil}
    }
    local M, inverse, dsize, xmap, ymap = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.buildWarpPerspectiveMapsCuda(cv.cuda._info(),
        cv.wrap_tensor(M), inverse, dsize, cv.wrap_tensor(xmap), cv.wrap_tensor(ymap)))
end

function cv.cuda.rotate(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"dsize", required = true},
        {"angle", required = true},
        {"xShift", default = 0},
        {"yShift", default = 0},
        {"interpolation", default = cv.INTER_LINEAR}
    }
    local src, dst, dsize, angle, xShift, yShift, interpolation = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.rotateCuda(cv.cuda._info(),
            cv.wrap_tensor(src), cv.wrap_tensor(dst), dsize, angle, xShift, yShift, interpolation))
end

function cv.cuda.pyrDown(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"dstSize", default = {0,0}, operator = cv.Size},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, dst, dstSize, borderType = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.pyrDownCuda(cv.cuda._info(),
            cv.wrap_tensor(src), cv.wrap_tensor(dst),
            dstSize, borderType))
end

function cv.cuda.pyrUp(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"dstSize", default = {0,0}, operator = cv.Size},
        {"borderType", default = cv.BORDER_DEFAULT}
    }
    local src, dst, dstSize, borderType = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.pyrUpCuda(cv.cuda._info(),
            cv.wrap_tensor(src), cv.wrap_tensor(dst),
            dstSize, borderType))
end

return cv.cuda
