local cv = require 'cv._env'
require 'cutorch'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or {}

local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper remap(struct THCState *state,
                           struct TensorWrapper src, struct TensorWrapper map1,
                           struct TensorWrapper map2, int interpolation, struct TensorWrapper dst,
                           int borderMode, struct ScalarWrapper borderValue);

struct TensorWrapper resize(struct THCState *state,
                            struct TensorWrapper src, struct TensorWrapper dst,
                            struct SizeWrapper dsize, double fx, double fy,
                            int interpolation);

struct TensorWrapper warpAffine(struct THCState *state,
                                struct TensorWrapper src, struct TensorWrapper dst,
                                struct TensorWrapper M, struct SizeWrapper dsize,
                                int flags, int borderMode, struct ScalarWrapper borderValue);

struct TensorArray buildWarpAffineMaps(
        struct THCState *state, struct TensorWrapper M, bool inverse,
        struct SizeWrapper dsize, struct TensorWrapper xmap, struct TensorWrapper ymap);

struct TensorWrapper warpPerspective(struct THCState *state,
                                     struct TensorWrapper src, struct TensorWrapper dst,
                                     struct TensorWrapper M, struct SizeWrapper dsize,
                                     int flags, int borderMode, struct ScalarWrapper borderValue);

struct TensorArray buildWarpPerspectiveMaps(
        struct THCState *state, struct TensorWrapper M, bool inverse,
        struct SizeWrapper dsize, struct TensorWrapper xmap, struct TensorWrapper ymap);

struct TensorWrapper rotate(
        struct THCState *state, struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dsize, double angle, double xShift, double yShift, int interpolation);

struct TensorWrapper pyrDown(struct THCState *state,
                             struct TensorWrapper src, struct TensorWrapper dst);

struct TensorWrapper pyrUp(struct THCState *state,
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
        C.remap(cutorch._state,
            cv.wrap_tensor(src), cv.wrap_tensor(map1), cv.wrap_tensor(map2),
            interpolation, cv.wrap_tensor(dst), borderMode, borderValue))
end

function cv.cuda.resize(t)
    local argRules = {
        {"src", required = true},
        {"dsize", required = true, operator = cv.Size},
        {"dst", default = nil},
        {"fx", default = 0},
        {"fy", default = 0},
        {"interpolation", default = cv.INTER_LINEAR}
    }
    local src, dsize, dst, fx, fy, interpolation = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.resize(cutorch._state,
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
        C.warpAffine(cutorch._state,
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

    return cv.unwrap_tensors(C.buildWarpAffineMaps(cutorch._state,
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
        C.warpPerspective(cutorch._state,
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

    return cv.unwrap_tensors(C.buildWarpPerspectiveMaps(cutorch._state,
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
        C.rotate(cutorch._state,
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
        C.pyrDown(cutorch._state,
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
        C.pyrUp(cutorch._state,
            cv.wrap_tensor(src), cv.wrap_tensor(dst),
            dstSize, borderType))
end

return cv.cuda
