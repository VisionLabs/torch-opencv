local cv = require 'cv._env'
local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper xphoto_autowbGrayworld(
        struct TensorWrapper src, struct TensorWrapper dst, float thresh);

struct TensorWrapper xphoto_balanceWhite(
        struct TensorWrapper src, struct TensorWrapper dst, int algorithmType,
        float inputMin, float inputMax, float outputMin, float outputMax);

struct TensorWrapper xphoto_dctDenoising(
        struct TensorWrapper src, struct TensorWrapper dst, double sigma, int psize);

struct TensorWrapper xphoto_inpaint(
        struct TensorWrapper src, struct TensorWrapper mask,
        struct TensorWrapper dst, int algorithmType);
]]

local C = ffi.load(cv.libPath('xphoto'))

cv.xphoto = {}

cv.WHITE_BALANCE_SIMPLE = 0
cv.WHITE_BALANCE_GRAYWORLD = 1
cv.INPAINT_SHIFTMAP = 0

function cv.xphoto.bm3dDenoising(t)
    local argRules = {
    {"src", required = true},
    {"dst", default = nil},
    {"h", default = 1},
    {"templateWindowSize", default = 4},
    {"searchWindowSize", default = 16},
    {"blockMatchingStep1", default = 2500},
    {"blockMatchingStep2", default = 400}, 
    {"groupSize", default = 8},
    {"slidingStep", default = 1},
    {"beta", default = 2.0},
    {"normType", default = 4},
    {"step", default = 0},
    {"transformType", default = 0} }
    local src, dst, h, templateWindowSize, searchWindowSize,
    blockMatchingStep1, blockMatchingStep2, groupSize,
    slidingStep, beta, normType, step, transformType = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
        C.xphoto_bm3dDenoising(cv.wrap_tensor(src), cv.wrap_tensor(dst), h,
            templateWindowSize, searchWindowSize, blockMatchingStep1, blockMatchingStep2, 
            groupSize, slidingStep, beta, normType, step, transformType))
end

function cv.xphoto.SimpleWB(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil} }
    local src, dst, thresh = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
        C.xphoto_SimpleWB(cv.wrap_tensor(src), cv.wrap_tensor(dst)))
end

function cv.xphoto.GrayworldWB(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil} }
    local src, dst, thresh = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
        C.xphoto_GrayworldWB(cv.wrap_tensor(src), cv.wrap_tensor(dst)))
end

function cv.xphoto.LearningBasedWB(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil} }
    local src, dst, thresh = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
        C.xphoto_LearningBasedWB(cv.wrap_tensor(src), cv.wrap_tensor(dst)))
end

function cv.xphoto.dctDenoising(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"sigma", required = true},
        {"psize", default = 16} }
    local src, dst, sigma, psize = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
                C.xphoto_dctDenoising(cv.wrap_tensor(src), cv.wrap_tensor(dst), sigma, psize)
    )
end

function cv.xphoto.inpaint(t)
    local argRules = {
        {"src", required = true},
        {"mask", required = true},
        {"dst", default = nil},
        {"algorithmType", required = true} }
    local src, mask, dst, algorithmType = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
                    C.xphoto_inpaint(
                        cv.wrap_tensor(src), cv.wrap_tensor(mask),
                        cv.wrap_tensor(dst), algorithmType))
end