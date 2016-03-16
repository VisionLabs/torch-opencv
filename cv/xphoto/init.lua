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

function cv.xphoto.autowbGrayworld(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"thresh", default = 0.5} }
    local src, dst, thresh = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
        C.xphoto_autowbGrayworld(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), thresh))
end

function cv.xphoto.balanceWhite(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"algorithmType", required = true},
        {"inputMin", default = 0},
        {"inputMax", default = 255},
        {"outputMin", default = 0},
        {"outputMax", default = 255} }
    local src, dst, algorithmType, inputMin, inputMax, outputMin, outputMax = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
        C.xphoto_balanceWhite(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), algorithmType, inputMin, inputMax, outputMin, outputMax))
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