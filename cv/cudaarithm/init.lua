local cv = require 'cv._env'
require 'cv.Classes'

local flann = {}

local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper min(
        struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper dst);
]]

local C = ffi.load(cv.libPath('cudaarithm'))

function flann.min(t)
    local argRules = {
        {"src1", required = true},
        {"src2", required = true},
        {"dst", default = nil}
    }
    local src1, src2, dst = cv.argcheck(t, argRules)

    -- TODO this
    C.min(cv.wrap_tensor(src1), cv.wrap_tensor(src2), cv.wrap_tensor(dst))
end

return flann
