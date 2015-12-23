local cv = require 'cv._env'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or {}

local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper min(
        void *state, struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper dst);
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

return cv
