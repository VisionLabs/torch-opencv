require 'cv'

local ffi = require 'ffi'

ffi.cdef[[

struct TensorWrapper inpaint(struct TensorWrapper src, struct TensorWrapper inpaintMask,
                            struct TensorWrapper dst, double inpaintRadius, int flags)

]]

local C = ffi.load(libPath('photo'))

function cv.inpaint(t)
    local src =           assert(t.src)
    local inpaintMask =   assert(t.inpaintMask)
    local dst =           assert(t.dst)
    local inpaintRadius = assert(t.inpaintRadius)
    local flags =         assert(t.flags)
    
    return cv.unwrap_tensors(
        C.inpaint(
            cv.wrap_tensors(src), cv.wrap_tensors(inpaintMask), cv.wrap_tensors(dst), inpaintRadius, flags))
end