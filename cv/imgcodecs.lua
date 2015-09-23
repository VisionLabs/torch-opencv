require 'cv'

local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper imread(const char * filename, int flags);
]]

local C = ffi.load 'lib/libimgcodecs.so'

cv.IMREAD_UNCHANGED  = -1
cv.IMREAD_GRAYSCALE  = 0
cv.IMREAD_COLOR      = 1
cv.IMREAD_ANYDEPTH   = 2
cv.IMREAD_ANYCOLOR   = 4
cv.IMREAD_LOAD_GDAL  = 8 

function cv.imread(filename, flags)
    return cv.unwrap_tensor(C.imread(filename, flags or cv.IMREAD_COLOR))
end
