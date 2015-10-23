require 'cv'

local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper imread(const char *filename, int flags);
]]

local C = ffi.load(libPath('imgcodecs'))

cv.IMREAD_UNCHANGED  = -1
cv.IMREAD_GRAYSCALE  = 0
cv.IMREAD_COLOR      = 1
cv.IMREAD_ANYDEPTH   = 2
cv.IMREAD_ANYCOLOR   = 4
cv.IMREAD_LOAD_GDAL  = 8 

function cv.imread(t)
	local filename = assert(t.filename)
	local flags = t.flags or cv.IMREAD_COLOR
    return cv.unwrap_tensors(C.imread(filename, flags))
end
