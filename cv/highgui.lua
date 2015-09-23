require 'cv'

local ffi = require 'ffi'

ffi.cdef[[
void imshow(const char *winname, struct TensorWrapper mat);
int waitKey(int delay);
]]

local C = ffi.load 'lib/libhighgui.so'

function cv.imshow(winname, image)
    C.imshow(winname, cv.wrap_tensor(image))
end

function cv.waitKey(delay)
    C.waitKey(delay)
end