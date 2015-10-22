require 'cv'

local ffi = require 'ffi'

ffi.cdef[[
void imshow(const char *winname, struct TensorWrapper mat);

int waitKey(int delay);

void namedWindow(const char *winname, int flags);
]]

local C = ffi.load(libPath('highgui'))

function cv.imshow(t)
	local winname = t.winname or "Window 1"
	local image = assert(t.image)
    C.imshow(winname, cv.wrap_tensors(image))
end

function cv.waitKey(delay)
    C.waitKey(delay)
end

function cv.namedWindow(t)
	local winname = t.winname or "Window 1"
	local flags = t.flags or cv.WINDOW_AUTOSIZE
end