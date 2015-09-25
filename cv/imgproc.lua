require 'cv'

local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper getGaussianKernel(int ksize, double sigma, int ktype);
struct MultipleTensorWrapper getDerivKernels(int dx, int dy, int ksize, bool normalize, int ktype);
]]

local C = ffi.load 'lib/libimgproc.so'

function cv.getGaussianKernel(ksize, sigma, ktype)
	return cv.unwrap_tensor(C.getGaussianKernel(ksize, sigma, ktype or cv.CV_64F))
end

function cv.getDerivKernels(dx, dy, ksize, normalize, ktype)
	ktype = ktype or cv.CV_32F
	normalize = normalize or false

	wrapper = C.getDerivKernels(dx, dy, ksize, normalize, ktype);
	return cv.unwrap_tensors(wrapper)
end