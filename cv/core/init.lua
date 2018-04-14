local cv = require 'cv._env'
require 'cv.Classes'

local ffi = require 'ffi'

ffi.cdef[[

int getNumThreads();

void setNumThreads(int nthreads);

struct TensorWrapper copyMakeBorder(struct TensorWrapper src, struct TensorWrapper dst, int top, 
                                    int bottom, int left, int right, int borderType,
                                    struct ScalarWrapper value);
]]

local C = ffi.load(cv.libPath('core'))

function cv.getNumThreads(t)
    local argRules = {
    }
    return C.getNumThreads(cv.argcheck(t, argRules))
end

function cv.setNumThreads(t)
    local argRules = {
        {"nthreads", required = true}
    }
    return C.setNumThreads(cv.argcheck(t, argRules))
end

function cv.copyMakeBorder(t)
	local argRules = {
		{"src", required = true, operator = cv.wrap_tensor},
		{"dst", default = nil, operator = cv.wrap_tensor},
		{"top", required = true},
		{"bottom", required = true},
		{"left", required = true},
		{"right", required = true},
		{"borderType", required = true},
		{"value", default = {0,0,0,0} , operator = cv.Scalar}
	}
	return cv.unwrap_tensors(C.copyMakeBorder(cv.argcheck(t, argRules)))
end

return cv