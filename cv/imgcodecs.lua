require 'cv'

local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper imread(const char *filename, int flags);

struct TensorArrayPlusBool imreadmulti(const char *filename, int flags);

bool imwrite(const char *filename, struct TensorWrapper img, struct TensorWrapper params);

struct TensorWrapper imdecode(struct TensorWrapper buf, int flags);
]]

local C = ffi.load(libPath('imgcodecs'))

function cv.imread(t)
	local filename = assert(t.filename)
	local flags = t.flags or cv.IMREAD_COLOR
    return cv.unwrap_tensors(C.imread(filename, flags))
end

function cv.imreadmulti(t)
	local filename = assert(t.filename)
	local flags = t.flags or cv.IMREAD_ANYCOLOR

	return cv.unwrap_tensors(C.imreadmulti(filename, flags), true)
end

function cv.imwrite(t)
	local filename = assert(t.filename)
	local img = assert(t.img)
	local params = torch.IntTensor(t.params or {})

	return C.imwrite(filename, cv.wrap_tensors(img), cv.wrap_tensors(params))
end

function cv.imdecode(t)
	local buf = assert(t.buf)
	local flags = assert(t.flags)

	return cv.unwrap_tensors(C.imdecode(cv.wrap_tensors(buf), flags))
end
