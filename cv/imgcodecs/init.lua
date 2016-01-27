local cv = require 'cv._env'

local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper imread(const char *filename, int flags);

struct TensorArrayPlusBool imreadmulti(const char *filename, int flags);

bool imwrite(const char *filename, struct TensorWrapper img, struct TensorWrapper params);

struct TensorWrapper imdecode(struct TensorWrapper buf, int flags);

struct TensorWrapper imencode(
        const char *ext, struct TensorWrapper img, struct TensorWrapper params);
]]

local C = ffi.load(cv.libPath('imgcodecs'))

function cv.imread(t)
    local argRules = {
        {"filename", required = true},
        {"flags", default = cv.IMREAD_COLOR}
    }
    local filename, flags = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(C.imread(filename, flags))
end

function cv.imreadmulti(t)
    local argRules = {
        {"filename", required = true},
        {"flags", default = cv.IMREAD_ANYCOLOR}
    }
    local filename, flags = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.imreadmulti(filename, flags), true)
end

function cv.imwrite(t)
    local argRules = {
        {"filename", required = true},
        {"img", required = true},
        {"params", default = nil,
            operator = function(...) if ... then return torch.IntTensor(...) end end}
    }
    local filename, img, params = cv.argcheck(t, argRules)

    return C.imwrite(filename, cv.wrap_tensor(img), cv.wrap_tensor(params))
end

function cv.imdecode(t)
    local argRules = {
        {"buf", required = true},
        {"flags", required = true}
    }
    local buf, flags = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.imdecode(cv.wrap_tensor(buf), flags))
end

function cv.imencode(t)
    local argRules = {
        {"ext", required = true},
        {"img", required = true},
        {"params", default = nil}
    }
    local ext, img, params = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.imencode(ext, cv.wrap_tensor(img), cv.wrap_tensor(params)))
end

return cv
