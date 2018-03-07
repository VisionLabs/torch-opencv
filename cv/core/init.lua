local cv = require 'cv._env'
require 'cv.Classes'

local ffi = require 'ffi'

ffi.cdef[[

int getNumThreads();

void setNumThreads(int nthreads);
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

return cv