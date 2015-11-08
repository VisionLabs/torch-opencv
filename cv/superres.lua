require 'cv'

local ffi = require 'ffi'

ffi.cdef[[

]]

local C = ffi.load(cv.libPath('superres'))
