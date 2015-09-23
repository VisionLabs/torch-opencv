require 'cv'

local ffi = require 'ffi'

ffi.cdef[[

]]

local C = ffi.load 'lib/libml.so'
