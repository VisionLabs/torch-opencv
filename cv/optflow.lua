require 'cv'

local ffi = require 'ffi'

ffi.cdef[[

]]

local C = ffi.load 'lib/liboptflow.so'
