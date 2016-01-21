local cv = require 'cv._env'
require 'cutorch'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or require 'cv._env_cuda'

local ffi = require 'ffi'

ffi.cdef[[

]]

local C = ffi.load(cv.libPath('cudaimgproc'))

require 'cv.Classes'
local Classes = ffi.load(cv.libPath('Classes'))



return cv.cuda
