-- a temporary testbed for testing
-- CudaTensor <-> GpuMat conversion

-- to be removed soon

local cv = require 'cv'
cv.cuda = require 'cv.cudaarithm'
require 'cutorch'

for i = 1,30 do
	local dims = torch.rand(2) * 20 + 1
	local g = torch.CudaTensor(dims[1], dims[2]) * 0 + 0.3
	print(g)
	cv.cuda.min{g, g}
end