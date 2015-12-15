-- a temporary testbed for testing
-- CudaTensor <-> GpuMat conversion

-- to be removed soon

local cv = require 'cv'
cv.cuda = require 'cv.cudaarithm'
require 'cutorch'

local n = 5

local a = (torch.rand(n, n) * 10):int():float():cuda()
local b = (torch.rand(n, n) * 10):int():float():cuda()
local c = torch.FloatTensor(n, n)

for i = 1, n do
	for j = 1, n do
		c[i][j] = math.min(a[i][j], b[i][j])
	end
end

print(c)

local c_cv = cv.cuda.min{a, b}

print(c_cv)

os.exit(0)

for i = 1,30 do
	local dims = torch.rand(2) * 20 + 1
	local g = torch.CudaTensor(dims[1], dims[2]) * 0 + 0.3
	print(g)
	cv.cuda.min{g, g}
end