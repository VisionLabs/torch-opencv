local cv = require 'cv'
require 'cutorch'
require 'cv.cudaarithm'

local x = torch.CudaTensor(5, 5) * 0 + 0.7
local y = torch.CudaTensor(5, 5) * 0 - 0.5

local magnitude, angle = cv.cuda.cartToPolar{x, y}

print(magnitude)
print(angle)