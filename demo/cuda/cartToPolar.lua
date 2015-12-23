local cv = require 'cv'
require 'cv.cudaarithm'
require 'cutorch'

local x = torch.CudaTensor(5, 5) * 0 + 0.7
local y = torch.CudaTensor(5, 5) * 0 - 0.5

local magnitude, angle = cv.cuda.cartToPolar{x, y}

print(magnitude)
print(angle)