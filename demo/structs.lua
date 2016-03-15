-- A demo of how to use complex structs
local cv = require 'cv'
require 'cv.imgproc'

-- RotatedRect
local rect = cv.RotatedRect{center={50, 120}, size={40, 90}, angle=30}
local points = cv.boxPoints{rect}
print(points)

-- Hu moments
local points = torch.FloatTensor {
    {10, 23.4},
    {17, 10.1},
    {20, 31.2},
    {65, 43.1},
    {5.67, 11},
    {10, 4.42}
}

local moments = cv.moments{points}

print(cv.HuMoments{moments})
