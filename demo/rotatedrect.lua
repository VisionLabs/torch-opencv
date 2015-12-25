-- A demo of how to use complex structs by the example of RotatedRect

local cv = require 'cv'
require 'cv.imgproc'

local rect = cv.RotatedRect{center={50, 120}, size={40, 90}, angle=30}
points = cv.boxPoints{rect}

print(points)
