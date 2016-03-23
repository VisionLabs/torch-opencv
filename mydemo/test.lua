local cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.tracking'
require 'cv.stitching'


local image = cv.imread{'demo/data/lena.jpg'}

if not image then
    print("Problem loading image\n")
    os.exit(0)
end

--cv.test()

local temp = cv.MultiTracker{"TLD"}
local rect_array = {}

local rect = cv.Rect2d(2,2,20,20)
rect_array[1] = rect
rect_array[2] = rect

local array = cv.newArray("cv.Rect2d", rect_array)

local v1, v2, v3 = temp:update{
    trackerType = "TLD",
    image = image,
    boundingBox = array}
--[[
]]
print(temp)
print(v1)
print("successfull")