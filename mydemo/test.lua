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

local x = torch.ByteTensor(100,1):fill(0)
local y = torch.ByteTensor(100,1):fill(1)

--[[
local temp = cv.CvFeatureParams{
    type = 0,
    featureType = 0}

local v1, v2, v3 = temp:printAttrs	{
    posx = x,
    negx = y,
    x = 3,
    log = false }
--[[
]]
print(temp)
print(v1)
print("successfull")