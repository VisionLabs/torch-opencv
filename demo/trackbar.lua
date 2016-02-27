-- A translated demo from here:
-- http://docs.opencv.org/2.4/doc/tutorials/highgui/trackbar/trackbar.htm
local cv = require 'cv'
require 'cv.highgui'
require 'cv.imgcodecs' -- for imread
require 'cv.imgproc'   -- for addWeighted

local ffi = require 'ffi'
local alphaSlider = ffi.new('int[1]', 40)
local alphaSliderMax = 100

local imgPrefixPath = 'demo/data/trackbar/'

local src1 = cv.imread{imgPrefixPath..  'LinuxLogo.png'}
local src2 = cv.imread{imgPrefixPath..'WindowsLogo.png'}

if not src1 or not src2 then
    print('Error loading images')
    os.exit(-1)
end

assert(src1:isSameSizeAs(src2))
local dst = src1.new(src1:size())

local function onTrackbar()
    local alpha = alphaSlider[0] / alphaSliderMax
    local beta = 1 - alpha
    -- can also be done in 2 lines with `torch.add()`
    cv.addWeighted {src1, alpha, src2, beta, 0.0, dst}
    cv.imshow {"Linear Blend", dst}
end

cv.namedWindow{'Linear Blend', cv.WINDOW_AUTOSIZE}

local trackbarName = 'Alpha x '..alphaSliderMax
cv.createTrackbar{trackbarName, 'Linear Blend', alphaSlider, alphaSliderMax, onTrackbar}

onTrackbar()
cv.waitKey{0}
