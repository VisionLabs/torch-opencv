require 'cutorch'
local cv = require 'cv'
require 'cv.cudawarping'
require 'cv.highgui'
require 'cv.imgcodecs'

if not arg[1] then
    print('Usage: `th demo/cuda/resize.lua path-to-image`')
    print('Now using demo/lena.jpg')
end

local img = cv.imread {arg[1] or 'demo/lena.jpg', cv.IMREAD_COLOR}
local imgCUDA = img:float():cuda() / 255
local resized = cv.cuda.resize{imgCUDA, {1024, 768}}

cv.imshow{"Resized to 1024x768", resized:float()}
cv.waitKey{0}
