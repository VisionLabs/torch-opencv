require 'cutorch'
local cv = require 'cv'
require 'cv.cudawarping'
require 'cv.highgui'
require 'cv.imgcodecs'

if not arg[1] then
    print('Usage: `th demo/cuda/resize.lua path-to-image`')
    print('Now using demo/lena.jpg')
end

-- TODO #58
local img = cv.imread {arg[1] or 'demo/lena.jpg', cv.IMREAD_GRAYSCALE}
local imgCUDA = img:float():cuda()
local resized = cv.cuda.resize{imgCUDA, {1024, 768}}

cv.imshow{"Resized to 1024x768", resized:byte()}
cv.waitKey{0}
