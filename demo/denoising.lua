local cv = require 'cv'
require 'cv.photo'
require 'cv.imgcodecs'
require 'cv.highgui'

if not arg[1] then
    print('Usage: `th demo/denoising.lua path-to-image`')
    print('Now using demo/data/lena.jpg')
end
    
local image = cv.imread{arg[1] or 'demo/data/lena.jpg'}

cv.imshow{"Denoised image", cv.fastNlMeansDenoising{image}}
cv.waitKey{0}
