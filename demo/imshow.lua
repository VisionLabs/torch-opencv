local cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.highgui'

if not arg[1] then
    print('Usage: `th demo/imshow.lua path-to-image`')
    print('Now using demo/lena.jpg')
end

local im = cv.imread {arg[1] or 'demo/lena.jpg', cv.IMREAD_GRAYSCALE}
cv.imshow {"Hello, Lua!", im}
cv.waitKey {0}
