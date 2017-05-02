local cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.highgui'

if not arg[1] then
    print('Usage: `th demo/imshow.lua path-to-image`')
    print('Now using demo/data/lena.jpg')
end

local path = arg[1] or 'demo/data/lena.jpg'
local im = cv.imread {path, cv.IMREAD_GRAYSCALE}

if im:nElement() == 0 then
    error('Couldn\'t load ' .. path)
end

cv.imshow {"Hello, Lua!", im}
cv.waitKey {0}
