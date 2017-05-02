local cv = require 'cv'
require 'cv.highgui'
require 'cv.imgproc'
require 'cv.imgcodecs'

if not arg[1] then
    print('Usage: `th demo/LineSegmentDetector.lua path-to-image`')
    print('Now using demo/data/lena.jpg')
end

local path = arg[1] or 'demo/data/lena.jpg'
local image = cv.imread{path, cv.IMREAD_GRAYSCALE}

if image:nDimension() == 0 then
    error('Couldn\'t load ' .. path)
end

local detector = cv.LineSegmentDetector{}
local lines = detector:detect{image}
image = detector:drawSegments{image, lines}

cv.imshow{"Detected lines", image}
cv.waitKey{0}
