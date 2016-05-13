local cv = require 'cv'
require 'cv.highgui'
require 'cv.imgproc'
require 'cv.imgcodecs'

if not arg[1] then
    print('Usage: `th demo/LineSegmentDetector.lua path-to-image`')
    print('Now using demo/data/lena.jpg')
end

local image = cv.imread{arg[1] or 'demo/data/lena.jpg', cv.IMREAD_GRAYSCALE}

if image:nDimension() == 0 then
    print('Problem loading image\n')
    os.exit(0)
end

local detector = cv.LineSegmentDetector{}
local lines = detector:detect{image}
image = detector:drawSegments{image, lines}

cv.imshow{"Detected lines", image}
cv.waitKey{0}
