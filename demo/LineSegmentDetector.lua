local cv = require 'cv'
require 'cv.highgui'
require 'cv.imgproc'
require 'cv.imgcodecs'

if not arg[1] then
	print('Usage: `th demo/LineSegmentDetector.lua path-to-image`\n')
	print('Now using demo/lena.jpg')
end

local image = cv.imread{arg[1], cv.IMREAD_GRAYSCALE}

local detector = cv.LineSegmentDetector{}
local lines = detector:detect{image}
image = detector:drawSegments{image, lines}

cv.imshow{"Detected lines", image}
cv.waitKey{0}
