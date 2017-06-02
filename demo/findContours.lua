local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'

if not arg[1] then
    print('`Usage: `th demo/descriptors.lua path-to-image`')
    print('Now using demo/data/lena.jpg')
end

local path = arg[1] or 'demo/data/lena.jpg'
local image = cv.imread{path}

if image:nDimension() == 0 then
    error('Couldn\'t load ' .. path)
end

cv.namedWindow{'win1'}
cv.setWindowTitle{'win1', 'Original image'}
cv.imshow{'win1', image}
cv.waitKey{0}

local imGray = cv.cvtColor{image, nil, cv.COLOR_BGR2GRAY}
local _, thresh = cv.threshold{imGray, nil, 170, 255, 0}

cv.setWindowTitle{'win1', 'Thresholded image'}
cv.imshow{'win1', thresh}
cv.waitKey{0}

local contours, hierarchy = cv.findContours{thresh, true, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE}
cv.drawContours{image, contours, -1, {0,255,0}, 3}

cv.setWindowTitle{'win1', 'Contours'}
cv.imshow{'win1', image}
cv.waitKey{0}