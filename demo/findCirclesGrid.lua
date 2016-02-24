local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.calib3d'

local img = cv.imread{arg[1] or 'demo/data/circles_pattern.png'}

local flag , centers = cv.findCirclesGrid{image = img, patternSize = {4,11}, flags = cv.CALIB_CB_ASYMMETRIC_GRID}

print(flag)

for i = 1, (#centers)[1] do
    local point = cv.Point(centers[i][1][1], centers[i][1][2])
    cv.circle{img = img, center = point, radius = 2, color = {0,0,255,1}, thickness = 3, lineType = cv.LINE_AA}
end

cv.imshow{winname="Original image with text", image = img}

cv.waitKey{0}

