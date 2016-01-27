local cv = require 'cv'
require 'cv.features2d'
require 'cv.imgcodecs'
require 'cv.highgui'

if not arg[1] then
    print('Usage: `th demo/descriptors.lua path-to-image`')
    print('Now using demo/lena.jpg')
end

local image = cv.imread{arg[1] or 'demo/lena.jpg'}

if not image then
    print("Problem loading image\n")
    os.exit(0)
end

cv.namedWindow{"win1"}
cv.setWindowTitle{"win1", "Original image"}
cv.imshow{"win1", image}
cv.waitKey{0}

local keyPts = cv.AGAST{image, threshold=34, nonmaxSuppression=true}

-- show keypoints to the user
local imgWithAllKeypoints = cv.drawKeypoints{image, keyPts}
cv.setWindowTitle{"win1", keyPts.size .. " keypoints by AGAST"}
cv.imshow{"win1", imgWithAllKeypoints}
cv.waitKey{0}

-- remove keypoints within 40 pixels from image border
local imageSize = {image:size()[2], image:size()[1]}
keyPts = cv.KeyPointsFilter.runByImageBorder{keyPts, imageSize, 40}

-- show again, with reduced number of keypoints
local imgWithSomeKeypoints = cv.drawKeypoints{image, keyPts}
cv.setWindowTitle{"win1", keyPts.size .. " remaining keypoints"}
cv.imshow{"win1", imgWithSomeKeypoints}
cv.waitKey{0}
