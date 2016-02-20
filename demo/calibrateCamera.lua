local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.calib3d'

local numImages = 7
local patternSize = cv.Size(4, 3)
local imageSize = cv.Size(640, 360)

local img = {}

for i = 1, numImages do
    img[i] = cv.imread{'demo/templates/template' .. i .. '.jpg'}
end

local corners = {}
local isFound = false

for i = 1,numImages do
    isFound, corners[i] = cv.findChessboardCorners{image = img[i], patternSize = patternSize }
    print(isFound)
    local img_gray = cv.cvtColor{src = img[i], code = cv.COLOR_BGR2GRAY }
    cv.cornerSubPix{image = img_gray, corners = corners[i], winSize = cv.Size(11, 11), zeroZone = cv.Size(-1, -1),
                    criteria = cv.TermCriteria(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)}
    cv.drawChessboardCorners{image = img[i], patternSize = patternSize, corners = corners[i], patternWasFound = isFound}
    cv.imshow{winname="template", image = img[i]}
    cv.waitKey{}
end

local objectPoints = {}
for k = 1, numImages do
    objectPoints[k] = torch.FloatTensor(patternSize.height * patternSize.width, 1, 3)
    for i = 1, patternSize.height do
        for j = 1, patternSize.width do
            objectPoints[k][(i-1)*patternSize.width + j][1][1] = j;
            objectPoints[k][(i-1)*patternSize.width + j][1][2] = i;
            objectPoints[k][(i-1)*patternSize.width + j][1][3] = 0;
        end
    end
end

local retval, cameraMatrix, distCoeffs, rvecs, tvecs =
                cv.calibrateCamera{objectPoints = objectPoints, imagePoints = corners,
                                imageSize = imageSize }


local src = cv.imread{'demo/templates/image.jpg'}

local dst = cv.undistort{src = src, cameraMatrix = cameraMatrix, distCoeffs = distCoeffs}

cv.imshow{winname = "undistorted image", image = dst}
cv.waitKey{}
