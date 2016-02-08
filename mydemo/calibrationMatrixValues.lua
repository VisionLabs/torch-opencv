local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.calib3d'

cameraMatrix = torch.FloatTensor(3,3)
apertureWidth = 10
apertureHeight = 10

local fovx,fovy,focalLength,principalPoint,aspectRatio = cv.calibrationMatrixValues{cameraMatrix,{300,300},apertureWidth,apertureHeight}

print(fovx,fovy,focalLength,aspectRatio)

local image = cv.imread{arg[1] or 'demo/lena.jpg'}
cv.imshow{winname="Original image with text", image = image}
cv.waitKey{0}

