local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.calib3d'

objectPoints = {}

objectPoints[1] = torch.FloatTensor(10,1,3)
--objectPoints[2] = torch.FloatTensor(10,1,3)
--objectPoints[3] = torch.FloatTensor(10,1,3)

imagePoints = {}
imagePoints[1] = torch.FloatTensor(10,1,2)
--imagePoints[2] = torch.FloatTensor(10,1,2)
--imagePoints[3] = torch.FloatTensor(10,1,2)

cameraMatrix = torch.FloatTensor(3,3)
distCoeffs = torch.FloatTensor(5,1)

rvecs = {}
rvecs[1] = torch.FloatTensor(3,1)
rvecs[2] = torch.FloatTensor(3,1)
rvecs[3] = torch.FloatTensor(3,1)

tvecs = {}
tvecs[1] = torch.FloatTensor(3,1)
tvecs[2] = torch.FloatTensor(3,1)
tvecs[3] = torch.FloatTensor(3,1)


cv.calibrateCamera{objectPoints, imagePoints, {3,3}, cameraMatrix, distCoeffs, rvecs, tvecs}

local image = cv.imread{arg[1] or 'demo/lena.jpg'}
cv.imshow{winname="Original image with text", image = image}
cv.waitKey{0}


