local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.calib3d'

points = torch.FloatTensor(10,2)
whichImage = 1
F = torch.FloatTensor(3,3)
result = cv.computeCorrespondEpilines{points,whichImage,F}

local image = cv.imread{arg[1] or 'demo/lena.jpg'}
cv.imshow{winname="Original image with text", image = image}
cv.waitKey{0}


