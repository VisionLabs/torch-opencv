local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'

local image = cv.imread{arg[1] or 'demo/lena.jpg'}

local dst = cv.GaussianBlur{image, 3, 2, image}

cv.imshow{winname="Original image with text", image = dst}

cv.waitKey{0}

