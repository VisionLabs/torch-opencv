local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'

local img = cv.imread{arg[1] or 'demo/lena.jpg'}
local dst = cv.imread{arg[1] or 'demo/lena2.jpg'}


local dst2 = cv.cvtColor{src = img, dst = dst, code = cv.COLOR_RGB2GRAY}

cv.imshow{winname="Original image with text", image = img}
cv.imshow{winname="Original image with text2", image = dst}

cv.waitKey{0}

