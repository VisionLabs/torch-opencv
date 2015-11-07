require 'cv.photo'
require 'cv.imgcodecs'
require 'cv.highgui'

local image = {cv.imread{arg[1]}, cv.imread{arg[2]}}

local dst = cv.fastNlMeansDenoisingColoredMulti{image, nil, 1, 1};

cv.imshow{winname = "Denoising image", image = dst}
cv.waitKey{0}
