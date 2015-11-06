require 'cv.photo'
require 'cv.imgcodecs'
require 'cv.highgui'

local image1 = cv.imread{arg[1]}
local image2 = cv.imread{arg[2]}

image = {image1, image2}

dst = cv.fastNlMeansDenoisingColoredMulti{image, nil, 1, 1};

cv.imshow{winname = "Denoising image", image = dst}
cv.waitKey{0}