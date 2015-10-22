require 'cv.photo'
require 'cv.imgcodecs'
require 'cv.highgui'

local image1 = cv.imread(arg[1])
local image2 = cv.imread(arg[2])

local dst_image1 = image1 * 0

image = {image1, image2}

--cv.fastNlMeansDenoising{src = image1, dst = dst_image1, h = 5, templateWindowSize = 7, searchWindowSize = 21}
--cv.fastNlMeansDenoisingColored{src = image1, dst = dst_image1, h = 5, hColor = 5, templateWindowSize = 7, searchWindowSize = 21}

--cv.fastNlMeansDenoisingMulti{srcImgs = image, dst = dst_image1, imgToDenoiseIndex = 1, temporalWindowSize = 1};
cv.fastNlMeansDenoisingColoredMulti{srcImgs = image, dst = dst_image1, imgToDenoiseIndex = 1, temporalWindowSize = 1};

cv.imshow{winname="Inpaint image", image=dst_image1}
cv.waitKey(0)
--cv.imshow("Inpaint image", dst_image2)
--cv.waitKey(0)