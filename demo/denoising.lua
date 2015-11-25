local cv = require 'cv'
require 'cv.photo'
require 'cv.imgcodecs'
require 'cv.highgui'

local image

if not arg[1] or not arg[2] then
    local lena = cv.imread{'demo/lena.jpg'}
    image = {lena, lena }
else
    image = {cv.imread{arg[1]}, cv.imread{arg[2]}}
end

local dst = cv.fastNlMeansDenoisingColoredMulti{image, nil, 1, 1};

cv.imshow{winname = "Denoising image", image = dst}
cv.waitKey{0}
