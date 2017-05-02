local cv = require 'cv'
require 'cv.photo'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.imgproc'

if not arg[1] or not arg[2] then
    print('Usage: `th demo/inpainting.lua path_to_image path_to_mask`')
    print('Now using `demo/data/lena.jpg demo/data/inpainting/lena_mask.jpg')
end

local image = cv.imread{arg[1] or 'demo/data/lena.jpg', cv.IMREAD_COLOR}
local mask = cv.imread{arg[2] or 'demo/data/inpainting/lena_mask.jpg', cv.IMREAD_GRAYSCALE}

if image:nDimension() == 0 or mask:nDimension() == 0 then
    error('Couldn\'t load image or mask')
end

cv.namedWindow{"Display"}
cv.setWindowTitle{"Display", "Image + mask"}
cv.imshow{"Display", cv.addWeighted{image, 0.5, cv.cvtColor{mask, nil, cv.COLOR_GRAY2BGR}, 0.5, 0}}
cv.waitKey{0}

local result = cv.inpaint{image, mask, dst=nil, inpaintRadius=1, flags=cv.INPAINT_TELEA}

cv.setWindowTitle{"Display", "Result"}
cv.imshow{"Display", result}
cv.waitKey{0}
