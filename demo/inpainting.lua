require 'cv.photo'
require 'cv.imgcodecs'
require 'cv.highgui'

if not arg[1] or not arg[2] then
    print('Usage: `th demo/inpainting.lua path_to_image path_to_mask`\n')
    os.exit(0)
end

local image = cv.imread(arg[1])
local mask = cv.imread(arg[2], cv.CV_8UC1)

local dst_image = image * 0
cv.inpaint{src = image, inpaintMask = mask, dst = dst_image, inpaintRadius = 1, flags = cv.INPAINT_TELEA}

cv.imshow("Inpaint image", dst_image)
cv.waitKey(0)