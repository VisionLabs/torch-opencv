local cv = require 'cv'
require 'cutorch'
require 'cv.cudaarithm'
require 'cv.highgui'
require 'cv.imgcodecs'

if not arg[1] then
    print('Usage: `th demo/cuda/convolution.lua path-to-image`')
    print('Now using demo/lena.jpg')
end

local image = cv.imread{arg[1] or 'demo/lena.jpg', cv.IMREAD_COLOR}

if not image or image:nDimension() == 0 then
    print("Problem loading image\n")
    os.exit(0)
end

image = (image:float() / 255):cuda()

local filter = torch.CudaTensor
    {
        { 0, -1,  0},
        {-1,  4, -1},
        { 0, -1,  0}
    }

local conv = cv.cuda.Convolution{}
local edges = conv:convolve{image, filter}

cv.imshow{"Edges", edges:float()}
cv.waitKey{0}
