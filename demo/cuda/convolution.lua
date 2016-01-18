local cv = require 'cv'
require 'cutorch'
require 'cv.cudaarithm'
require 'cv.highgui'
require 'cv.imgcodecs'

if not arg[1] then
    print('Usage: `th demo/cuda/convolution.lua path-to-image`')
    print('Now using demo/lena.jpg')
end

local image = cv.imread{arg[1] or 'demo/lena.jpg', cv.IMREAD_GRAYSCALE}

if not image then
    print("Problem loading image\n")
    os.exit(0)
end

image = image:float():cuda() / 255

local filter = torch.CudaTensor
    {
        {-0.3, -1.0, -0.3},
        {-1.0,  5.2, -1.0},
        {-0.3, -1.0, -0.3}
    }

local conv = cv.cuda.Convolution{}
local edges = conv:convolve{image, filter}

cv.imshow{"Edges", edges:float()}
cv.waitKey{0}

print(edges:size())

require 'cv.cudafilters'

-- You can also get `cv.CV_32F` value by calling 
-- `cv.tensorType(x)`, where x is a FloatTensor
local Sobel = cv.cuda.createSobelFilter{cv.CV_32F, cv.CV_32F, 1, 1}
edges = Sobel:apply{image}

cv.imshow{"Edges", edges:float()}
cv.waitKey{0}

print(edges:size())