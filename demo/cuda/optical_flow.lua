#!/usr/bin/env luajit
if not arg[1] or not arg[2] then
    print ('Usage: '..arg[0]..' image1.png image2.png')
    print ('ex: '..arg[0]..' ../data/basketball1.png ../data/basketball2.png')
    print ('Now using demo/data/basketball*.png')
    arg[1] = 'demo/data/basketball1.png'
    arg[2] = 'demo/data/basketball2.png'
end

local cv = require 'cv'
require 'cv.cudaoptflow'
require 'cv.imgcodecs'
require 'cv.imgproc'
require 'cv.highgui'
require 'image'

-- different method expect different range of values
local flowMethods={
    Brox     ={computer=cv.cuda.BroxOpticalFlow{},      multiplier=1},
    TVL1     ={computer=cv.cuda.OpticalFlowDual_TVL1{}, multiplier=1},
    Farneback={computer=cv.cuda.FarnebackOpticalFlow{}, multiplier=255},
    --LK=cv.cuda.DensePyrLKOpticalFlow{} -- LK require explicitly CV_8UC1 format which currently torch-opencv toGpuMat doesn't support
}

function alignUp(x, alignBoundary)
    alignBoundary=alignBoundary or 512
    require 'math'
    local rowBytes=x:size(x:dim())*x:elementSize()
    local strideBytes=math.ceil(rowBytes/alignBoundary)*alignBoundary
    return strideBytes/x:elementSize()
end

-- load images and copy into memory-aligned CudaTensor
cpuImages={};
gpuImages={}

for i=1, 2 do
    cpuImages[i]=cv.imread{arg[i], cv.IMREAD_GRAYSCALE}
    assert(cpuImages[i]:nDimension() > 0, 'Could not load '..arg[i])

    cpuImages[i]=cpuImages[i]:float():div(255)
    local sizes=cpuImages[i]:size()
    local strides=torch.LongStorage{alignUp(cpuImages[i]), -1}
    gpuImages[i]=torch.CudaTensor(sizes, strides)
    gpuImages[i]:copy(cpuImages[i])
end

-- perform optical flow calculation
for key, flowMethod in pairs(flowMethods) do
    local flow = flowMethod.computer:calc{I0=gpuImages[1]*flowMethod.multiplier, I1=gpuImages[2]*flowMethod.multiplier}:float()
    local flowMag = torch.sqrt(torch.pow(flow, 2):sum(3)):squeeze()
    local flowAng = torch.atan2(flow:select(3, 1), flow:select(3, 2))
    -- visualize flow using HSV 
    -- color model ranges H:0-360, S:0-1, V:0-1 --> R:0-1, G:0-1, V:0-1
    local flowHSV=torch.cat({flowAng*180/math.pi+180, flowMag/flowMag:max(), torch.FloatTensor():ones(flowMag:size())}, 3)
    local flowRGB=(cv.cvtColor{src=flowHSV, code=cv.COLOR_HSV2RGB}*255):byte()
    cv.imshow{key, flowRGB}
end
cv.waitKey{0}
