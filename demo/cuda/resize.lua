require 'cutorch'

local cv = require 'cv'
require 'cv.cudawarping' -- cv.cuda.resize
require 'cv.imgproc'     -- cv.resize
require 'cv.highgui'     -- cv.imshow
require 'cv.imgcodecs'   -- cv.imread

if not arg[1] then
    print('Usage: `th demo/cuda/resize.lua path-to-image`')
    print('Now using demo/data/lena.jpg')
    print('')
end

local img = cv.imread {arg[1] or 'demo/data/lena.jpg', cv.IMREAD_COLOR}:float() / 255
local imgCUDA = img:cuda()

require 'xlua'
local numIterations, dsize = 2000, {1024, 768}

local resized = cv.resize{img, dsize}
local resizedCUDA = torch.CudaTensor(resized:size())

print(('Doing `cv.resize{}` (CPU) %d times (OpenCV\'s number of threads is %d):')
    :format(numIterations, cv.getNumThreads{}))
local timer = torch.Timer()

for iter = 1,numIterations do
    cv.resize{img, dsize, dst=resized}
    if iter % 100 == 0 then xlua.progress(iter, numIterations) end
end
local timeCPU = timer:time().real

print(('Doing `cv.cuda.resize{}` (GPU) %d times:'):format(numIterations))
timer:reset()

for iter = 1,numIterations do
    cv.cuda.resize{imgCUDA, dsize, dst=resizedCUDA}
    if iter % 100 == 0 then xlua.progress(iter, numIterations) end
end
cutorch.synchronize()
local timeGPU = timer:time().real

-- a technical test to check if Tensor freeing works without errors
for iter = 1,40 do
    local _ = cv.cuda.resize{imgCUDA, dsize}
end
collectgarbage()

local title = 
    ("Lena resized to 1024x768 by your GPU (%.3f times faster than CPU)"):format(timeCPU / timeGPU)
cv.imshow{title, resizedCUDA:float()}
cv.waitKey{0}
