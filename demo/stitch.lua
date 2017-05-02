local cv = require 'cv'
require 'cv.highgui'
require 'cv.imgcodecs'
require 'cv.stitching'

cv.namedWindow{'Display'}

local images = {}
for i = 1,5 do
    local pathToImage = ('demo/data/stitch/s%d.jpg'):format(i)
    images[i] = cv.imread{pathToImage}
    if images[i]:nElement() == 0 then
        error('Couldn\'t load ' .. pathToImage)
    end

    cv.imshow{'Display', images[i]}
    cv.waitKey{0}
end

local stitcher = cv.Stitcher{}

local timer = torch.Timer()
local status, pano = stitcher:stitch{images}
print('Processing time: ' .. timer:time().real .. ' seconds')

cv.setWindowTitle{'Display', 'Panorama'}
if status == 0 then
    cv.imshow{'Display', pano}
    cv.waitKey{0}
else
    error('Stitching fail')
end
