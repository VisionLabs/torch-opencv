local cv = require "cv"
require 'cv.highgui'
require 'cv.imgcodecs'
require 'cv.stitching'

local imgs = {}
for i = 1,5 do
    imgs[i] = cv.imread{"demo/data/stitch/s" .. i .. ".jpg" }
    cv.imshow{"s"..i, imgs[i]}
    if not imgs[i] then
        print("Problem with loading image")
        os.exit(0)
    end
end
cv.waitKey{0}

local stitcher = cv.Stitcher{}

local timer = torch.Timer()
local status, pano = stitcher:stitch{imgs}
print("Processing time: " .. timer:time().real .. " seconds")

if status == 0 then
    cv.imshow{"pano", pano}
    cv.waitKey{0}
else
    print("Stitching fail")
end
