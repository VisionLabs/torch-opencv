local cv = require "cv"
require 'cv.highgui'
require 'cv.imgcodecs'
require 'cv.stitching'
require 'cv.imgproc'

local imgs = {}
for i = 1, 5 do
    imgs[i] = cv.imread{"demo/data/stitch/s" .. i .. ".jpg" }
    cv.imshow{"s"..i, imgs[i]}
    if not imgs[i] then
        print("Promlem with loading image")
        os.exit(0)
    end
end

cv.waitKey{50}

local stitcher = cv.Stitcher{}

local AAtime = os.clock()   --check processing time

local status, pano = stitcher:stitch{imgs}

local BBtime = os.clock()

print("processing time = " .. BBtime - AAtime .. " c")

if status == cv.OK then
    cv.imshow{"pano", pano}
else
    print("Stitching fail")
end

cv.waitKey{}