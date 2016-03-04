-- a translated demo from here:
-- http://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html

local cv = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'

local cap = cv.VideoCapture{device=0}
if not cap:isOpened() then
    print("Failed to open the default camera")
    os.exit(-1)
end

cv.namedWindow{"edges", cv.WINDOW_AUTOSIZE}
local _, frame = cap:read{}
-- make a tensor of same type, but a 2-dimensional one
local edges = frame.new(frame:size()[1], frame:size()[2])

while true do
    
    cv.cvtColor{frame, edges, cv.COLOR_BGR2GRAY}

    cv.GaussianBlur{
        edges,
        ksize = {7,7},
        sigmaX = 1.5,
        sigmaY = 1.5,
        dst = edges
    }

    cv.Canny{
        image = edges,
        threshold1 = 0,
        threshold2 = 30,
        apertureSize = 3,
        edges = edges
    }

    cv.imshow{"edges", edges}
    if cv.waitKey{30} >= 0 then break end

    cap:read{frame}
end
