-- A demo of scaling images by 2 from camera on CPU
-- Just to check sresolution works

local cv = require 'cv'
cv.superres = require 'cv.superres'
require 'cv.videoio'
require 'cv.highgui'

local camera = cv.superres.createFrameSource_Camera{0}
local sres = cv.superres.createSuperResolution_BTVL1{}
local optFlow = cv.superres.createOptFlow_Farneback{}

-- These parameters are NOT realistic
-- They're here just to show the example runs
sres:setInput{camera}
sres:setScale{2}
sres:setIterations{1}
sres:setTemporalAreaRadius{1}

-- For some reason, an attempt to set custom optical flow algorithm results in a segfault
-- see https://github.com/VisionLabs/torch-opencv/issues/29

--sres:setOpticalFlow{optFlow}

-- skip the first frame
camera:nextFrame{}

cv.namedWindow{"Big frame", cv.WINDOW_AUTOSIZE}
local frame = sres:nextFrame{}

while true do
	cv.imshow{"Big frame", frame}
	if cv.waitKey{30} >= 0 then break end

	sres:nextFrame{frame}
end
