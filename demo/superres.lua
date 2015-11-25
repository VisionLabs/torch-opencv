-- A demo of scaling images by 2 from camera on CPU
-- Just to check sresolution works

local cv = require 'cv'
cv.superres = require 'cv.superres'
require 'cv.videoio'
require 'cv.highgui'

local sres = cv.superres.createSuperResolution_BTVL1{}

local camera = cv.superres.createFrameSource_Camera{0}
sres:setInput{camera}

local optFlow = cv.superres.createOptFlow_Farneback{}
sres:setOpticalFlow{optFlow}

-- These parameters are NOT realistic
-- They're here just to show the example runs
sres:setScale{2}
sres:setIterations{1}
sres:setTemporalAreaRadius{1}

-- skip the first frame
camera:nextFrame{}

cv.namedWindow{"Big frame", cv.WINDOW_AUTOSIZE}
local frame = sres:nextFrame{}

while true do
	cv.imshow{"Big frame", frame}
	if cv.waitKey{30} >= 0 then break end

	sres:nextFrame{frame}
end
