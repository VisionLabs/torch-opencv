-- a translated demo from here:
-- http://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html

require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'

local cap = cv.VideoCapture{device=0}
if not cap:isOpened() then
	print("Failed to open the default camera")
	os.exit(-1)
end

cv.namedWindow{winname="edges", flags=cv.WINDOW_AUTOSIZE}
local _, frame = cap:read{}
local edges

while true do
	edges = cv.cvtColor{src=frame, code=cv.COLOR_BGR2GRAY}
	
	cv.GaussianBlur{
		src = edges, 
		dst = edges, 
		ksize = {7,7}, 
		sigmaX = 1.5,
		sigmaY = 1.5
	}
	
	cv.Canny{
		image = edges,
		edges = edges,
		threshold1 = 0,
		threshold2 = 30,
		apertureSize = 3
	}

	cv.imshow{winname="edges", image=edges}
	if cv.waitKey{30} >= 0 then break end

	cap:read{frame}
end
