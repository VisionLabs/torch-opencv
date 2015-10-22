require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'

cap = cv.VideoCapture{device=0}
if not cap:isOpened() then
	print("Failed to open the default camera")
	os.exit(-1)
end

_, im = cap:read{}

cv.imshow{winname="camera", image=im}
cv.waitKey(0)