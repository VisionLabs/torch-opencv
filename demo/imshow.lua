require 'cv.imgcodecs'
require 'cv.highgui'

im = cv.imread("/home/shrubb/Pictures/blending.jpg")
cv.imshow("Hello, Lua!", im)
cv.waitKey(0)
