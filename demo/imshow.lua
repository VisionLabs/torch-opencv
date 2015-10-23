require 'cv.imgcodecs'
require 'cv.highgui'

im = cv.imread("/home/shrubb/Pictures/blending.jpg", cv.IMREAD_GRAYSCALE)
cv.imshow {winname="Hello, Lua!", image=im}
cv.waitKey{0}
