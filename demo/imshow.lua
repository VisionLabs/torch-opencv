local cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.highgui'

im = cv.imread {arg[1], flags=cv.IMREAD_GRAYSCALE}
cv.imshow {winname="Hello, Lua!", image=im}
cv.waitKey{delay=0}
