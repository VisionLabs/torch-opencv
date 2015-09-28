require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'

image = cv.imread("/home/shrubb/Pictures/2015-09-09/DSC_2191_o.jpg", cv.IMREAD_COLOR)
cv.imshow("Original image", image)
cv.waitKey(0)

-- output to another Tensor of same size & type
image_A = image * 0
cv.GaussianBlur{src=image, dst=image_A, ksize={3,7}, sigmaX=3}

-- output to return value
image_B = cv.GaussianBlur{src=image, ksize={3,7}, sigmaX=3}

-- filter in-place
cv.GaussianBlur{src=image, dst=image, ksize={3,7}, sigmaX=3}

-- results are equal
assert((image:eq(image_B) - 1):sum() == 0)
assert((image:eq(image_A) - 1):sum() == 0)

cv.imshow("Blurred", image)
cv.waitKey(0)