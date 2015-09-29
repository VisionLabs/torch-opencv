require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'

image = cv.imread("/home/shrubb/Pictures/2015-09-09/DSC_2191_o.jpg", cv.IMREAD_GRAYSCALE)
cv.imshow("Original image", image)
cv.waitKey(0)

-- output to another Tensor of same size & type
image_A = image * 0
cv.GaussianBlur{src=image, dst=image_A, ksize={7,7}, sigmaX=1.5, sigmaY=1.5}

-- or to a return value
image_B = cv.GaussianBlur{src=image, ksize={7,7}, sigmaX=1.5, sigmaY=1.5}

-- or filter in-place
cv.GaussianBlur{src=image, dst=image, ksize={7,7}, sigmaX=1.5, sigmaY=1.5}

-- results are equal
assert((image:eq(image_B) - 1):sum() == 0)
assert((image:eq(image_A) - 1):sum() == 0)

cv.imshow("Blurred", image_B)
cv.waitKey(0)

-- output to a single-channel ByteTensor of same size
edges = torch.ByteTensor(image:size()[1], image:size()[2])
cv.Canny{image=image, edges=edges, threshold1=0, threshold2=30}

-- or to a return value
edges2 = cv.Canny{image=image, threshold1=0, threshold2=30}

-- results are equal
assert((edges:eq(edges2) - 1):sum() == 0)

cv.imshow("Edges by Canny", edges2)
cv.waitKey(0)
