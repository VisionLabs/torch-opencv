require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'

if not arg[1] then
	print('Usage: `th demo/filtering.lua path-to-image`\n')
	os.exit(0)
end

local image = cv.imread{filename=arg[1]}

if image:nDimension() == 0 then
	print("Problem loading image\n")
	os.exit(0)
end

cv.putText{
	img=image, 
	text="Some text on top of the image", 
	org={x=50, y=50},
	fontFace=cv.FONT_HERSHEY_DUPLEX,
	fontScale=2,
	color={255, 255, 0},
	thickness=3,
}

cv.imshow{winname="Original image with text", image=image}
cv.waitKey{0}

-- output to another Tensor of same size & type...
local image_A = image * 0
cv.GaussianBlur{src=image, dst=image_A, ksize={7, 7}, sigmaX=3.5, sigmaY=3.5}

-- or to a return value...
local image_B = cv.GaussianBlur{src=image, ksize={7, 7}, sigmaX=3.5, sigmaY=3.5}

-- or filter in-place.
-- we can also specify ksize as a string-number table:
cv.GaussianBlur{src=image, dst=image, ksize={width=7, height=7}, sigmaX=3.5, sigmaY=3.5}

-- results are equal
assert((image:eq(image_B) - 1):sum() == 0)
assert((image:eq(image_A) - 1):sum() == 0)

cv.imshow{winname="Blurred", image=image_B}
cv.waitKey{0}

-- output to a single-channel ByteTensor of same size
local edges = torch.ByteTensor(image:size()[1], image:size()[2])
cv.Canny{image=image, edges=edges, threshold1=0, threshold2=30}

-- or to a return value
local edges2 = cv.Canny{image=image, threshold1=0, threshold2=30}

-- results are equal
assert((edges:eq(edges2) - 1):sum() == 0)

cv.imshow{winname="Edges by Canny", image=edges2}
cv.waitKey{0}
