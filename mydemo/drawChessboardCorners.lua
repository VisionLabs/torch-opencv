local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.calib3d'


if not arg[1] then
	print('Usage: `th demo/filtering.lua path-to-image`')
    print('Now using demo/lena.jpg')
end

local image = cv.imread{arg[1] or 'demo/lena.jpg'}

if image:nDimension() == 0 then
	print("Problem loading image\n")
	os.exit(0)
end

corners = torch.FloatTensor(16,2)

for i = 1,4 do
    for j = 1,4 do
        corners[(i-1)*4+j][1] = j*100
        corners[(i-1)*4+j][2] = i*100
    end
end

--cv.drawChessboardCorners{image=image,patternSize={4,4},corners = corners,patternWasFound = true}
cv.drawChessboardCorners{image,{4,4},corners,true}

cv.imshow{winname="Original image with text", image=image}
cv.waitKey{0}

