local cv = require 'cv'
require 'cv.features2d'
require 'cv.imgcodecs'

if not arg[1] then
	print('Usage: `th demo/descriptors.lua path-to-image`\n')
	os.exit(0)
end

local image = cv.imread{arg[1]}

if image:nDimension() == 0 then
	print("Problem loading image\n")
	os.exit(0)
end

local keyPts = cv.AGAST{image, 0.3, false}

-- show image to the user after cv.drawKeypoints...
-- <>

-- filter keypoints
keyPts = cv.KeyPointsFilter:runByImageBorder{keyPts, image:size():totable(), 40}

-- show again, with reduced number of keypoints...
-- <>