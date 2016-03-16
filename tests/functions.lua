local cv = require 'cv'
cv.optflow = require 'cv.optflow'
cv.flann = require 'cv.flann'
cv.ml = require 'cv.ml'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.photo'
require 'cv.calib3d'
require 'cv.ximgproc'
require 'cv.stitching'
require 'cv.xphoto'

if not arg[1] then
    print('Usage: `th demo/filtering.lua path-to-image`')
    print('Now using demo/data/lena.jpg')
end

local image = cv.imread{arg[1] or 'demo/data/lena.jpg'}

if not image then
    print("Problem loading image\n")
    os.exit(0)
end

local memDiff = 10

collectgarbage("collect")
------------------------------
print('>')

print("cv.Sobel testing...")
local memory_before = collectgarbage("count")
do
    local src = image:clone()
    local dst = torch.ByteTensor(image:size(1), image:size(2), image:size(3))

    cv.Sobel{
        src = src,
        dst = dst,
        ddepth = -1,
        dx = 1,
        dy = 1,
    }

    local dst2 = cv.Sobel{
        src = src,
        ddepth = -1,
        dx = 1,
        dy = 1,
    }

    cv.Sobel{
        src = src,
        dst = src,
        ddepth = -1,
        dx = 1,
        dy = 1,
    }

    assert((src:eq(dst) - 1):sum() == 0)
    assert((src:eq(dst2) - 1):sum() == 0)
end
collectgarbage("collect")
collectgarbage("collect")
local memory_after = collectgarbage("count")
print('memory change ' .. memory_after - memory_before .. ' Kb')
assert(memory_after - memory_before < memDiff)
print("OK")

-------------------------------
print('>')

print('cv.decolor testing...')
local memory_before = collectgarbage("count")
do
    local src = image:clone()
    local dst = torch.ByteTensor(image:size(1), image:size(2), image:size(3))
    local dst_gray = torch.ByteTensor(image:size(1), image:size(2), 1)

    cv.illuminationChange{
        src = src,
        mask = src,
        dst = dst,
        alpha = 2.5,
        beta = 2.5 }

    local dst2 = cv.illuminationChange{
        src = src,
        mask = src,
        alpha = 2.5,
        beta = 2.5 }

    cv.illuminationChange{
        src = src,
        mask = src,
        dst = src,
        alpha = 2.5,
        beta = 2.5 }


    assert((src:eq(dst) - 1):sum() == 0)
    assert((src:eq(dst2) - 1):sum() == 0)

end
collectgarbage("collect")
collectgarbage("collect")
local memory_after = collectgarbage("count")
print('memory change ', memory_after - memory_before .. ' Kb')
assert(memory_after - memory_before < memDiff)
print("OK")

-------------------------------
print('>')

print('cv.calibrateCamera testing...')
local memory_before = collectgarbage("count")
do
    local numImages = 7
    local patternSize = cv.Size(4, 3)
    local imageSize = cv.Size(640, 360)
    local imagesPathPrefix = 'demo/data/calibrateCamera/'

    local img = {}

    for i = 1, numImages do
        img[i] = cv.imread{imagesPathPrefix..'template'..i..'.jpg' }
        if not img[i] then
            print("Problem with loading template image\n")
            os.exit(0)
        end
    end

    local corners = {}
    local isFound = false

    for i = 1, numImages do
        isFound, corners[i] = cv.findChessboardCorners{image = img[i], patternSize = patternSize }

        if not isFound then
            print('Chessboard at image #'..i..' was NOT found!')
        end

        local img_gray = cv.cvtColor{img[i], code = cv.COLOR_BGR2GRAY }

        cv.cornerSubPix{
            image    = img_gray,
            corners  = corners[i],
            winSize  = cv.Size(11, 11),
            zeroZone = cv.Size(-1, -1),
            criteria = {cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1}
        }
        cv.drawChessboardCorners{
            image           = img[i],
            patternSize     = patternSize,
            corners         = corners[i],
            patternWasFound = isFound
        }
    end

    local objectPoints = {}

    for k = 1, numImages do
        objectPoints[k] = torch.FloatTensor(patternSize.height * patternSize.width, 1, 3)
        for i = 1, patternSize.height do
            for j = 1, patternSize.width do
                objectPoints[k][(i-1)*patternSize.width + j][1][1] = j;
                objectPoints[k][(i-1)*patternSize.width + j][1][2] = i;
                objectPoints[k][(i-1)*patternSize.width + j][1][3] = 0;
            end
        end
    end

    local error = cv.calibrateCamera{
        objectPoints = objectPoints,
        imagePoints  = corners,
        imageSize    = imageSize
    }

    assert(error < 0.2)
end
collectgarbage("collect")
collectgarbage("collect")
local memory_after = collectgarbage("count")
print('memory change ', memory_after - memory_before .. ' Kb')
assert(memory_after - memory_before < memDiff)
print("OK")

-------------------------------
print('>')

print('cv.LineSegmentDetector testing...')
local memory_before = collectgarbage("count")
do
    local src = image:clone()

    local src = cv.cvtColor{
        src = src,
        code = cv.COLOR_BGR2GRAY
    }

    local detector = cv.LineSegmentDetector{}
    local lines = detector:detect{
        image = src }

    local lines2 = torch.FloatTensor(lines:size(1)+1, lines:size(2), lines:size(3))

    detector:detect{
        image = src,
        lines = lines2 }

    local image_lines = detector:drawSegments{src, lines }
    local image_lines2 = detector:drawSegments{src, lines}

    assert((image_lines:eq(image_lines2) - 1):sum() == 0)
end
collectgarbage("collect")
collectgarbage("collect")
local memory_after = collectgarbage("count")
print('memory change ', memory_after - memory_before .. ' Kb')
assert(memory_after - memory_before < memDiff)
print("OK")

-------------------------------
print('>')

print('cv.BRISK testing...')
local memory_before = collectgarbage("count")
do
    local src = image:clone()

    local BRISK = cv.BRISK{}
    local keyPts = BRISK:detect{
        image = src}

    local keypoints, dst = BRISK:compute{
        image = src,
        keypoints = keyPts }

    local dst2 = torch.ByteTensor(dst:size(1), dst:size(2))

    BRISK:compute{
        image = src,
        keypoints = keyPts,
        descriptors = dst2 }

    assert((dst:eq(dst2) - 1):sum() == 0)
end
collectgarbage("collect")
collectgarbage("collect")
local memory_after = collectgarbage("count")
print('memory change ', memory_after - memory_before .. ' Kb')
assert(memory_after - memory_before < memDiff)
print("OK")

-------------------------------
print('>')

print('cv.optflow.calcOpticalFlowSF testing...')
local memory_before = collectgarbage("count")
do
    local src = image:clone()

    local RotMat2D = cv.getRotationMatrix2D{
        center = {src:size(1)/2, src:size(2)/2},
        angle = 3,
        scale = 1 }

    local dst = cv.warpAffine{
        src = src,
        M = RotMat2D}

    local flow = cv.optflow.calcOpticalFlowSF{
        from = src,
        to = dst,
        layers = 3,
        averaging_block_size = 3,
        max_flow = 5 }

    local flow2 = torch.FloatTensor(flow:size(1), flow:size(2), flow:size(3))

    cv.optflow.calcOpticalFlowSF{
        from = src,
        to = dst,
        flow = flow2,
        layers = 3,
        averaging_block_size = 3,
        max_flow = 5 }

    assert((flow:sub(25, 500, 25, 500):eq(flow2:sub(25, 500, 25, 500)) - 1):sum() == 0)
end
collectgarbage("collect")
collectgarbage("collect")
local memory_after = collectgarbage("count")
print('memory change ', memory_after - memory_before .. ' Kb')
assert(memory_after - memory_before < memDiff)
print("OK")


-------------------------------
print('>')

print('cv.flann.KDTreeIndexParams testing...')
local memory_before = collectgarbage("count")
do
    local N_samples, N_features, fieldSize, N_query = 300, 2, 600, 20
    local samples = torch.rand(N_samples, N_features):float() * fieldSize

    local params = cv.flann.KDTreeIndexParams{trees=10}
    local index = cv.flann.Index{samples, params, cv.FLANN_DIST_EUCLIDEAN}

    local query = torch.rand(1, N_features):float() * fieldSize
    local indices, dists = index:knnSearch{query, N_query}
end
collectgarbage("collect")
collectgarbage("collect")
local memory_after = collectgarbage("count")
print('memory change ', memory_after - memory_before .. ' Kb')
assert(memory_after - memory_before < memDiff)
print("OK")

-------------------------------
print('>')

print('cv.ml.SVM testing...')
local memory_before = collectgarbage("count")
do
    local width, height = 512, 512
    local im = torch.ByteTensor(height, width, 3):zero()

    local labelsMat = torch.IntTensor{1, -1, -1, -1}
    local trainingDataMat = torch.FloatTensor{ {501, 10}, {255, 10}, {501, 255}, {10, 501} }

    local svm = cv.ml.SVM()
    svm:setType  		{cv.ml.SVM_C_SVC}
    svm:setKernel		{cv.ml.SVM_POLY}
    svm:setDegree 		{2}
    svm:setTermCriteria {cv.TermCriteria{cv.TermCriteria_MAX_ITER, 100, 1e-6}}

    svm:train{trainingDataMat, cv.ml.ROW_SAMPLE, labelsMat}

    local green, blue = torch.ByteTensor{0,255,0}, torch.ByteTensor{255,0,0}

    for i=1,im:size(1) do
        for j=1,im:size(2) do
            local erespons = svm:predict{torch.FloatTensor{{j, i}}}

            im[{i,j,{}}]:copy(response == 1 and green or blue)
        end
    end
end
collectgarbage("collect")
collectgarbage("collect")
local memory_after = collectgarbage("count")
print('memory change ', memory_after - memory_before .. ' Kb')
assert(memory_after - memory_before < memDiff)
print("OK")

-------------------------------
print('>')

print('cv.Stitcher testing...')
local memory_before = collectgarbage("count")
do
    local imgs = {}
    for i = 1, 5 do
        imgs[i] = cv.imread{"demo/data/stitch/s" .. i .. ".jpg" }
        if not imgs[i] then
            print("Promlem with loading image")
            os.exit(0)
        end
    end

    local stitcher = cv.Stitcher{}

    local status, pano = stitcher:stitch{imgs}

    assert(status == 0)
end
collectgarbage("collect")
collectgarbage("collect")
local memory_after = collectgarbage("count")
print('memory change ', memory_after - memory_before .. ' Kb')
assert(memory_after - memory_before < memDiff)
print("OK")

-------------------------------
print('>')

print('cv.xphoto.autowbGrayworld testing...')
local memory_before = collectgarbage("count")
do
    local src = image:clone()
    local dst2 = torch.ByteTensor(src:size(1), src:size(2), src:size(3))

    local dst = cv.xphoto.autowbGrayworld{src = src}

    cv.xphoto.autowbGrayworld{src = src, dst = dst2}

    cv.xphoto.autowbGrayworld{src = src, dst = src}

    assert((src:eq(dst) - 1):sum() == 0)
    assert((src:eq(dst2) - 1):sum() == 0)

end
collectgarbage("collect")
collectgarbage("collect")
local memory_after = collectgarbage("count")
print('memory change ', memory_after - memory_before .. ' Kb')
assert(memory_after - memory_before < memDiff)
print("OK")
