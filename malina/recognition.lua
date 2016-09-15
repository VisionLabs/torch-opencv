cv = require 'cv'
require 'cv.imgproc'
require 'cv.videoio'
require 'cv.highgui'
cv.ml = require 'cv.ml'
require 'cv.imgcodecs'

require 'nn'
require 'dpnn'
require 'xlua'
require 'align_dlib'

torch.setdefaulttensortype('torch.FloatTensor')

faceDetector = cv.FaceDetector{}
landmarkDetector = cv.LandmarkDetector{
    "/home/shrubb/Programs/dlib-19.1/shape_predictor_68_face_landmarks.dat"}

-- OpenFace convolutional neural network face descriptor
print('Loading the CNN...')
local cnnURL = 'http://openface-models.storage.cmusatyalab.org/nn4.small2.v1.t7'
local cnnName = paths.basename(cnnURL)
if not paths.filep(cnnName) then os.execute('wget '..cnnURL) end

cnn = torch.load(cnnName)
cnn:evaluate()

knn = cv.ml.KNearest{}
knn:setDefaultK{1}
knnTrainData = torch.FloatTensor(136, 128)
knnTrainResponses = torch.range(1, 136) -- fake 'classes'

function getDescriptor(image)
    local imageGray = cv.cvtColor{image, nil, cv.COLOR_BGR2GRAY}
    local faces = faceDetector:detect{imageGray}
    assert(faces.size > 0)

    local f = faces.data[1]
    local faceCropped = 
        cv.getRectSubPix{
            image, 
            {f.width-2, f.height-2}, 
            {f.x + f.width/2, f.y + f.height/2}
        }:float()

    -- cv.imshow{'original', faceCropped / 255}
    local landmarks = landmarkDetector:detect{imageGray, faces.data[1]}
    local faceAligned = align{image, landmarks}
    -- cv.imshow{'aligned', faceAligned / 255}
    -- cv.waitKey{0}

    local netInput = faceAligned:div(255):permute(3,1,2):clone()
    return cnn:forward(netInput:view(1, 3, 96, 96))
end

for i = 1,136 do
    -- display progress bar
    xlua.progress(i, 136)

    local image = cv.imread{'dataset/'..i..'.jpg'}:float()
    local descriptor = getDescriptor(image)

    knnTrainData[i]:copy(descriptor)
end

print('')

collectgarbage()
knn:train{knnTrainData, cv.ml.ROW_SAMPLE, knnTrainResponses}

local sample = cv.imread{'sample.jpg'}
local imageGray = cv.cvtColor{sample, nil, cv.COLOR_BGR2GRAY}
local faces = faceDetector:detect{imageGray:float()}
local f = faces.data[1]
local faceCropped = 
    cv.getRectSubPix{
        sample, 
        {f.width-2, f.height-2}, 
        {f.x + f.width/2, f.y + f.height/2}
    }:float()
local netInput = cv.resize{faceCropped, {96, 96}}:div(255):permute(3,1,2):clone()

descriptor = cnn:forward(netInput:view(1, 3, 96, 96))