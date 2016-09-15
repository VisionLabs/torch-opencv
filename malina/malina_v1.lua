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

function getDescriptor(image)
    local imageGray = cv.cvtColor{image, nil, cv.COLOR_BGR2GRAY}
    local faces = faceDetector:detect{imageGray}
    if faces.size ~= 1 then return nil end

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

math.randomseed(os.time())

local datasetPath = 'result/total_artyom/'
fileList = paths.dir(datasetPath)

-- remove '.' and '..'
table.sort(fileList, function(a,b) return a>b end)
fileList[#fileList] = nil
fileList[#fileList] = nil
-- random shuffle
for i = #fileList, 2, -1 do
    local j = math.random(i)
    fileList[i], fileList[j] = fileList[j], fileList[i]
end

local n_like, n_dislike = 0, 0
local min_like, min_dislike, min_queries = 6, 6, 15
local KEY_Y = ('y'):byte()
local KEY_N = ('n'):byte()
local KEY_Q = ('q'):byte()

svm = cv.ml.SVM{}
svm:setKernel {cv.ml.SVM_RBF}
svm:setTermCriteria {cv.TermCriteria{cv.TermCriteria_MAX_ITER, 100000, 1e-6}}
svm:setGamma {1}
svm:setC {0.01}

trainData = {}
trainLabels = {}

cv.namedWindow{'mainWindow'}
local currFileIdx = 1

-- Learning phase
while n_like + n_dislike < min_queries or 
      n_like < min_like or 
      n_dislike < min_dislike do

    local descriptor
    local image

    while not descriptor do
        image = cv.imread{datasetPath..fileList[currFileIdx]}
        currFileIdx = currFileIdx + 1
        descriptor = getDescriptor(image:float())
    end

    cv.setWindowTitle{'mainWindow', 'Hot or not? '..(n_like + n_dislike)..'/'..min_queries}
    cv.imshow{'mainWindow', image}
    local key = cv.waitKey{0} % 256

    if key == KEY_Q then
        return
    elseif key == KEY_Y or key == KEY_N then
        table.insert(trainData, descriptor:clone())
        table.insert(trainLabels, key == KEY_Y and 1 or 0)

        if key == KEY_Y then
            n_like = n_like + 1
        else
            n_dislike = n_dislike + 1
        end
    end
end

collectgarbage()

-- Train the classifier
trainDataTensor = torch.FloatTensor(#trainData, trainData[1]:nElement())
for i,row in ipairs(trainData) do
    trainDataTensor[i]:copy(row)
end
trainDataCV = cv.ml.TrainData{trainDataTensor, cv.ml.ROW_SAMPLE, torch.IntTensor(trainLabels)}
-- svm:train{trainData, cv.ml.ROW_SAMPLE, trainLabels}
svm:trainAuto{trainDataCV, 4}
print('C: '..svm:getC{})
print('gamma: '..svm:getGamma{})

for i = 1,trainData:size(1) do
    print(svm:predict{trainData[i]:view(1, trainData:size(2))})
end

-- Evaluation phase
while true do
    local descriptor
    local image

    while not descriptor do
        local randIdx = math.random(#fileList)
        image = cv.imread{datasetPath..fileList[currFileIdx]}
        currFileIdx = currFileIdx + 1
        descriptor = getDescriptor(image:float())
    end
    local prediction = svm:predict{descriptor:view(1, descriptor:nElement())}

    cv.setWindowTitle{'mainWindow', prediction == 1 and 'Hot' or 'Cold'}
    cv.imshow{'mainWindow', image}
    local key = cv.waitKey{0} % 256

    if key == KEY_Q then break end
end
