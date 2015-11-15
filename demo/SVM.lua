--[[
A translated demo from here:
http://docs.opencv.org/3.0.0/d1/d73/tutorial_introduction_to_svm.html

When running the above example in C++ (OpenCV 3.0.0), for some reason .getSupportVectors()
outputs [-0.008130081, 0.008163265]. That's why here I've set kernel type to quadratic.

Original version by @szagoruyko
--]]

require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.ml'

-- Data for visual representation
local width, height = 512, 512
local im = torch.ByteTensor(height, width, 3):zero()

-- Set up training data
local labelsMat = torch.IntTensor{1, -1, -1, -1}
local trainingDataMat = torch.FloatTensor{ {501, 10}, {255, 10}, {501, 255}, {10, 501} }

-- Set up SVM's parameters
local svm = cv.SVM()
svm:setType  		{cv.ml.SVM_C_SVC}
svm:setKernel		{cv.ml.SVM_POLY}
svm:setDegree 		{2}
svm:setTermCriteria {cv.TermCriteria{cv.TermCriteria_MAX_ITER, 100, 1e-6}}

-- Train the SVM
svm:train{trainingDataMat, cv.ml.ROW_SAMPLE, labelsMat}

-- Show the decision regions given by the SVM
local timer = torch.Timer()

local green, blue = torch.ByteTensor{0,255,0}, torch.ByteTensor{255,0,0}

for i=1,im:size(1) do
    for j=1,im:size(2) do
        local response, _ = svm:predict{torch.FloatTensor{{j, i}}}

        im[{i,j,{}}]:copy(response == 1 and green or blue)
    end
end

print("SVM evaluation time: " .. timer:time().real .. " seconds")

-- Show the training data
local thickness = -1
local lineType = 8
cv.circle{ im, {501,  10}, 5, {  0,   0,   0}, thickness, lineType }
cv.circle{ im, {255,  10}, 5, {255, 255, 255}, thickness, lineType }
cv.circle{ im, {501, 255}, 5, {255, 255, 255}, thickness, lineType }
cv.circle{ im, { 10, 501}, 5, {255, 255, 255}, thickness, lineType }

-- Show support vectors
thickness = 2
lineType  = 8
local sv = svm:getSupportVectors()

for i=1,sv:size(1) do
    cv.circle{im, {sv[i][1], sv[i][2]}, 6, {128,128,128}, thickness, lineType}
end

cv.imwrite{"result.png", im}          -- save the image
cv.imshow{"SVM Simple Example", im}   -- show it to the user
cv.waitKey{0}
