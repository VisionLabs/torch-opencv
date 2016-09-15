local cv = require 'cv'
require 'cv.imgproc'
require 'cv.videoio'
require 'cv.highgui'

local cap = cv.VideoCapture{device=0}
if not cap:isOpened() then
    print("Failed to open the default camera")
    os.exit(-1)
end

local _, frame = cap:read{}
-- make a tensor of same type & size, but a 2-dimensional one
local gray = frame.new(frame:size(1), frame:size(2))
local grayFloat = torch.FloatTensor(frame:size(1), frame:size(2))

local faceDetector = cv.FaceDetector{}
local landmarkDetector = cv.LandmarkDetector{
    "/home/shrubb/Programs/dlib-19.1/shape_predictor_68_face_landmarks.dat"}

while true do
    cv.cvtColor{frame, gray, cv.COLOR_BGR2GRAY}
    grayFloat:copy(gray)

    local faces = faceDetector:detect{grayFloat}
    for i = 1,faces.size do
        cv.rectangle2{frame, faces.data[i], {0,0,255}, 2}
    end

    local landmarks = landmarkDetector:detect{grayFloat, faces.data[1]}
    for i = 1,landmarks:size(1) do
        cv.circle{frame, {landmarks[i][1], landmarks[i][2]}, 0, {255,255,0}, 4}
    end
    
    cv.imshow{"frame", frame}
    if cv.waitKey{30} % 256 == ('q'):byte() then break end

    cap:read{frame}
end
