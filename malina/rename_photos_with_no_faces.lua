require 'paths'
require 'xlua'

cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'

faceDetector = cv.FaceDetector{}

PERIOD = ('.'):byte()
UNDERSCORE = ('_'):byte()

local count = 0
local fileList = paths.dir('.')

local function isGrayscale(image)
    return 
        image:nDimension() == 2 or 
        image:size(3) == 1 or
        image:mean(1):mean(2):squeeze():std() < 0.00001
end

if arg[1] == '-i' then
    for i, file in ipairs(fileList) do
        if file:byte(1) == UNDERSCORE then
            os.rename(file, file:sub(2))
        end
        xlua.progress(i, #fileList - 2)
    end

    print('')
    return
end

for i, file in ipairs(fileList) do
    if i % 50 == 0 then
        collectgarbage()
    end
    xlua.progress(i, #fileList)

    if file:byte() ~= PERIOD then
        local image = cv.imread{file}:float()
        local imageGray

        local ok = true

        if image:nDimension() == 0 then
            ok = false
        elseif isGrayscale(image) then
            ok = false
        else
            imageGray = cv.cvtColor{image, nil, cv.COLOR_BGR2GRAY}
            local faces = faceDetector:detect{imageGray}
            if faces.size ~= 1 then
                ok = false
            end
        end
        
        if not ok then
            -- rename this file
            os.rename(file, '_'..file)
        end
    end
end

print('')
