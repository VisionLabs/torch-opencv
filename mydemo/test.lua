local cv = require 'cv'
local ffi = require 'ffi'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.tracking'

local C = ffi.load(cv.libPath('Common'))

local image = cv.imread{'demo/data/lena.jpg'}

if not image then
    print("Problem loading image\n")
    os.exit(0)
end

local images = {}
images[1] = image
images[2] = image
images[3] = image

-------
--[[
local cls = cv.TrackerTargetState{}
cls:setTargetHeight{55};
local class_array = {}
class_array[1] = cls
class_array[2] = cls
class_array[3] = cls

local float_array = {}
float_array[1] = 1
float_array[2] = 2
float_array[3] = 3

local val = ffi.new('struct ConfidenceMap')
val.class_array = cv.newArray("Class", class_array)
val.float_array = cv.newArray("Float", float_array)
val.size = 3

local val_array = ffi.new('struct ConfidenceMapArray')
val_array.size = 2;
val_array.array = ffi.gc(C.malloc(2 * ffi.sizeof('struct ConfidenceMap')), C.free)

val_array.array[0] = val
val_array.array[1] = val

local rect_array = {}
local rect = cv.Rect2d(2,2,20,20)
rect_array[1] = rect
rect_array[2] = rect
local array = cv.newArray("cv.Rect2d", rect_array)

--local res = cv.test(val_array)
--print(res)
]]

local int = {}
int[1] = 1
int[2] = 2
int[3] = 3
local intarray = cv.newArray("Int", int)

--------------------

local temp = cv.TrackerFeatureSet{
    detectorType = "HAAR",
    descriptorType = "HAAR",
    trackerFeatureType = "HAAR",
    position = {1,1},
    width = 1,
    height = 1,
    foreground = true,
    features = image,
    responses = image,
    trackeStateEstimatorType = "SVM",
    nFeatures = 1,
    numClassifer = 1,
    initIterations = 1,
    nFeatures = 1,
    patchSize = {1,1},
    ROI = {1,1,1,1}
}

local feature = cv.TrackerFeature{
    trackerFeatureType = "HAAR"
}

temp:extraction{
    images = images}

local v1, v2 = temp:getResponses{
    trackerFeatureType = nil,--"HAAR",
    feature = feature,
    source = 0,
    target = nil,
    id = 1,
    npoints = 2,
    selFeatures = intarray,
    images = images,
    response = image,
    npoints = 3,
    features = image,
    foreground = true,
    confidenceMaps = val_array,
    confidenceMap = val,
    ROI = {1,1,1,1},
    width = 1,
    position = {1,1},
    height = 1,
    responses = image,
    windowName = "test",
    trackerType = "TLD",
    boundingBox = array,
    img = image,
    boundingBox = rect,
    tracker_algorithm_name = "TLD"}
--[[
]]
print(temp.ptr)
print(v1)
print(v2)
print("successfull")