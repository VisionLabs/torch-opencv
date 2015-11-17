local cv = require 'cv._env'

local ffi = require 'ffi'

local C = ffi.load(cv.libPath('features2d'))

--- ***************** Classes *****************

require 'cv.Classes'

local Classes = ffi.load(cv.libPath('Classes'))

--[[ffi.cdef[[
struct KeyPointWrapper {
    struct Point2fWrapper pt;
    float size, angle, response;
    int octave, class_id;
};

struct PtrWrapper KeyPointsFilter_ctor();

void KeyPointsFilter_dtor(struct PtrWrapper ptr);

void KeyPointsFilter_runByImageBorder(struct PtrWrapper ptr, std::vector<cv::KeyPoint>& keypoints,
                struct SizeWrapper imageSize, int borderSize);

void KeyPointsFilter_test(struct Point2fWrapper temp);
]]

ffi.cdef[[
std::vector<cv::KeyPoint> AGAST(struct TensorWrapper image, std::vector<cv::KeyPoint> keypoints,
                int threshold, bool nonmaxSuppression);
]]

--[[function cv.KeyPoint(...)
    return ffi.new('struct KeyPointWrapper', ...)
end

-- KeyPointsFilter

do
    local KeyPointsFilter = cv.newTorchClass('cv.KeyPointsFilter')

    function KeyPointsFilter:__init()
        self.ptr = ffi.gc(C.KeyPointsFilter_ctor(), C.KeyPointsFilter_dtor)
    end

    function KeyPointsFilter:runByImageBorder(t)
        local argRules = {
            {"keypoints", required = true},
            {"imageSize", required = true, operator = cv.Size},
            {"borderSize", required = true}
        }
        local keypoints, imageSize, borderSize = cv.argcheck(t, argRules)
        
        C.KeyPointsFilter_runByImageBorder(self.ptr, keypoints, imageSize, borderSize);
    end
end]]

function AGAST(t)
    local argRules = {
        {"image", required = true},
        {"keypoints", default = nil},
        {"threshold", required = true},
        {"nonmaxSuppression", default = true}
    }
    local image, keypoints, threshold, nonmaxSuppression = cv.argcheck(t, argRules)

    return C.AGAST(cv.wrap_tensor(image), keypoints, threshold, nonmaxSuppression)
end

return cv