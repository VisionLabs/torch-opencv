local cv = require 'cv._env'
local ffi = require 'ffi'

ffi.cdef[[

struct TensorWrapper niBlackThreshold(struct TensorWrapper src, struct TensorWrapper dst, double maxValue, int type, int blockSize, double delta);

]]

local C = ffi.load(cv.libPath('ximgproc'))

function cv.niBlackThreshold(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil},
        {"maxValue", required = true},
        {"type", required = true},
        {"blockSize", required = true},
        {"delta", required = true},
    }

    local src, dst, maxValue, type_, blockSize, delta = cv.argcheck(t, argRules)

    assert(src:nDimension() == 2 and cv.tensorType(src) == cv.CV_8U)
    if dst then assert(dst:nDimension() == 2 and cv.tensorType(dst) == cv.CV_8U) end

    return cv.unwrap_tensors(C.niBlackThreshold(cv.wrap_tensor(src), cv.wrap_tensor(dst), maxValue, type_, blockSize, delta))
end


--- ***************** Classes *****************
require 'cv.Classes'

local Classes = ffi.load(cv.libPath('Classes'))

ffi.cdef[[

struct PtrWrapper GraphSegmentation_ctor(double sigma, float k, int min_size);

struct TensorWrapper GraphSegmentation_processImage(struct PtrWrapper ptr, struct TensorWrapper);

void GraphSegmentation_setSigma(struct PtrWrapper ptr, double s);

double GraphSegmentation_getSigma(struct PtrWrapper ptr);

void GraphSegmentation_setK(struct PtrWrapper ptr, float k);

float GraphSegmentation_getK(struct PtrWrapper ptr);

void GraphSegmentation_setMinSize(struct PtrWrapper ptr, int min_size);

int GraphSegmentation_getMinSize(struct PtrWrapper ptr);

struct PtrWrapper SelectiveSearchSegmentation_ctor();

void SelectiveSearchSegmentation_setBaseImage(struct PtrWrapper ptr, struct TensorWrapper);

void SelectiveSearchSegmentation_switchToSingleStrategy(struct PtrWrapper ptr, int, float);

void SelectiveSearchSegmentation_switchToSelectiveSearchFast(struct PtrWrapper ptr, int, int, float);

void SelectiveSearchSegmentation_switchToSelectiveSearchQuality(struct PtrWrapper ptr, int, int, float);

void SelectiveSearchSegmentation_addImage(struct PtrWrapper ptr, struct TensorWrapper);

void SelectiveSearchSegmentation_clearImages(struct PtrWrapper ptr);

void SelectiveSearchSegmentation_addGraphSegmentation(struct PtrWrapper ptr, struct GraphSegmentationPtr);

void SelectiveSearchSegmentation_clearGraphSegmentations(struct PtrWrapper ptr);

void SelectiveSearchSegmentation_addStrategy(struct PtrWrapper ptr, struct SelectiveSearchSegmentationStrategyPtr);

void SelectiveSearchSegmentation_clearStrategies(struct PtrWrapper ptr);

struct RectArray SelectiveSearchSegmentation_process(struct PtrWrapper ptr);


]]


do
    local GraphSegmentation = torch.class('cv.GraphSegmentation', 'cv.Algorithm', cv)

    function GraphSegmentation:__init(t)

        local argRules = {
            {"sigma", default = 0.5},
            {"k", default = 300},
            {"min_size", default = 100},
        }
        local sigma, k, min_size = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.GraphSegmentation_ctor(sigma, k, min_size), Classes.Algorithm_dtor)
    end

    function GraphSegmentation:processImage(t)

        local argRules = {
            {"src", required = true},
        }
        local src = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.GraphSegmentation_processImage(self.ptr, cv.wrap_tensor(src)))
    end

    function GraphSegmentation:setSigma(s)
        C.GraphSegmentation_setSigma(self.ptr, s)
    end

    function GraphSegmentation:getSigma()
        return C.GraphSegmentation_getSigma(self.ptr)
    end

    function GraphSegmentation:setK(k)
        C.GraphSegmentation_setK(self.ptr, k)
    end

    function GraphSegmentation:getK()
        return C.GraphSegmentation_getK(self.ptr)
    end

    function GraphSegmentation:setMinSize(s)
        C.GraphSegmentation_setMinSize(self.ptr, s)
    end

    function GraphSegmentation:getMinSize()
        return C.GraphSegmentation_getMinSize(self.ptr)
    end

end

do

    local SelectiveSearchSegmentation = torch.class('cv.SelectiveSearchSegmentation', 'cv.Algorithm', cv)

    function SelectiveSearchSegmentation:__init()

        self.ptr = ffi.gc(C.SelectiveSearchSegmentation_ctor(), Classes.Algorithm_dtor)
    end

    function SelectiveSearchSegmentation:setBaseImage(t)

        local argRules = {
            {"img", required = true},
        }
        local img = cv.argcheck(t, argRules)

        C.SelectiveSearchSegmentation_setBaseImage(self.ptr, cv.wrap_tensor(img))
    end

    function SelectiveSearchSegmentation:switchToSingleStrategy(t)

        local argRules = {
            {"k", default = 200},
            {"sigma", default = 0.8},
        }
        local k, sigma = cv.argcheck(t, argRules)

        C.SelectiveSearchSegmentation_switchToSingleStrategy(self.ptr, k, sigma)
    end


    function SelectiveSearchSegmentation:switchToSelectiveSearchFast(t)

        local argRules = {
            {"k", default = 150},
            {"inc_k", default = 150},
            {"sigma", default = 0.8},
        }
        local k, inc_k, sigma = cv.argcheck(t, argRules)

        C.SelectiveSearchSegmentation_switchToSelectiveSearchFast(self.ptr, k, inc_k, sigma)
    end

    function SelectiveSearchSegmentation:switchToSelectiveSearchQuality(t)

        local argRules = {
            {"k", default = 150},
            {"inc_k", default = 150},
            {"sigma", default = 0.8},
        }
        local k, inc_k, sigma = cv.argcheck(t, argRules)

        C.SelectiveSearchSegmentation_switchToSelectiveSearchQuality(self.ptr, k, inc_k, sigma)
    end

    function SelectiveSearchSegmentation:addImage(t)

        local argRules = {
            {"img", required = true},
        }
        local img = cv.argcheck(t, argRules)

        C.SelectiveSearchSegmentation_addImage(self.ptr, cv.wrap_tensor(img))
    end

    function SelectiveSearchSegmentation:clearImages()
        C.SelectiveSearchSegmentation_clearImages(self.ptr)
    end

    function SelectiveSearchSegmentation:addGraphSegmentation(gs)
        C.SelectiveSearchSegmentation_addGraphSegmentation(self.ptr, gs.ptr)
    end

    function SelectiveSearchSegmentation:clearGraphSegmentations()
        C.SelectiveSearchSegmentation_clearGraphSegmentations(self.ptr)
    end

    function SelectiveSearchSegmentation:addStrategy(s)
        C.SelectiveSearchSegmentation_addStrategy(self.ptr, s.ptr)
    end

    function SelectiveSearchSegmentation:clearStrategies()
        C.SelectiveSearchSegmentation_clearStrategies(self.ptr)
    end

    function SelectiveSearchSegmentation:process()
        return cv.gcarray(C.SelectiveSearchSegmentation_process(self.ptr))
    end

end

return cv
