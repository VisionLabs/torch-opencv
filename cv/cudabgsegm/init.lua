local cv = require 'cv._env'
require 'cutorch'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or require 'cv._env_cuda'

local ffi = require 'ffi'

ffi.cdef[[
struct PtrWrapper BackgroundSubtractorMOG_ctor(
        int History, int NMixtures, double BackgroundRatio, double NoiseSigma);

struct TensorWrapper BackgroundSubtractorMOG_apply(struct cutorchInfo info,
                                                    struct PtrWrapper ptr, struct TensorWrapper image,
                                                    struct TensorWrapper fgmask, double learningRate);

struct TensorWrapper BackgroundSubtractorMOG_getBackgroundImage(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper backgroundImage);

void BackgroundSubtractorMOG_setHistory(struct PtrWrapper ptr, int val);

int BackgroundSubtractorMOG_getHistory(struct PtrWrapper ptr);

void BackgroundSubtractorMOG_setNMixtures(struct PtrWrapper ptr, int val);

int BackgroundSubtractorMOG_getNMixtures(struct PtrWrapper ptr);

void BackgroundSubtractorMOG_setBackgroundRatio(struct PtrWrapper ptr, double val);

double BackgroundSubtractorMOG_getBackgroundRatio(struct PtrWrapper ptr);

void BackgroundSubtractorMOG_setNoiseSigma(struct PtrWrapper ptr, double val);

double BackgroundSubtractorMOG_getNoiseSigma(struct PtrWrapper ptr);

struct PtrWrapper BackgroundSubtractorMOG2_ctor(
        int history, double varThreshold, bool detectShadows);

struct TensorWrapper BackgroundSubtractorMOG2_apply(struct cutorchInfo info,
        struct PtrWrapper ptr, struct TensorWrapper image,
        struct TensorWrapper fgmask, double learningRate);

struct TensorWrapper BackgroundSubtractorMOG2_getBackgroundImage(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper backgroundImage);

int BackgroundSubtractorMOG2_getHistory(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setHistory(struct PtrWrapper ptr, int history);

int BackgroundSubtractorMOG2_getNMixtures(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setNMixtures(struct PtrWrapper ptr, int nmixtures);

int BackgroundSubtractorMOG2_getShadowValue(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setShadowValue(struct PtrWrapper ptr, int shadow_value);

double BackgroundSubtractorMOG2_getBackgroundRatio(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setBackgroundRatio(struct PtrWrapper ptr, double ratio);

double BackgroundSubtractorMOG2_getVarThreshold(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setVarThreshold(struct PtrWrapper ptr, double varThreshold);

double BackgroundSubtractorMOG2_getVarThresholdGen(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setVarThresholdGen(struct PtrWrapper ptr, double varThresholdGen);

double BackgroundSubtractorMOG2_getVarInit(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setVarInit(struct PtrWrapper ptr, double varInit);

double BackgroundSubtractorMOG2_getVarMin(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setVarMin(struct PtrWrapper ptr, double varMin);

double BackgroundSubtractorMOG2_getVarMax(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setVarMax(struct PtrWrapper ptr, double varMax);

bool BackgroundSubtractorMOG2_getDetectShadows(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setDetectShadows(struct PtrWrapper ptr, bool detectShadows);

double BackgroundSubtractorMOG2_getComplexityReductionThreshold(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setComplexityReductionThreshold(struct PtrWrapper ptr, double ct);

double BackgroundSubtractorMOG2_getShadowThreshold(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setShadowThreshold(struct PtrWrapper ptr, double shadowThreshold);
]]

local C = ffi.load(cv.libPath('cudabgsegm'))

require 'cv.Classes'
local Classes = ffi.load(cv.libPath('Classes'))

do
    local BackgroundSubtractorMOG = torch.class('cuda.BackgroundSubtractorMOG', 'cv.Algorithm', cv.cuda)

    function BackgroundSubtractorMOG:__init(t)
        local argRules = {
            {"history", default = 200},
            {"nmixtures", default = 5},
            {"backgroundRatio", default = 0.7},
            {"noiseSigma", default = 0}
        }
        self.ptr = ffi.gc(C.BackgroundSubtractorMOG_ctor(cv.argcheck(t, argRules)), Classes.Algorithm_dtor)
    end

    function BackgroundSubtractorMOG:apply(t)
        local argRules = {
            {"image", required = true, operator = cv.wrap_tensor},
            {"fgmask", default = nil, operator = cv.wrap_tensor},
            {"learningRate", default = -1}
        }
        return cv.unwrap_tensors(C.BackgroundSubtractor_apply(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end

    function BackgroundSubtractorMOG:getBackgroundImage(t)
        local argRules = {
            {"backgroundImage", default = nil, operator = cv.wrap_tensor}
        }
        return cv.unwrap_tensors(C.BackgroundSubtractor_getBackgroundImage(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end

    function BackgroundSubtractorMOG:setHistory(t)
        local argRules = {
            {"val", required = true}
        }
        C.BackgroundSubtractorMOG_setHistory(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG:getHistory()
        return C.BackgroundSubtractorMOG_getHistory(self.ptr)
    end

    function BackgroundSubtractorMOG:setNMixtures(t)
        local argRules = {
            {"val", required = true}
        }
        C.BackgroundSubtractorMOG_setNMixtures(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG:getNMixtures()
        return C.BackgroundSubtractorMOG_getNMixtures(self.ptr)
    end

    function BackgroundSubtractorMOG:setBackgroundRatio(t)
        local argRules = {
            {"val", required = true}
        }
        C.BackgroundSubtractorMOG_setBackgroundRatio(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG:getBackgroundRatio()
        return C.BackgroundSubtractorMOG_getBackgroundRatio(self.ptr)
    end

    function BackgroundSubtractorMOG:setNoiseSigma(t)
        local argRules = {
            {"val", required = true}
        }
        C.BackgroundSubtractorMOG_setNoiseSigma(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG:getNoiseSigma()
        return C.BackgroundSubtractorMOG_getNoiseSigma(self.ptr)
    end
end

do
    local BackgroundSubtractorMOG2 = torch.class('cuda.BackgroundSubtractorMOG2', 'cv.Algorithm', cv.cuda)
    
    function BackgroundSubtractorMOG2:__init(t)
        local argRules = {
            {"history", default = 500},
            {"varThreshold", default = 16},
            {"detectShadows", default = true}
        }
        self.ptr = ffi.gc(C.BackgroundSubtractorMOG2_ctor(cv.argcheck(t, argRules)), Classes.Algorithm_dtor)
    end

    function BackgroundSubtractorMOG2:apply(t)
        local argRules = {
            {"image", required = true, operator = cv.wrap_tensor},
            {"fgmask", default = nil, operator = cv.wrap_tensor},
            {"learningRate", default = -1}
        }
        return cv.unwrap_tensors(C.BackgroundSubtractor_apply(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end

    function BackgroundSubtractorMOG2:getBackgroundImage(t)
        local argRules = {
            {"backgroundImage", default = nil, operator = cv.wrap_tensor}
        }
        local backgroundImage = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.BackgroundSubtractor_getBackgroundImage(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end

    function BackgroundSubtractorMOG2:getHistory()
        return C.BackgroundSubtractorMOG2_getHistory(self.ptr)
    end

    function BackgroundSubtractorMOG2:setHistory(t)
        local argRules = {
            {"history", required = true}
        }
        C.BackgroundSubtractorMOG2_setHistory(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getNMixtures()
        return C.BackgroundSubtractorMOG2_getNMixtures(self.ptr)
    end

    function BackgroundSubtractorMOG2:setNMixtures(t)
        local argRules = {
            {"nmixtures", required = true}
        }
        C.BackgroundSubtractorMOG2_setNMixtures(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getShadowValue()
        return C.BackgroundSubtractorMOG2_getShadowValue(self.ptr)
    end

    function BackgroundSubtractorMOG2:setShadowValue(t)
        local argRules = {
            {"shadow_value", required = true}
        }
        C.BackgroundSubtractorMOG2_setShadowValue(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getBackgroundRatio()
        return C.BackgroundSubtractorMOG2_getBackgroundRatio(self.ptr)
    end

    function BackgroundSubtractorMOG2:setBackgroundRatio(t)
        local argRules = {
            {"ratio", required = true}
        }
        C.BackgroundSubtractorMOG2_setBackgroundRatio(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getVarThreshold()
        return C.BackgroundSubtractorMOG2_getVarThreshold(self.ptr)
    end

    function BackgroundSubtractorMOG2:setVarThreshold(t)
        local argRules = {
            {"varThreshold", required = true}
        }
        C.BackgroundSubtractorMOG2_setVarThreshold(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getVarThresholdGen()
        return C.BackgroundSubtractorMOG2_getVarThresholdGen(self.ptr)
    end

    function BackgroundSubtractorMOG2:setVarThresholdGen(t)
        local argRules = {
            {"varThresholdGen", required = true}
        }
        C.BackgroundSubtractorMOG2_setVarThresholdGen(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getVarInit()
        return C.BackgroundSubtractorMOG2_getVarInit(self.ptr)
    end

    function BackgroundSubtractorMOG2:setVarInit(t)
        local argRules = {
            {"varInit", required = true}
        }
        C.BackgroundSubtractorMOG2_setVarInit(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getVarMin()
        return C.BackgroundSubtractorMOG2_getVarMin(self.ptr)
    end

    function BackgroundSubtractorMOG2:setVarMin(t)
        local argRules = {
            {"varMin", required = true}
        }
        C.BackgroundSubtractorMOG2_setVarMin(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getVarMax()
        return C.BackgroundSubtractorMOG2_getVarMax(self.ptr)
    end

    function BackgroundSubtractorMOG2:setVarMax(t)
        local argRules = {
            {"varMax", required = true}
        }
        C.BackgroundSubtractorMOG2_setVarMax(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getDetectShadows()
        return C.BackgroundSubtractorMOG2_getDetectShadows(self.ptr)
    end

    function BackgroundSubtractorMOG2:setDetectShadows(t)
        local argRules = {
            {"detectShadows", required = true}
        }
        C.BackgroundSubtractorMOG2_setDetectShadows(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getComplexityReductionThreshold()
        return C.BackgroundSubtractorMOG2_getComplexityReductionThreshold(self.ptr)
    end

    function BackgroundSubtractorMOG2:setComplexityReductionThreshold(t)
        local argRules = {
            {"ct", required = true}
        }
        C.BackgroundSubtractorMOG2_setComplexityReductionThreshold(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getShadowThreshold()
        return C.BackgroundSubtractorMOG2_getShadowThreshold(self.ptr)
    end

    function BackgroundSubtractorMOG2:setShadowThreshold(t)
        local argRules = {
            {"threshold", required = true}
        }
        C.BackgroundSubtractorMOG2_setShadowThreshold(self.ptr, cv.argcheck(t, argRules))
    end
end

return cv.cuda
