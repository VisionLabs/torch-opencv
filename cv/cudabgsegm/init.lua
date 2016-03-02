local cv = require 'cv._env'
require 'cutorch'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or require 'cv._env_cuda'

local ffi = require 'ffi'

ffi.cdef[[
struct PtrWrapper BackgroundSubtractorMOG_ctorCuda(
        int History, int NMixtures, double BackgroundRatio, double NoiseSigma);

struct TensorWrapper BackgroundSubtractorMOG_applyCuda(struct cutorchInfo info,
                                                    struct PtrWrapper ptr, struct TensorWrapper image,
                                                    struct TensorWrapper fgmask, double learningRate);

struct TensorWrapper BackgroundSubtractorMOG_getBackgroundImageCuda(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper backgroundImage);

void BackgroundSubtractorMOG_setHistoryCuda(struct PtrWrapper ptr, int val);

int BackgroundSubtractorMOG_getHistoryCuda(struct PtrWrapper ptr);

void BackgroundSubtractorMOG_setNMixturesCuda(struct PtrWrapper ptr, int val);

int BackgroundSubtractorMOG_getNMixturesCuda(struct PtrWrapper ptr);

void BackgroundSubtractorMOG_setBackgroundRatioCuda(struct PtrWrapper ptr, double val);

double BackgroundSubtractorMOG_getBackgroundRatioCuda(struct PtrWrapper ptr);

void BackgroundSubtractorMOG_setNoiseSigmaCuda(struct PtrWrapper ptr, double val);

double BackgroundSubtractorMOG_getNoiseSigmaCuda(struct PtrWrapper ptr);

struct PtrWrapper BackgroundSubtractorMOG2_ctorCuda(
        int history, double varThreshold, bool detectShadows);

struct TensorWrapper BackgroundSubtractorMOG2_applyCuda(struct cutorchInfo info,
        struct PtrWrapper ptr, struct TensorWrapper image,
        struct TensorWrapper fgmask, double learningRate);

struct TensorWrapper BackgroundSubtractorMOG2_getBackgroundImageCuda(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper backgroundImage);

int BackgroundSubtractorMOG2_getHistoryCuda(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setHistoryCuda(struct PtrWrapper ptr, int history);

int BackgroundSubtractorMOG2_getNMixturesCuda(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setNMixturesCuda(struct PtrWrapper ptr, int nmixtures);

int BackgroundSubtractorMOG2_getShadowValueCuda(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setShadowValueCuda(struct PtrWrapper ptr, int shadow_value);

double BackgroundSubtractorMOG2_getBackgroundRatioCuda(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setBackgroundRatioCuda(struct PtrWrapper ptr, double ratio);

double BackgroundSubtractorMOG2_getVarThresholdCuda(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setVarThresholdCuda(struct PtrWrapper ptr, double varThreshold);

double BackgroundSubtractorMOG2_getVarThresholdGenCuda(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setVarThresholdGenCuda(struct PtrWrapper ptr, double varThresholdGen);

double BackgroundSubtractorMOG2_getVarInitCuda(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setVarInitCuda(struct PtrWrapper ptr, double varInit);

double BackgroundSubtractorMOG2_getVarMinCuda(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setVarMinCuda(struct PtrWrapper ptr, double varMin);

double BackgroundSubtractorMOG2_getVarMaxCuda(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setVarMaxCuda(struct PtrWrapper ptr, double varMax);

bool BackgroundSubtractorMOG2_getDetectShadowsCuda(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setDetectShadowsCuda(struct PtrWrapper ptr, bool detectShadows);

double BackgroundSubtractorMOG2_getComplexityReductionThresholdCuda(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setComplexityReductionThresholdCuda(struct PtrWrapper ptr, double ct);

double BackgroundSubtractorMOG2_getShadowThresholdCuda(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setShadowThresholdCuda(struct PtrWrapper ptr, double shadowThreshold);
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
        self.ptr = ffi.gc(C.BackgroundSubtractorMOG_ctorCuda(cv.argcheck(t, argRules)), Classes.Algorithm_dtor)
    end

    function BackgroundSubtractorMOG:apply(t)
        local argRules = {
            {"image", required = true, operator = cv.wrap_tensor},
            {"fgmask", default = nil, operator = cv.wrap_tensor},
            {"learningRate", default = -1}
        }
        return cv.unwrap_tensors(C.BackgroundSubtractor_applyCuda(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end

    function BackgroundSubtractorMOG:getBackgroundImage(t)
        local argRules = {
            {"backgroundImage", default = nil, operator = cv.wrap_tensor}
        }
        return cv.unwrap_tensors(C.BackgroundSubtractor_getBackgroundImageCuda(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end

    function BackgroundSubtractorMOG:setHistory(t)
        local argRules = {
            {"val", required = true}
        }
        C.BackgroundSubtractorMOG_setHistoryCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG:getHistory()
        return C.BackgroundSubtractorMOG_getHistoryCuda(self.ptr)
    end

    function BackgroundSubtractorMOG:setNMixtures(t)
        local argRules = {
            {"val", required = true}
        }
        C.BackgroundSubtractorMOG_setNMixturesCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG:getNMixtures()
        return C.BackgroundSubtractorMOG_getNMixturesCuda(self.ptr)
    end

    function BackgroundSubtractorMOG:setBackgroundRatio(t)
        local argRules = {
            {"val", required = true}
        }
        C.BackgroundSubtractorMOG_setBackgroundRatioCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG:getBackgroundRatio()
        return C.BackgroundSubtractorMOG_getBackgroundRatioCuda(self.ptr)
    end

    function BackgroundSubtractorMOG:setNoiseSigma(t)
        local argRules = {
            {"val", required = true}
        }
        C.BackgroundSubtractorMOG_setNoiseSigmaCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG:getNoiseSigma()
        return C.BackgroundSubtractorMOG_getNoiseSigmaCuda(self.ptr)
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
        self.ptr = ffi.gc(C.BackgroundSubtractorMOG2_ctorCuda(cv.argcheck(t, argRules)), Classes.Algorithm_dtor)
    end

    function BackgroundSubtractorMOG2:apply(t)
        local argRules = {
            {"image", required = true, operator = cv.wrap_tensor},
            {"fgmask", default = nil, operator = cv.wrap_tensor},
            {"learningRate", default = -1}
        }
        return cv.unwrap_tensors(C.BackgroundSubtractor_applyCuda(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end

    function BackgroundSubtractorMOG2:getBackgroundImage(t)
        local argRules = {
            {"backgroundImage", default = nil, operator = cv.wrap_tensor}
        }
        local backgroundImage = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.BackgroundSubtractor_getBackgroundImageCuda(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end

    function BackgroundSubtractorMOG2:getHistory()
        return C.BackgroundSubtractorMOG2_getHistoryCuda(self.ptr)
    end

    function BackgroundSubtractorMOG2:setHistory(t)
        local argRules = {
            {"history", required = true}
        }
        C.BackgroundSubtractorMOG2_setHistoryCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getNMixtures()
        return C.BackgroundSubtractorMOG2_getNMixturesCuda(self.ptr)
    end

    function BackgroundSubtractorMOG2:setNMixtures(t)
        local argRules = {
            {"nmixtures", required = true}
        }
        C.BackgroundSubtractorMOG2_setNMixturesCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getShadowValue()
        return C.BackgroundSubtractorMOG2_getShadowValueCuda(self.ptr)
    end

    function BackgroundSubtractorMOG2:setShadowValue(t)
        local argRules = {
            {"shadow_value", required = true}
        }
        C.BackgroundSubtractorMOG2_setShadowValueCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getBackgroundRatio()
        return C.BackgroundSubtractorMOG2_getBackgroundRatioCuda(self.ptr)
    end

    function BackgroundSubtractorMOG2:setBackgroundRatio(t)
        local argRules = {
            {"ratio", required = true}
        }
        C.BackgroundSubtractorMOG2_setBackgroundRatioCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getVarThreshold()
        return C.BackgroundSubtractorMOG2_getVarThresholdCuda(self.ptr)
    end

    function BackgroundSubtractorMOG2:setVarThreshold(t)
        local argRules = {
            {"varThreshold", required = true}
        }
        C.BackgroundSubtractorMOG2_setVarThresholdCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getVarThresholdGen()
        return C.BackgroundSubtractorMOG2_getVarThresholdGenCuda(self.ptr)
    end

    function BackgroundSubtractorMOG2:setVarThresholdGen(t)
        local argRules = {
            {"varThresholdGen", required = true}
        }
        C.BackgroundSubtractorMOG2_setVarThresholdGenCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getVarInit()
        return C.BackgroundSubtractorMOG2_getVarInitCuda(self.ptr)
    end

    function BackgroundSubtractorMOG2:setVarInit(t)
        local argRules = {
            {"varInit", required = true}
        }
        C.BackgroundSubtractorMOG2_setVarInitCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getVarMin()
        return C.BackgroundSubtractorMOG2_getVarMinCuda(self.ptr)
    end

    function BackgroundSubtractorMOG2:setVarMin(t)
        local argRules = {
            {"varMin", required = true}
        }
        C.BackgroundSubtractorMOG2_setVarMinCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getVarMax()
        return C.BackgroundSubtractorMOG2_getVarMaxCuda(self.ptr)
    end

    function BackgroundSubtractorMOG2:setVarMax(t)
        local argRules = {
            {"varMax", required = true}
        }
        C.BackgroundSubtractorMOG2_setVarMaxCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getDetectShadows()
        return C.BackgroundSubtractorMOG2_getDetectShadowsCuda(self.ptr)
    end

    function BackgroundSubtractorMOG2:setDetectShadows(t)
        local argRules = {
            {"detectShadows", required = true}
        }
        C.BackgroundSubtractorMOG2_setDetectShadowsCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getComplexityReductionThreshold()
        return C.BackgroundSubtractorMOG2_getComplexityReductionThresholdCuda(self.ptr)
    end

    function BackgroundSubtractorMOG2:setComplexityReductionThreshold(t)
        local argRules = {
            {"ct", required = true}
        }
        C.BackgroundSubtractorMOG2_setComplexityReductionThresholdCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function BackgroundSubtractorMOG2:getShadowThreshold()
        return C.BackgroundSubtractorMOG2_getShadowThresholdCuda(self.ptr)
    end

    function BackgroundSubtractorMOG2:setShadowThreshold(t)
        local argRules = {
            {"threshold", required = true}
        }
        C.BackgroundSubtractorMOG2_setShadowThresholdCuda(self.ptr, cv.argcheck(t, argRules))
    end
end

return cv.cuda
