local cv = require 'cv._env'

local ffi = require 'ffi'
require 'cv.Classes'

ffi.cdef[[
struct TensorWrapper BackgroundSubtractor_apply(struct PtrWrapper ptr, struct TensorWrapper image,
                        struct TensorWrapper fgmast, double learningRate);

struct TensorWrapper BackgroundSubtractor_getBackgroundImage(struct PtrWrapper ptr,
                        struct TensorWrapper backgroundImage);

struct PtrWrapper BackgroundSubtractorMOG2_ctor(int history, double varThreshold, bool detectShadows);

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

void BackgroundSubtractorMOG2_setComplexityReductionThreshold(struct PtrWrapper ptr, double complexityReductionThreshold);

double BackgroundSubtractorMOG2_getShadowThreshold(struct PtrWrapper ptr);

void BackgroundSubtractorMOG2_setShadowThreshold(struct PtrWrapper ptr, double shadowThreshold);

struct PtrWrapper BackgroundSubtractorKNN_ctor(int history, double dist2Threshold, bool detectShadows);

int BackgroundSubtractorKNN_getHistory(struct PtrWrapper ptr);

void BackgroundSubtractorKNN_setHistory(struct PtrWrapper ptr, int history);

int BackgroundSubtractorKNN_getNSamples(struct PtrWrapper ptr);

void BackgroundSubtractorKNN_setNSamples(struct PtrWrapper ptr, int nSamples);

int BackgroundSubtractorKNN_getkNNSamples(struct PtrWrapper ptr);

void BackgroundSubtractorKNN_setkNNSamples(struct PtrWrapper ptr, int kNNSamples);

int BackgroundSubtractorKNN_getShadowValue(struct PtrWrapper ptr);

void BackgroundSubtractorKNN_setShadowValue(struct PtrWrapper ptr, int shadowValue);

double BackgroundSubtractorKNN_getDist2Threshold(struct PtrWrapper ptr);

void BackgroundSubtractorKNN_setDist2Threshold(struct PtrWrapper ptr, double dist2Threshold);

double BackgroundSubtractorKNN_getShadowThreshold(struct PtrWrapper ptr);

void BackgroundSubtractorKNN_setShadowThreshold(struct PtrWrapper ptr, double shadowThreshold);

bool BackgroundSubtractorKNN_getDetectShadows(struct PtrWrapper ptr);

void BackgroundSubtractorKNN_setDetectShadows(struct PtrWrapper ptr, bool detectShadows);

struct RotatedRectPlusRect CamShift(struct TensorWrapper probImage, struct RectWrapper window,
                        struct TermCriteriaWrapper criteria);

struct RectPlusInt meanShift(struct TensorWrapper probImage, struct RectWrapper window,
                        struct TermCriteriaWrapper criteria);

struct TensorArrayPlusInt buildOpticalFlowPyramid(struct TensorWrapper img, struct TensorArray pyramid,
                        struct SizeWrapper winSize, int maxLevel, bool withDerivatives, int pyrBorder,
                        int derivBorder, bool tryReuseInputImage);

struct TensorPlusTensorPlusTensor calcOpticalFlowPyrLK(struct TensorWrapper prevImg, struct TensorWrapper nextImg,
                        struct TensorWrapper prevPts, struct TensorWrapper nextPts, struct TensorWrapper status,
                        struct TensorWrapper err, struct SizeWrapper winSize, int maxLevel,
                        struct TermCriteriaWrapper criteria, int flags, double minEigThreshold);

struct TensorWrapper calcOpticalFlowFarneback(struct TensorWrapper prev, struct TensorWrapper next,
                        struct TensorWrapper flow, double pyr_scale, int levels, int winsize,
                        int iterations, int poly_n, double poly_sigma, int flags);

struct TensorWrapper estimateRigidTransform(struct TensorWrapper src, struct TensorWrapper dst, bool fullAffine);

struct TensorPlusDouble findTransformECC(struct TensorWrapper templateImage, struct TensorWrapper inputImage,
                        struct TensorWrapper warpMatrix, int motionType, struct TermCriteriaWrapper criteria,
                        struct TensorWrapper inputMask);

struct PtrWrapper KalmanFilter_ctor_default();

struct PtrWrapper KalmanFilter_ctor(int dynamParams, int measureParams, int controlParams, int type);

void KalmanFilter_dtor(struct PtrWrapper ptr);

void KalmanFilter_init(struct PtrWrapper ptr, int dynamParams, int measureParams, int controlParams, int type);

struct TensorWrapper KalmanFilter_predict(struct PtrWrapper ptr, struct TensorWrapper control);

struct TensorWrapper KalmanFilter_correct(struct PtrWrapper ptr, struct TensorWrapper measurement);

struct TensorWrapper DenseOpticalFlow_calc(struct PtrWrapper ptr, struct TensorWrapper I0,
                        struct TensorWrapper I1, struct TensorWrapper flow);

void DenseOpticalFlow_collectGarbage(struct PtrWrapper ptr);

struct PtrWrapper DualTVL1OpticalFlow_ctor();

void DualTVL1OpticalFlow_setTau(struct PtrWrapper ptr, double val);

double DualTVL1OpticalFlow_getTau(struct PtrWrapper ptr);

void DualTVL1OpticalFlow_setLambda(struct PtrWrapper ptr, double val);

double DualTVL1OpticalFlow_getLambda(struct PtrWrapper ptr);

void DualTVL1OpticalFlow_setTheta(struct PtrWrapper ptr, double val);

double DualTVL1OpticalFlow_getTheta(struct PtrWrapper ptr);

void DualTVL1OpticalFlow_setGamma(struct PtrWrapper ptr, double val);

double DualTVL1OpticalFlow_getGamma(struct PtrWrapper ptr);

void DualTVL1OpticalFlow_setEpsilon(struct PtrWrapper ptr, double val);

double DualTVL1OpticalFlow_getEpsilon(struct PtrWrapper ptr);

void DualTVL1OpticalFlow_setScaleStep(struct PtrWrapper ptr, double val);

double DualTVL1OpticalFlow_getScaleStep(struct PtrWrapper ptr);

void DualTVL1OpticalFlow_setScalesNumber(struct PtrWrapper ptr, int val);

int DualTVL1OpticalFlow_getScalesNumber(struct PtrWrapper ptr);

void DualTVL1OpticalFlow_setWarpingsNumber(struct PtrWrapper ptr, int val);

int DualTVL1OpticalFlow_getWarpingsNumber(struct PtrWrapper ptr);

void DualTVL1OpticalFlow_setInnerIterations(struct PtrWrapper ptr, int val);

int DualTVL1OpticalFlow_getInnerIterations(struct PtrWrapper ptr);

void DualTVL1OpticalFlow_setOuterIterations(struct PtrWrapper ptr, int val);

int DualTVL1OpticalFlow_getOuterIterations(struct PtrWrapper ptr);

void DualTVL1OpticalFlow_setMedianFiltering(struct PtrWrapper ptr, int val);

int DualTVL1OpticalFlow_getMedianFiltering(struct PtrWrapper ptr);

void DualTVL1OpticalFlow_setUseInitialFlow(struct PtrWrapper ptr, bool val);

bool DualTVL1OpticalFlow_getUseInitialFlow(struct PtrWrapper ptr);

struct PtrWrapper BackgroundSubtractorMOG_ctor(int history, int nmixtures,
                        double backgroundRatio, double noiseSigma);

void BackgroundSubtractorMOG_setHistory(struct PtrWrapper ptr, int val);

int BackgroundSubtractorMOG_getHistory(struct PtrWrapper ptr);

void BackgroundSubtractorMOG_setNMixtures(struct PtrWrapper ptr, int val);

int BackgroundSubtractorMOG_getNMixtures(struct PtrWrapper ptr);

void BackgroundSubtractorMOG_setBackgroundRatio(struct PtrWrapper ptr, double backgroundRatio);

double BackgroundSubtractorMOG_getBackgroundRatio(struct PtrWrapper ptr);

void BackgroundSubtractorMOG_setNoiseSigma(struct PtrWrapper ptr, double noiseSigma);

double BackgroundSubtractorMOG_getNoiseSigma(struct PtrWrapper ptr);

struct PtrWrapper BackgroundSubtractorGMG_ctor(int initializationFrames, double decisionThreshold);

void BackgroundSubtractorGMG_setMaxFeatures(struct PtrWrapper ptr, int maxFeatures);

int BackgroundSubtractorGMG_getMaxFeatures(struct PtrWrapper ptr);

void BackgroundSubtractorGMG_setNumFrames(struct PtrWrapper ptr, int numFrames);

int BackgroundSubtractorGMG_getNumFrames(struct PtrWrapper ptr);

void BackgroundSubtractorGMG_setQuantizationLevels(struct PtrWrapper ptr, int quantizationLevels);

int BackgroundSubtractorGMG_getQuantizationLevels(struct PtrWrapper ptr);

void BackgroundSubtractorGMG_setSmoothingRadius(struct PtrWrapper ptr, int smoothingRadius);

int BackgroundSubtractorGMG_getSmoothingRadius(struct PtrWrapper ptr);

void BackgroundSubtractorGMG_setDefaultLearningRate(struct PtrWrapper ptr, double defaultLearningRate);

double BackgroundSubtractorGMG_getDefaultLearningRate(struct PtrWrapper ptr);

void BackgroundSubtractorGMG_setBackgroundPrior(struct PtrWrapper ptr, double backgroundPrior);

double BackgroundSubtractorGMG_getBackgroundPrior(struct PtrWrapper ptr);

void BackgroundSubtractorGMG_setDecisionThreshold(struct PtrWrapper ptr, double decisionThreshold);

double BackgroundSubtractorGMG_getDecisionThreshold(struct PtrWrapper ptr);

void BackgroundSubtractorGMG_setMinVal(struct PtrWrapper ptr, double minVal);

double BackgroundSubtractorGMG_getMinVal(struct PtrWrapper ptr);

void BackgroundSubtractorGMG_setMaxVal(struct PtrWrapper ptr, double maxVal);

double BackgroundSubtractorGMG_getMaxVal(struct PtrWrapper ptr);

void BackgroundSubtractorGMG_setUpdateBackgroundModel(struct PtrWrapper ptr, bool updateBackgroundModel);

bool BackgroundSubtractorGMG_getUpdateBackgroundModel(struct PtrWrapper ptr);
]]

local C = ffi.load(cv.libPath('video'))
local Classes = ffi.load(cv.libPath('Classes'))

-- BackgroundSubtractor

do
    local BackgroundSubtractor = torch.class('cv.BackgroundSubtractor', 'cv.Algorithm', cv)

    function BackgroundSubtractor:apply(t)
        local argRules = {
            {"image", required = true},
            {"fgmast", default = nil},
            {"learningRate", default = -1}
        }
        local image, fgmast, learningRate = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(
                C.BackgroundSubtractor_apply(
                    self.ptr, cv.wrap_tensors(image), cv.wrap_tensor(fgmast), learningRate))
    end

    function BackgroundSubtractor:getBackgroundImage(t)
        local argRules = {
            {"backgroundImage", default = nil}
        }
        local backgroundImage = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(
            C.BackgroundSubtractor_getBackgroundImage(self.ptr, cv.wrap_tensor(backgroundImage)))
    end
end 

-- BackgroundSubtractorMOG2

do
    local BackgroundSubtractorMOG2 = torch.class('cv.BackgroundSubtractorMOG2', 'cv.BackgroundSubtractor', cv)

    function BackgroundSubtractorMOG2:__init(t)
        local argRules = {
            {"history", default = 500},
            {"varThreshold", default = 16},
            {"detectShadows", default = true}
        }
        local history, varThreshold, detectShadows = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.BackgroundSubtractorMOG2_ctor(history, varThreshold, detectShadows), Classes.Algorithm_dtor)
    end

    function BackgroundSubtractorMOG2:getHistory()
        return C.BackgroundSubtractorMOG2_getHistory(self.ptr)
    end

    function BackgroundSubtractorMOG2:setHistory(t)
        local argRules = {
            {"history", required = true}
        }
        local history = cv.argcheck(t, argRules)

        C.BackgroundSubtractorMOG2_setHistory(self.ptr, history)
    end

    function BackgroundSubtractorMOG2:getNMixtures()
        return C.BackgroundSubtractorMOG2_getNMixtures(self.ptr)
    end

    function BackgroundSubtractorMOG2:setNMixtures(t)
        local argRules = {
            {"nmixtures", required = true}
        }
        local nmixtures = cv.argcheck(t, argRules)

        C.BackgroundSubtractorMOG2_setNMixtures(self.ptr, nmixtures)
    end

    function BackgroundSubtractorMOG2:getShadowValue()
        return C.BackgroundSubtractorMOG2_getShadowValue(self.ptr)
    end

    function BackgroundSubtractorMOG2:setShadowValue(t)
        local argRules = {
            {"shadow_value", required = true}
        }
        local shadow_value = cv.argcheck(t, argRules)

        C.BackgroundSubtractorMOG2_setShadowValue(self.ptr, shadow_value)
    end

    function BackgroundSubtractorMOG2:getBackgroundRatio()
        return C.BackgroundSubtractorMOG2_getBackgroundRatio(self.ptr)
    end

    function BackgroundSubtractorMOG2:setBackgroundRatio(t)
        local argRules = {
            {"ratio", required = true}
        }
        local ratio = cv.argcheck(t, argRules)

        C.BackgroundSubtractorMOG2_setBackgroundRatio(self.ptr, ratio)
    end

    function BackgroundSubtractorMOG2:getVarThreshold()
        return C.BackgroundSubtractorMOG2_getVarThreshold(self.ptr)
    end

    function BackgroundSubtractorMOG2:setVarThreshold(t)
        local argRules = {
            {"varThreshold", required = true}
        }
        local varThreshold = cv.argcheck(t, argRules)

        C.BackgroundSubtractorMOG2_setVarThreshold(self.ptr, varThreshold)
    end

    function BackgroundSubtractorMOG2:getVarThresholdGen()
        return C.BackgroundSubtractorMOG2_getVarThresholdGen(self.ptr)
    end

    function BackgroundSubtractorMOG2:setVarThresholdGen(t)
        local argRules = {
            {"varThresholdGen", required = true}
        }
        local varThresholdGen = cv.argcheck(t, argRules)

        C.BackgroundSubtractorMOG2_setVarThresholdGen(self.ptr, varThresholdGen)
    end

    function BackgroundSubtractorMOG2:getVarInit()
        return C.BackgroundSubtractorMOG2_getVarInit(self.ptr)
    end

    function BackgroundSubtractorMOG2:setVarInit(t)
        local argRules = {
            {"varInit", required = true}
        }
        local varInit = cv.argcheck(t, argRules)

        C.BackgroundSubtractorMOG2_setVarInit(self.ptr, varInit)
    end

    function BackgroundSubtractorMOG2:getVarMin()
        return C.BackgroundSubtractorMOG2_getVarMin(self.ptr)
    end

    function BackgroundSubtractorMOG2:setVarMin(t)
        local argRules = {
            {"varMin", required = true}
        }
        local varMin = cv.argcheck(t, argRules)

        C.BackgroundSubtractorMOG2_setVarMin(self.ptr, varMin)
    end

    function BackgroundSubtractorMOG2:getVarMax()
        return C.BackgroundSubtractorMOG2_getVarMax(self.ptr)
    end

    function BackgroundSubtractorMOG2:setVarMax(t)
        local argRules = {
            {"varMax", required = true}
        }
        local varMax = cv.argcheck(t, argRules)

        C.BackgroundSubtractorMOG2_setVarMax(self.ptr, varMax)
    end

    function BackgroundSubtractorMOG2:getDetectShadows()
        return C.BackgroundSubtractorMOG2_getDetectShadows(self.ptr)
    end

    function BackgroundSubtractorMOG2:setDetectShadows(t)
        local argRules = {
            {"detectShadows", required = true}
        }
        local detectShadows = cv.argcheck(t, argRules)

        C.BackgroundSubtractorMOG2_setDetectShadows(self.ptr, detectShadows)
    end

    function BackgroundSubtractorMOG2:getComplexityReductionThreshold()
        return C.BackgroundSubtractorMOG2_getComplexityReductionThreshold(self.ptr)
    end

    function BackgroundSubtractorMOG2:setComplexityReductionThreshold(t)
        local argRules = {
            {"ct", required = true}
        }
        local complexityReductionThreshold = cv.argcheck(t, argRules)

        C.BackgroundSubtractorMOG2_setComplexityReductionThreshold(self.ptr, complexityReductionThreshold)
    end

    function BackgroundSubtractorMOG2:getShadowThreshold()
        return C.BackgroundSubtractorMOG2_getShadowThreshold(self.ptr)
    end

    function BackgroundSubtractorMOG2:setShadowThreshold(t)
        local argRules = {
            {"threshold", required = true}
        }
        local shadowThreshold = cv.argcheck(t, argRules)

        C.BackgroundSubtractorMOG2_setShadowThreshold(self.ptr, shadowThreshold)
    end
end

-- BackgroundSubtractorKNN

do
    local BackgroundSubtractorKNN = torch.class('cv.BackgroundSubtractorKNN', 'cv.BackgroundSubtractor', cv)

    function BackgroundSubtractorKNN:__init(t)
        local argRules = {
            {"history", default = 500},
            {"dist2Threshold", default = 400.0},
            {"detectShadows", default = true}
        }
        local history, dist2Threshold, detectShadows = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.BackgroundSubtractorKNN_ctor(history, dist2Threshold, detectShadows), Classes.Algorithm_dtor)
    end

    function BackgroundSubtractorKNN:getHistory()
        return C.BackgroundSubtractorKNN_getHistory(self.ptr)
    end

    function BackgroundSubtractorKNN:setHistory(t)
        local argRules = {
            {"history", required = true}
        }
        local history = cv.argcheck(t, argRules)

        C.BackgroundSubtractorKNN_setHistory(self.ptr, history)
    end

    function BackgroundSubtractorKNN:getNSamples()
        return C.BackgroundSubtractorKNN_getNSamples(self.ptr)
    end

    function BackgroundSubtractorKNN:setNSamples(t)
        local argRules = {
            {"_nN", required = true}
        }
        local nSamples = cv.argcheck(t, argRules)

        C.BackgroundSubtractorKNN_setNSamples(self.ptr, nSamples)
    end

    function BackgroundSubtractorKNN:getkNNSamples()
        return C.BackgroundSubtractorKNN_getkNNSamples(self.ptr)
    end

    function BackgroundSubtractorKNN:setkNNSamples(t)
        local argRules = {
            {"_nkNN", required = true}
        }
        local kNNSamples = cv.argcheck(t, argRules)

        C.BackgroundSubtractorKNN_setkNNSamples(self.ptr, kNNSamples)
    end

    function BackgroundSubtractorKNN:getShadowValue()
        return C.BackgroundSubtractorKNN_getShadowValue(self.ptr)
    end

    function BackgroundSubtractorKNN:setShadowValue(t)
        local argRules = {
            {"value", required = true}
        }
        local shadowValue = cv.argcheck(t, argRules)

        C.BackgroundSubtractorKNN_setShadowValue(self.ptr, shadowValue)
    end

    function BackgroundSubtractorKNN:getDist2Threshold()
        return C.BackgroundSubtractorKNN_getDist2Threshold(self.ptr)
    end

    function BackgroundSubtractorKNN:setDist2Threshold(t)
        local argRules = {
            {"_dist2Threshold", required = true}
        }
        local dist2Threshold = cv.argcheck(t, argRules)

        C.BackgroundSubtractorKNN_setDist2Threshold(self.ptr, dist2Threshold)
    end

    function BackgroundSubtractorKNN:getShadowThreshold()
        return C.BackgroundSubtractorKNN_getShadowThreshold(self.ptr)
    end

    function BackgroundSubtractorKNN:setShadowThreshold(t)
        local argRules = {
            {"threshold", required = true}
        }
        local shadowThreshold = cv.argcheck(t, argRules)

        C.BackgroundSubtractorKNN_setShadowThreshold(self.ptr, shadowThreshold)
    end

    function BackgroundSubtractorKNN:getDetectShadows()
        return C.BackgroundSubtractorKNN_getDetectShadows(self.ptr)
    end

    function BackgroundSubtractorKNN:setDetectShadows(t)
        local argRules = {
            {"detectShadows", required = true}
        }
        local detectShadows = cv.argcheck(t, argRules)

        C.BackgroundSubtractorKNN_setDetectShadows(self.ptr, detectShadows)
    end
end

function cv.CamShift(t)
    local argRules = {
        {"probImage", required = true},
        {"window", required = true, operator = cv.Rect},
        {"criteria", required = true, operator = cv.TermCriteria}
    }
    local probImage, window, criteria = cv.argcheck(t, argRules)

    local result = C.CamShift(cv.wrap_tensor(probImage), window, criteria)
    return result.rotrect, result.rect
end

function cv.meanShift(t)
    local argRules = {
        {"probImage", required = true},
        {"window", required = true},
        {"criteria", required = true, operator = cv.TermCriteria}
    }
    local probImage, window, criteria = cv.argcheck(t, argRules)

    local result = C.meanShift(cv.wrap_tensor(probImage), window, criteria)
    return result.val, result.rect
end

function cv.buildOpticalFlowPyramid(t)
    local argRules = {
        {"img", required = true},
        {"pyramid", default = nil},
        {"winSize", required = true, operator = cv.Size},
        {"maxLevel", required = true},
        {"withDerivatives", default = true},
        {"pyrBorder", default = cv.BORDER_REFLECT_101},
        {"derivBorder", default = cv.BORDER_CONSTANT},
        {"tryReuseInputImage", default = true}
    }
    local img, pyramid, winSize, maxLevel, withDerivatives, pyrBorder, derivBorder, tryReuseInputImage = cv.argcheck(t, argRules)

    local result = C.buildOpticalFlowPyramid(cv.wrap_tensor(img), cv.wrap_tensors(pyramid), winSize, maxLevel,
        withDerivatives, pyrBorder, derivBorder, tryReuseInputImage)
    return result.val, cv.unwrap_tensors(result.tensors)
end

function cv.calcOpticalFlowPyrLK(t)
    local argRules = {
        {"prevImg", required = true},
        {"nextImg", required = true},
        {"prevPts", required = true},
        {"nextPts", required = true},
        {"status", default = nil},
        {"err", default = nil},
        {"winSize", default = {21, 21}, operator = cv.Size},
        {"maxLevel", default = 3},
        {"criteria", default = 0, operator = cv.TermCriteria},
        {"flags", default = 0},
        {"minEigThreshold", default = 1e-4}
    }
    local prevImg, nextImg, prevPts, nextPts, status, err, winSize, maxLevel, criteria, flags, minEigThreshold = cv.argcheck(t, argRules)

    local result = C.calcOpticalFlowPyrLK(cv.wrap_tensor(prevImg), cv.wrap_tensor(nextImg),
            cv.wrap_tensor(prevPts), cv.wrap_tensor(nextPts), cv.wrap_tensor(status),
            cv.wrap_tensor(err), winSize, maxLevel, criteria, flags, minEigThreshold)
    return cv.unwrap_tensors(result.tensor), cv.unwrap_tensors(result.status), cv.unwrap_tensors(result.err)
end

function cv.calcOpticalFlowFarneback(t)
    local argRules = {
        {"prev", required = true},
        {"next", required = true},
        {"flow", required = true},
        {"pyr_scale", required = true},
        {"levels", required = true},
        {"winsize", required = true},
        {"iterations", required = true},
        {"poly_n", required = true},
        {"poly_sigma", required = true},
        {"flags", required = true}
    }
    local prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.calcOpticalFlowFarneback(cv.wrap_tensor(prev), cv.wrap_tensor(next), cv.wrap_tensor(flow),
                pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags))
end

function cv.estimateRigidTransform(t)
    local argRules = {
        {"src", required = true},
        {"dst", required = true},
        {"fullAffine", required = true}
    }
    local src, dst, fullAffine = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.estimateRigidTransform(cv.wrap_tensor(src), cv.wrap_tensor(dst), fullAffine))
end

function cv.findTransformECC(t)
    local argRules = {
        {"templateImage", required = true},
        {"inputImage", required = true},
        {"warpMatrix", required = true},
        {"motionType", default = cv.MOTION_AFFINE},
        {"criteria", default = 0, operator = cv.TermCriteria},
        {"inputMask", default = nil}
    }
    local templateImage, inputImage, warpMatrix, motionType, criteria, inputMask = cv.argcheck(t, argRules)

    local result = C.findTransformECC(cv.wrap_tensor(templateImage), cv.wrap_tensor(inputImage),
                cv.wrap_tensor(warpMatrix), motionType, criteria, cv.wrap_tensor(inputMask))
    return result.val, result.tensor
end

-- KalmanFilter

do
    local KalmanFilter = torch.class('cv.KalmanFilter', cv)

    function KalmanFilter:__init(t)
        if table.getn(t) == 0 then
            self.ptr = ffi.gc(C.KalmanFilter_ctor_default(), C.KalmanFilter_dtor)
        else
            local argRules = {
                {"dynamParams", required = true},
                {"measureParams", required = true},
                {"controlParams", default = 0},
                {"type", default = cv.CV_32F}
            }
            
            local dynamParams, measureParams, controlParams, type = cv.argcheck(t, argRules)

            self.ptr = ffi.gc(C.KalmanFilter_ctor(dynamParams, measureParams, controlParams, type), C.KalmanFilter_dtor)
        end

        self.statePre = nil
        self.statePost = nil
        self.transitionMatrix = nil
        self.controlMatrix = nil
        self.measurementMatrix = nil
        self.processNoiseCov = nil
        self.measurementNoiseCov = nil
        self.errorCovPre = nil
        self.gain = nil
        self.errorCovPost = nil
        
        self.temp1 = nil
        self.temp2 = nil
        self.temp3 = nil
        self.temp4 = nil
        self.temp5 = nil
    end

    function KalmanFilter:init(t)
        local argRules = {
            {"dynamParams", required = true},
            {"measureParams", required = true},
            {"controlParams", default = 0},
            {"type", default = cv.CV_32F}
        }
            
        local dynamParams, measureParams, controlParams, type = cv.argcheck(t, argRules)

        C.KalmanFilter_init(self.ptr, dynamParams, measureParams, controlParams, type)
    end

    function KalmanFilter:predict(t)
        local argRules = {
            {"control", defauil = nil}
        }

        local control = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.KalmanFilter_predict(self.ptr, cv.wrap_tensor(control)))
    end

    function KalmanFilter:correct(t)
        local argRules = {
            {"measurement", required = true}
        }

        local measurement = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.KalmanFilter_predict(self.ptr, cv.wrap_tensor(measurement)))
    end
end

-- DenseOpticalFlow
do
    DenseOpticalFlow = torch.class('cv.DenseOpticalFlow', 'cv.Algorithm', cv)

    function DenseOpticalFlow:calc(t)
        local argRules = {
            {"I0", required = true},
            {"I1", required = true},
            {"flow", default = nil}
        }
        local I0, I1, flow = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(
            C.DenseOpticalFlow_calc(ptr, cv.wrap_tensor(I0), cv.wrap_tensor(I1), cv.wrap_tensor(flow)))
    end

    function DenseOpticalFlow:collectGarbage()
        return C.DenseOpticalFlow_collectGarbage(ptr)
    end
end

-- DualTVL1OpticalFlow

do
    DualTVL1OpticalFlow = torch.class('cv.DualTVL1OpticalFlow', 'cv.DenseOpticalFlow', cv)

    function DualTVL1OpticalFlow:__init(t)
        self.ptr = ffi.gc(C.DualTVL1OpticalFlow_ctor(), Classes.Algorithm_dtor)
    end

    function DualTVL1OpticalFlow:setTau(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setTau(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getTau()
        return C.DualTVL1OpticalFlow_getTau(self.ptr)
    end

    function DualTVL1OpticalFlow:setLambda(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setLambda(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getLambda()
        return C.DualTVL1OpticalFlow_getLambda(self.ptr)
    end

    function DualTVL1OpticalFlow:setTheta(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setTheta(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getTheta()
        return C.DualTVL1OpticalFlow_getTheta(self.ptr)
    end

    function DualTVL1OpticalFlow:setGamma(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setGamma(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getGamma()
        return C.DualTVL1OpticalFlow_getGamma(self.ptr)
    end

    function DualTVL1OpticalFlow:setEpsilon(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setEpsilon(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getEpsilon()
        return C.DualTVL1OpticalFlow_getEpsilon(self.ptr)
    end

    function DualTVL1OpticalFlow:setScaleStep(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setScaleStep(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getScaleStep()
        return C.DualTVL1OpticalFlow_getScaleStep(self.ptr)
    end

    function DualTVL1OpticalFlow:setScalesNumber(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setScalesNumber(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getScalesNumber()
        return C.DualTVL1OpticalFlow_getScalesNumber(self.ptr)
    end

    function DualTVL1OpticalFlow:setWarpingsNumber(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setWarpingsNumber(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getWarpingsNumber()
        return C.DualTVL1OpticalFlow_getWarpingsNumber(self.ptr)
    end

    function DualTVL1OpticalFlow:setInnerIterations(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setInnerIterations(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getInnerIterations()
        return C.DualTVL1OpticalFlow_getInnerIterations(self.ptr)
    end

    function DualTVL1OpticalFlow:setOuterIterations(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setOuterIterations(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getOuterIterations()
        return C.DualTVL1OpticalFlow_getOuterIterations(self.ptr)
    end

    function DualTVL1OpticalFlow:setMedianFiltering(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setMedianFiltering(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getMedianFiltering()
        return C.DualTVL1OpticalFlow_getMedianFiltering(self.ptr)
    end

    function DualTVL1OpticalFlow:setUseInitialFlow(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setUseInitialFlow(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getUseInitialFlow()
        return C.DualTVL1OpticalFlow_getUseInitialFlow(self.ptr)
    end
end

-- BackgroundSubtractorMOG

do
    BackgroundSubtractorMOG = torch.class('cv.BackgroundSubtractorMOG', 'cv.BackgroundSubtractor', cv)

    function BackgroundSubtractorMOG:__init(t)
        local argRules = {
            {"history", default = 200},
            {"nmixtures", default = 5},
            {"backgroundRatio", default = 0.7},
            {"noiseSigma", default = 0}
        }
        local history, nmixtures, backgroundRatio, noiseSigma = cv.argcheck(t, argRules)
        
        self.ptr = ffi.gc(C.BackgroundSubtractorMOG_ctor(history, nmixtures, backgroundRatio, noiseSigma), Classes.Algorithm_dtor)
    end

    function BackgroundSubtractorMOG:setHistory(t)
        local argRules = {
            {"nframes", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.BackgroundSubtractorMOG_setHistory(self.ptr, val)
    end

    function BackgroundSubtractorMOG:getHistory()
        return C.BackgroundSubtractorMOG_getHistory(self.ptr)
    end

    function BackgroundSubtractorMOG:setNMixtures(t)
        local argRules = {
            {"nmix", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.BackgroundSubtractorMOG_setNMixtures(self.ptr, val)
    end

    function BackgroundSubtractorMOG:getNMixtures()
        return C.BackgroundSubtractorMOG_getNMixtures(self.ptr)
    end

    function BackgroundSubtractorMOG:setBackgroundRatio(t)
        local argRules = {
            {"backgroundRatio", required = true}
        }
        local backgroundRatio = cv.argcheck(t, argRules)

        C.BackgroundSubtractorMOG_setBackgroundRatio(self.ptr, backgroundRatio)
    end

    function BackgroundSubtractorMOG:getBackgroundRatio()
        return C.BackgroundSubtractorMOG_getBackgroundRatio(self.ptr)
    end

    function BackgroundSubtractorMOG:setNoiseSigma(t)
        local argRules = {
            {"noiseSigma", required = true}
        }
        local noiseSigma = cv.argcheck(t, argRules)

        C.BackgroundSubtractorMOG_setNoiseSigma(self.ptr, noiseSigma)
    end

    function BackgroundSubtractorMOG:getNoiseSigma()
        return C.BackgroundSubtractorMOG_getNoiseSigma(self.ptr)
    end
end

-- BackgroundSubtractorGMG

do
    BackgroundSubtractorGMG = torch.class('cv.BackgroundSubtractorGMG', 'cv.BackgroundSubtractor', cv)

    function BackgroundSubtractorGMG:__init(t)
        local argRules = {
            {"initializationFrames", default = 120},
            {"decisionThreshold", default = 0.8}
        }
        local initializationFrames, decisionThreshold = cv.argcheck(t, argRules)
        
        self.ptr = ffi.gc(C.BackgroundSubtractorGMG_ctor(initializationFrames, decisionThreshold), Classes.Algorithm_dtor)
    end

    function BackgroundSubtractorGMG:setMaxFeatures(t)
        local argRules = {
            {"maxFeatures", required = true}
        }
        local maxFeatures = cv.argcheck(t, argRules)

        C.BackgroundSubtractorGMG_setMaxFeatures(self.ptr, maxFeatures)
    end

    function BackgroundSubtractorGMG:getMaxFeatures()
        return C.BackgroundSubtractorGMG_getMaxFeatures(self.ptr)
    end

    function BackgroundSubtractorGMG:setNumFrames(t)
        local argRules = {
            {"nframes", required = true}
        }
        local numFrames = cv.argcheck(t, argRules)

        C.BackgroundSubtractorGMG_setNumFrames(self.ptr, numFrames)
    end

    function BackgroundSubtractorGMG:getNumFrames()
        return C.BackgroundSubtractorGMG_getNumFrames(self.ptr)
    end

    function BackgroundSubtractorGMG:setQuantizationLevels(t)
        local argRules = {
            {"nlevels", required = true}
        }
        local quantizationLevels = cv.argcheck(t, argRules)

        C.BackgroundSubtractorGMG_setQuantizationLevels(self.ptr, quantizationLevels)
    end

    function BackgroundSubtractorGMG:getQuantizationLevels()
        return C.BackgroundSubtractorGMG_getQuantizationLevels(self.ptr)
    end

    function BackgroundSubtractorGMG:setSmoothingRadius(t)
        local argRules = {
            {"radius", required = true}
        }
        local smoothingRadius = cv.argcheck(t, argRules)

        C.BackgroundSubtractorGMG_setSmoothingRadius(self.ptr, smoothingRadius)
    end

    function BackgroundSubtractorGMG:getSmoothingRadius()
        return C.BackgroundSubtractorGMG_getSmoothingRadius(self.ptr)
    end

    function BackgroundSubtractorGMG:setDefaultLearningRate(t)
        local argRules = {
            {"lr", required = true}
        }
        local defaultLearningRate = cv.argcheck(t, argRules)

        C.BackgroundSubtractorGMG_setDefaultLearningRate(self.ptr, defaultLearningRate)
    end

    function BackgroundSubtractorGMG:getDefaultLearningRate()
        return C.BackgroundSubtractorGMG_getDefaultLearningRate(self.ptr)
    end

    function BackgroundSubtractorGMG:setBackgroundPrior(t)
        local argRules = {
            {"bgprior", required = true}
        }
        local backgroundPrior = cv.argcheck(t, argRules)

        C.BackgroundSubtractorGMG_setBackgroundPrior(self.ptr, backgroundPrior)
    end

    function BackgroundSubtractorGMG:getBackgroundPrior()
        return C.BackgroundSubtractorGMG_getBackgroundPrior(self.ptr)
    end

    function BackgroundSubtractorGMG:setDecisionThreshold(t)
        local argRules = {
            {"thresh", required = true}
        }
        local decisionThreshold = cv.argcheck(t, argRules)

        C.BackgroundSubtractorGMG_setDecisionThreshold(self.ptr, decisionThreshold)
    end

    function BackgroundSubtractorGMG:getDecisionThreshold()
        return C.BackgroundSubtractorGMG_getDecisionThreshold(self.ptr)
    end

    function BackgroundSubtractorGMG:setMinVal(t)
        local argRules = {
            {"val", required = true}
        }
        local minVal = cv.argcheck(t, argRules)

        C.BackgroundSubtractorGMG_setMinVal(self.ptr, minVal)
    end

    function BackgroundSubtractorGMG:getMinVal()
        return C.BackgroundSubtractorGMG_getMinVal(self.ptr)
    end

    function BackgroundSubtractorGMG:setMaxVal(t)
        local argRules = {
            {"val", required = true}
        }
        local maxVal = cv.argcheck(t, argRules)

        C.BackgroundSubtractorGMG_setMaxVal(self.ptr, maxVal)
    end

    function BackgroundSubtractorGMG:getMaxVal()
        return C.BackgroundSubtractorGMG_getMaxVal(self.ptr)
    end

    function BackgroundSubtractorGMG:setUpdateBackgroundModel(t)
        local argRules = {
            {"update", required = true}
        }
        local updateBackgroundModel = cv.argcheck(t, argRules)

        C.BackgroundSubtractorGMG_setUpdateBackgroundModel(self.ptr, updateBackgroundModel)
    end

    function BackgroundSubtractorGMG:getUpdateBackgroundModel()
        return C.BackgroundSubtractorGMG_getUpdateBackgroundModel(self.ptr)
    end
end

return cv