local cv = require 'cv._env'
local ffi = require 'ffi'
require 'cv.Classes'

local Classes = ffi.load(cv.libPath('Classes'))
local C = ffi.load(cv.libPath('tracking'))

--- ***************** Classes *****************

ffi.cdef[[
void test();

struct FloatArrayPlusInt {
    struct FloatArray array;
    int val;
};

struct PtrWrapper WeakClassifierHaarFeature_ctor();

void WeakClassifierHaarFeature_dtor(
       struct PtrWrapper ptr);

int WeakClassifierHaarFeature_eval(
        struct PtrWrapper ptr, float value);

bool WeakClassifierHaarFeature_update(
        struct PtrWrapper ptr, float value, int target);

struct PtrWrapper BaseClassifier_ctor(
        int numWeakClassifier, int iterationInit);

void BaseClassifier_dtor(
        struct PtrWrapper ptr);

int BaseClassifier_computeReplaceWeakestClassifier(
        struct PtrWrapper ptr, struct FloatArray errors);

int BaseClassifier_eval(
        struct PtrWrapper ptr, struct TensorWrapper image);

float BaseClassifier_getError(
        struct PtrWrapper ptr, int curWeakClassifier);

struct FloatArray BaseClassifier_getErrors(
        struct PtrWrapper ptr, struct FloatArray errors);

int BaseClassifier_getIdxOfNewWeakClassifier(
        struct PtrWrapper ptr);

int BaseClassifier_getSelectedClassifier(
        struct PtrWrapper ptr);

void BaseClassifier_replaceClassifierStatistic(
        struct PtrWrapper ptr, int sourceIndex, int targetIndex);

void BaseClassifier_replaceWeakClassifier(
        struct PtrWrapper ptr, int index);

struct FloatArrayPlusInt BaseClassifier_selectBestClassifier(
        struct PtrWrapper ptr, struct BoolArray errorMask,
        float importance, struct FloatArray errors);

struct BoolArray BaseClassifier_trainClassifier(
        struct PtrWrapper ptr, struct TensorWrapper image,
        int target, float importance, struct BoolArray errorMask);

struct PtrWrapper EstimatedGaussDistribution_ctor();

struct PtrWrapper EstimatedGaussDistribution_ctor2(
        float P_mean, float R_mean, float P_sigma, float R_sigma);

void EstimatedGaussDistribution_dtor(
        struct PtrWrapper ptr);

float EstimatedGaussDistribution_getMean(
        struct PtrWrapper ptr);

float EstimatedGaussDistribution_getSigma(
        struct PtrWrapper ptr);

void EstimatedGaussDistribution_setValues(
        struct PtrWrapper ptr, float mean, float sigma);

struct PtrWrapper ClassifierThreshold_ctor(
        struct PtrWrapper posSamples,
        struct PtrWrapper negSamples);

void ClassifierThreshold_dtor(
        struct PtrWrapper ptr);

int ClassifierThreshold_eval(
        struct PtrWrapper ptr, float value);

void ClassifierThreshold_update(
        struct PtrWrapper ptr,
        float value, int target);

struct Params {
    int _numSel;
    int _numFeat;
    float _lRate;
};

struct PtrWrapper ClfMilBoost_ctor();

void ClfMilBoost_dtor(
        struct PtrWrapper ptr);

struct FloatArray ClfMilBoost_classify(
        struct PtrWrapper ptr, struct TensorWrapper x, bool logR);

void ClfMilBoost_init(
    struct PtrWrapper ptr, struct Params parameters);

float ClfMilBoost_sigmoid(
        struct PtrWrapper ptr, float x);

void ClfMilBoost_update(
        struct PtrWrapper ptr, struct TensorWrapper posx,
        struct TensorWrapper negx);

struct PtrWrapper ClfOnlineStump_ctor();

void ClfOnlineStump_dtor(
        struct PtrWrapper ptr);

bool ClfOnlineStump_classify(
        struct PtrWrapper ptr, struct TensorWrapper x, int i);

float ClfOnlineStump_classifyF(
        struct PtrWrapper ptr, struct TensorWrapper x, int i);

struct FloatArray ClfOnlineStump_classifySetF(
        struct PtrWrapper ptr, struct TensorWrapper x);

void ClfOnlineStump_init(
        struct PtrWrapper ptr);

void ClfOnlineStump_update(
        struct PtrWrapper ptr, struct TensorWrapper posx,
        struct TensorWrapper negx);

void CvParams_dtor(
        struct PtrWrapper ptr);

void CvParams_printAttrs(
        struct PtrWrapper ptr);

void CvParams_printDefaults(
        struct PtrWrapper ptr);

bool CvParams_read(
        struct PtrWrapper ptr, struct PtrWrapper node);

bool CvParams_scanAttr(
        struct PtrWrapper ptr, const char* prmName, const char* val);

void CvParams_write(
        struct PtrWrapper ptr, struct PtrWrapper fs);

struct PtrWrapper CvFeatureParams_ctor(
        int featureType);

void CvFeatureParams_init(
        struct PtrWrapper ptr, struct PtrWrapper fp);

bool CvFeatureParams_read(
        struct PtrWrapper ptr, struct PtrWrapper node);

void CvFeatureParams_write(
        struct PtrWrapper ptr, struct PtrWrapper fs);

struct PtrWrapper CvHaarFeatureParams_ctor();

void CvHaarFeatureParams_init(
        struct PtrWrapper ptr, struct PtrWrapper fp);

void CvHaarFeatureParams_printAttrs(
        struct PtrWrapper ptr);

void CvHaarFeatureParams_printDefaults(
        struct PtrWrapper ptr);

bool CvHaarFeatureParams_read(
        struct PtrWrapper ptr, struct PtrWrapper node);

bool CvHaarFeatureParams_scanAttr(
        struct PtrWrapper ptr, const char* prmName, const char* val);

void CvHaarFeatureParams_write(
        struct PtrWrapper ptr, struct PtrWrapper fs);

struct PtrWrapper CvHOGFeatureParams_ctor();

struct PtrWrapper CvLBPFeatureParams_ctor();

struct PtrWrapper CvFeatureEvaluator_ctor(
        int type);

void CvFeatureEvaluator_dtor(
        struct PtrWrapper ptr);

struct TensorWrapper CvFeatureEvaluator_getCls(
        struct PtrWrapper ptr);

float CvFeatureEvaluator_getCls2(
        struct PtrWrapper ptr, int si);

int CvFeatureEvaluator_getFeatureSize(
        struct PtrWrapper ptr);

int CvFeatureEvaluator_getMaxCatCount(
        struct PtrWrapper ptr);

int CvFeatureEvaluator_getNumFeatures(
        struct PtrWrapper ptr);

void CvFeatureEvaluator_init(
        struct PtrWrapper ptr, struct PtrWrapper _featureParams,
        int _maxSampleCount, struct SizeWrapper _winSize);

float CvFeatureEvaluator_call(
        struct PtrWrapper ptr, int featureIdx, int sampleIdx);

void CvFeatureEvaluator_setImage(
        struct PtrWrapper ptr, struct TensorWrapper img,
        unsigned char clsLabel, int idx);

void CvFeatureEvaluator_writeFeatures(
        struct PtrWrapper ptr, struct PtrWrapper fs,
        struct TensorWrapper featureMap);

struct PtrWrapper CvHaarEvaluator_ctor();

void CvHaarEvaluator_dtor(
        struct PtrWrapper ptr);

void CvHaarEvaluator_generateFeatures(
        struct PtrWrapper ptr);

void CvHaarEvaluator_generateFeatures2(
        struct PtrWrapper ptr, int numFeatures);

void CvHaarEvaluator_init(
        struct PtrWrapper ptr, struct PtrWrapper _featureParams,
        int _maxSampleCount, struct SizeWrapper _winSize);

float CvHaarEvaluator_call(
        struct PtrWrapper ptr, int featureIdx, int sampleIdx);

void CvHaarEvaluator_setImage(
        struct PtrWrapper ptr, struct TensorWrapper img,
        unsigned char clsLabel, int idx);

void CvHaarEvaluator_setWinSize(
        struct PtrWrapper ptr, struct SizeWrapper patchSize);

struct SizeWrapper CvHaarEvaluator_setWinSize2(
        struct PtrWrapper ptr);

void CvHaarEvaluator_writeFeature(
        struct PtrWrapper ptr, struct PtrWrapper fs);

void CvHaarEvaluator_writeFeatures(
        struct PtrWrapper ptr, struct PtrWrapper fs,
        struct TensorWrapper featureMap);

struct PtrWrapper CvHOGEvaluator_ctor();

void CvHOGEvaluator_dtor(
        struct PtrWrapper ptr);

void CvHOGEvaluator_init(
        struct PtrWrapper ptr, struct PtrWrapper _featureParams,
        int _maxSampleCount, struct SizeWrapper _winSize);

void CvHOGEvaluator_setImage(
        struct PtrWrapper ptr, struct TensorWrapper img,
        unsigned char clsLabel, int idx);

void CvHOGEvaluator_writeFeatures(
        struct PtrWrapper ptr, struct PtrWrapper fs,
        struct TensorWrapper featureMap);

struct PtrWrapper CvLBPEvaluator_ctor();

void CvLBPEvaluator_dtor(
        struct PtrWrapper ptr);

void CvLBPEvaluator_init(
        struct PtrWrapper ptr, struct PtrWrapper _featureParams,
        int _maxSampleCount, struct SizeWrapper _winSize);

float CvLBPEvaluator_call(
        struct PtrWrapper ptr, int featureIdx, int sampleIdx);

void CvLBPEvaluator_setImage(
        struct PtrWrapper ptr, struct TensorWrapper img,
        unsigned char clsLabel, int idx);

void CvLBPEvaluator_writeFeatures(
        struct PtrWrapper ptr, struct PtrWrapper fs,
        struct TensorWrapper featureMap);

struct PtrWrapper StrongClassifierDirectSelection_ctor(
        int numBaseClf, int numWeakClf, struct SizeWrapper patchSz,
        struct RectWrapper sampleROI, bool useFeatureEx, int iterationInit);

void StrongClassifierDirectSelection_dtor(
        struct PtrWrapper ptr);

struct FloatPlusInt {
    float v1;
    int v2;
};

struct FloatPlusInt StrongClassifierDirectSelection_classifySmooth(
        struct PtrWrapper ptr, struct TensorArray images,
        struct RectWrapper sampleROI);

float StrongClassifierDirectSelection_eval(
        struct PtrWrapper ptr, struct TensorWrapper response);

int StrongClassifierDirectSelection_getNumBaseClassifier(
        struct PtrWrapper ptr);

struct SizeWrapper StrongClassifierDirectSelection_getPatchSize(
    struct PtrWrapper ptr);

struct RectWrapper StrongClassifierDirectSelection_getROI(
        struct PtrWrapper ptr);

struct IntArray StrongClassifierDirectSelection_getSelectedWeakClassifier(
        struct PtrWrapper ptr);

int StrongClassifierDirectSelection_getSwappedClassifier(
        struct PtrWrapper ptr);

bool StrongClassifierDirectSelection_getUseFeatureExchange(
        struct PtrWrapper ptr);

void StrongClassifierDirectSelection_initBaseClassifier(
        struct PtrWrapper ptr);

void StrongClassifierDirectSelection_replaceWeakClassifier(
        struct PtrWrapper ptr, int idx);

bool StrongClassifierDirectSelection_update(
        struct PtrWrapper ptr, struct TensorWrapper image,
        int target, float importance);

struct PtrWrapper Detector_ctor(
        struct PtrWrapper ptr);

void Detector_dtor(
        struct PtrWrapper ptr);

void Detector_classifySmooth(
        struct PtrWrapper ptr, struct TensorArray image, float minMargin);

float Detector_getConfidence(
        struct PtrWrapper ptr, int patchIdx);

float Detector_getConfidenceOfBestDetection(
        struct PtrWrapper ptr);

float Detector_getConfidenceOfDetection(
        struct PtrWrapper ptr, int detectionIdx);

struct FloatArray Detector_getConfidences(
        struct PtrWrapper ptr);

struct TensorWrapper Detector_getConfImageDisplay(
        struct PtrWrapper ptr);

struct IntArray Detector_getIdxDetections(
        struct PtrWrapper ptr);

int Detector_getNumDetections(
        struct PtrWrapper ptr);

int Detector_getPatchIdxOfDetection(
        struct PtrWrapper ptr, int detectionIdx);

struct PtrWrapper MultiTracker_ctor(
        const char *trackerType);

void MultiTracker_dtor(
        struct PtrWrapper ptr);

bool MultiTracker_add(
        struct PtrWrapper ptr, struct TensorWrapper image,
        struct Rect2dWrapper boundingBox);

bool MultiTracker_add2(
        struct PtrWrapper ptr, const char *trackerType,
        struct TensorWrapper image, struct Rect2dWrapper boundingBox);

bool MultiTracker_add3(
        struct PtrWrapper ptr, const char *trackerType,
        struct TensorWrapper image, struct Rect2dArray boundingBox);

bool MultiTracker_add4(
        struct PtrWrapper ptr, struct TensorWrapper image,
        struct Rect2dArray boundingBox);

bool MultiTracker_update(
        struct PtrWrapper ptr, struct TensorWrapper image);

struct Rect2dArrayPlusBool {
    struct Rect2dArray rect2d;
    bool val;
};

struct Rect2dArrayPlusBool MultiTracker_update2(
        struct MultiTrackerPtr ptr, struct TensorWrapper image);
]]

function cv.test()
    C.test()
end

do
    local WeakClassifierHaarFeature = torch.class('cv.WeakClassifierHaarFeature', cv)

    function WeakClassifierHaarFeature:__init()
        self.ptr = ffi.gc(C.WeakClassifierHaarFeature_ctor(), C.WeakClassifierHaarFeature_dtor)
    end

    function WeakClassifierHaarFeature:eval(t)
        local argRules = {
            {"value", required = true} }
        local value = cv.argcheck(t, argRules)
        return C.WeakClassifierHaarFeature_eval(self.ptr, value)
    end

    function WeakClassifierHaarFeature:update(t)
        local argRules = {
            {"value", required = true},
            {"target", required = true} }
        local value, target = cv.argcheck(t, argRules)
        return C.WeakClassifierHaarFeature_update(self.ptr, value, target)
    end
end

--BaseClassifier

do
    local BaseClassifier = torch.class('cv.BaseClassifier', cv)

    function BaseClassifier:__init(t)
        local argRules = {
            {"numWeakClassifier", required = true},
            {"iterationInit", required = true} }
        local numWeakClassifier, iterationInit = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.BaseClassifier_ctor(numWeakClassifier, iterationInit), C.BaseClassifier_dtor)
    end

    function BaseClassifier:computeReplaceWeakestClassifier(t)
        local argRules = {
            {"errors", required = true} }
        local errors = cv.argcheck(t, argRules)
        return C.BaseClassifier_computeReplaceWeakestClassifier(self.ptr, errors)
    end

    function BaseClassifier:eval(t)
        local argRules = {
            {"image", required = true} }
        local image = cv.argcheck(t, argRules)
        return C.BaseClassifier_eval(cv.wrap_tensor(self.ptr, image))
    end

    function BaseClassifier:getError(t)
        local argRules = {
            {"curWeakClassifier", required = true} }
        local curWeakClassifier = cv.argcheck(t, argRules)
        return C.BaseClassifier_getError(self.ptr, curWeakClassifier)
    end

    function BaseClassifier:getErrors(t)
        local argRules = {
            {"errors", required = true} }
        local errors = cv.argcheckt(t, argRules)
        return ffi.gcarray(C.BaseClassifier_getErrors(self.ptr, errors))
    end

    function BaseClassifier:etIdxOfNewWeakClassifier()
        return C.BaseClassifier_getIdxOfNewWeakClassifier(self.ptr)
    end

    function BaseClassifier:getSelectedClassifier()
        return C.BaseClassifier_getSelectedClassifier(self.ptr)
    end

    function BaseClassifier:replaceClassifierStatistic(t)
        local argRules = {
            {"sourceIndex", required = true},
            {"targetIndex", required = true } }
        local sourceIndex, targetIndex = cv.argcheck(t, argRules)
        C.BaseClassifier_replaceClassifierStatistic(self.ptr, sourceIndex, targetIndex)
    end

    function BaseClassifier:replaceWeakClassifier(t)
        local argRules = {
            {"index", required = true} }
        local index = cv.argcheck(t, argRules)
        C.BaseClassifier_replaceWeakClassifier(self.ptr)
    end

    function BaseClassifier:selectBestClassifier(t)
        local argRules = {
            {"errorMask", required = true},
            {"importance", required = true},
            {"errors", required = true} }
        local errorMask, importance, errors = cv.argcheck(t, argRules)
        local result = C.BaseClassifier_selectBestClassifier(self.ptr, errorMask, importance, errors)
        return result.val, cv.gcarray(result.array)
    end

    function BaseClassifier:trainClassifier(t)
        local argRules = {
            {"image", required = true},
            {"target", required = true},
            {"importance", required = true},
            {"errorMask", required = true} }
        local image, target, importance, errorMask = cv.argcheck(t, argRules)
        return cv.gcarray(
                    C.BaseClassifier_trainClassifier(
                        self.ptr, cv.wrap_tensor(image), target, importance, errorMask))
    end
end

--EstimatedGaussDistribution

do
    local EstimatedGaussDistribution = torch.class('cv.EstimatedGaussDistribution', cv)

    function EstimatedGaussDistribution:__init(t)
        local argRules = {
            {"P_mean", required = true},
            {"R_mean", required = true},
            {"P_sigma", required = true},
            {"R_sigma", required = true} }
        local P_mean, R_mean, P_sigma, R_sigma = cv.argcheck(t, argRules)
        if P_mean then
            self.ptr = ffi.gc(
                C.EstimatedGaussDistribution_ctor2(P_mean, R_mean, P_sigma, R_sigma),
                C.EstimatedGaussDistribution_dtor)
        else
            self.ptr = ffi.gc(
                C.EstimatedGaussDistribution_ctor(),
                C.EstimatedGaussDistribution_dtor)
        end
    end

    function EstimatedGaussDistribution:getMean()
        return C.EstimatedGaussDistribution_getMean(self.ptr)
    end

    function EstimatedGaussDistribution:getSigma()
        return C.EstimatedGaussDistribution_getSigma(self.ptr)
    end

    function EstimatedGaussDistribution:setValues(t)
        local argRules = {
            {"mean", required = true},
            {"sigma", required = true} }
        local mean, sigma = cv.argcheck(t, argRules)
        C.EstimatedGaussDistribution_setValues(self.ptr, mean, sigma)
    end
end

--ClassifierThreshold

do
    local ClassifierThreshold = torch.class('cv.ClassifierThreshold', cv)

    function ClassifierThreshold:__init(t)
        local argRules = {
            {"posSamples", required = true},
            {"negSamples", required = true} }
        local posSamples, negSamples = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(
                    C.ClassifierThreshold_ctor(posSamples.ptr, negSamples.ptr),
                    C.ClassifierThreshold_dtor)
    end

    function ClassifierThreshold:eval(t)
        local argRules = {
            {"value", required = true} }
        local value = cv.argcheck(t, argRules)
        return C.ClassifierThreshold_eval(self.ptr, value)
    end

    function ClassifierThreshold:update(t)
        local argRules = {
            {"value", required = true},
            {"target", required = true} }
        local value, target = cv.argcheck(t, argRules)
        C.ClassifierThreshold_update(self.ptr, value, target)
    end
end

--ClfMilBoost

do
    local ClfMilBoost = torch.class('cv.ClfMilBoost', cv)

    function ClfMilBoost:__init()
        self.ptr = ffi.gc(C.ClfMilBoost_ctor(), C.ClfMilBoost_dtor)
    end

    function ClfMilBoost:classify(t)
        local argRules = {
            {"x", required = true},
            {"logR", default = true} }
        local x, logR = cv.argcheck(t, argRules)
        return cv.gcarray(C.ClfMilBoost_classify(self.ptr, cv.wrap_tensor(x), logR))
    end

    function ClfMilBoost:init(t)
        local argRules = {
            {"parameters", default = ffi.new('struct Params')} }
        local parameters = cv.argcheck(t, argRules)
        C.ClfMilBoost_init(self.ptr, parameters)
    end

    function ClfMilBoost:sigmoid(t)
        local argRules = {
            {"x", required = true} }
        local x = cv.argcheck(t, argRules)
        return C.ClfMilBoost_sigmoid(self.ptr, x)
    end

    function ClfMilBoost:update(t)
        local argRules = {
            {"posx", required = true},
            {"negx", required = true} }
        local posx, negx = cv.argcheck(t, argRules)
        C.ClfMilBoost_update(self.ptr, cv.wrap_tensor(posx), cv.wrap_tensor(negx))
    end
end

--ClfOnlineStump

do
    local ClfOnlineStump = torch.class('cv.ClfOnlineStump', cv)

    function ClfOnlineStump:__init()
        self.ptr = ffi.gc(C.ClfOnlineStump_ctor(), C.ClfOnlineStump_dtor)
    end

    function ClfOnlineStump:classify(t)
        local argRules = {
            {"x", required = true},
            {"i", required = true} }
        local x, i = cv.argcheck(t, argRules)
        return C.ClfOnlineStump_classify(self.ptr, cv.wrap_tensor(x), i)
    end

    function ClfOnlineStump:classifyF(t)
        local argRules = {
            {"x", required = true},
            {"i", required = true} }
        local x, i = cv.argcheck(t, argRules)
        return C.ClfOnlineStump_classifyF(self.ptr, cv.wrap_tensor(x), i)
    end

    function ClfOnlineStump:classifySetF(t)
        local argRules = {
            {"x", required = true} }
        local x = cv.argcheck(t, argRules)
        return cv.gcarray(
                    C.ClfOnlineStump_classifySetF(cv.wrap_tensor(x)))
    end

    function ClfOnlineStump:init()
        C.ClfOnlineStump_init(self.ptr)
    end

    function ClfOnlineStump:update(t)
        local argRules = {
            {"posx", required = true},
            {"negx", required = true} }
        local posx, negx = cv.argcheck(t, argRules)
        C.ClfOnlineStump_update(self.ptr, cv.wrap_tensor(posx), cv.wrap_tensor(negx))
    end
end

--CvParams
do
    local CvParams = torch.class('cv.CvParams', cv)

    function CvParams:printAttrs()
        C.CvParams_printAttrs(self.ptr)
    end

    function CvParams:printDefaults()
        C.CvParams_printDefaults(self.ptr)
    end

    function CvParams:read(t)
        local argRules = {
            {"node", required = true} }
        local node = cv.argcheck(t, argRules)
        return C.CvParams_read(self.ptr, node.ptr)
    end

    function CvParams:scanAttr(t)
        local argRules = {
            {"prmName", required = true},
            {"val", required = true} }
        local prmName, val = cv.argcheck(t, argRules)
        return C.CvParams_scanAttr(self.ptr, prmName, val)
    end

    function CvParams:write(t)
        local argRules = {
            {"fs", required = true} }
        local fs = cv.argcheck(t, argRules)
        C.CvParams_write(self.ptr, fs.ptr)
    end
end

--CvFeatureParams

do
    local CvFeatureParams = torch.class('cv.CvFeatureParams', 'cv.CvParams', cv)

    function CvFeatureParams:__init(t)
        local argRules = {
            {"featureType", required = true} }
        local featureType = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.CvFeatureParams_ctor(featureType), C.CvParams_dtor)
    end

    function CvFeatureParams:init(t)
        local argRules = {
            {"fp", required = true} }
        local fp = cv.argcheck(t, argRules)
        C.CvFeatureParams_init(self.ptr, fp.ptr)
    end

    function CvFeatureParams:read(t)
        local argRules = {
            {"node", required = true} }
        local node = cv.argcheck(t, argRules)
        return C.CvFeatureParams_read(self.ptr, node.ptr)
    end

    function CvFeatureParams:write(t)
        local argRules = {
            {"fs", required = true} }
        local fs = cv.argcheck(t, argRules)
        C.CvFeatureParams_write(self.ptr, fs.ptr)
    end
end

--CvHaarFeatureParams

do
    local CvHaarFeatureParams = torch.class('cv.CvHaarFeatureParams', 'cv.CvFeatureParams', cv)

    function CvHaarFeatureParams:__init()
        self.ptr = ffi.gc(C.CvHaarFeatureParams_ctor(), C.CvParams_dtor)
    end

    function CvHaarFeatureParams:init(t)
        local argRules = {
            {"fp", required = true} }
        local fp = cv.argcheck(t, argRules)
        C.CvHaarFeatureParams_init(self.ptr, fp.ptr)
    end

    function CvHaarFeatureParams:printAttrs()
        C.CvHaarFeatureParams_printAttrs(self.ptr)
    end

    function CvHaarFeatureParams:printDefaults()
        C.CvHaarFeatureParams_printDefaults(self.ptr)
    end

    function CvHaarFeatureParams:read(t)
        local argRules = {
            {"node", required = true} }
        local node = cv.argcheck(t, argRules)
        return C.CvHaarFeatureParams_read(self.ptr, node.ptr)
    end

    function CvHaarFeatureParams:scanAttr(t)
        local argRules = {
            {"prm", required = true},
            {"val", required = true} }
        local prm, val = cv.argcheck(t, argRules)
        return C.CvHaarFeatureParams_scanAttr(self.ptr, prm, val)
    end

    function CvHaarFeatureParams:write(t)
        local argRules = {
            {"fs", required = true} }
        local fs = cv.argcheck(t, argRules)
        C.CvHaarFeatureParams_write(self.ptr, fs.ptr)
    end
end

--CvHOGFeatureParams

do
    local CvHOGFeatureParams = torch.class('cv.CvHOGFeatureParams', 'cv.CvFeatureParams', cv)

    function CvHOGFeatureParams:__init()
        self.ptr = ffi.gc(C.CvHOGFeatureParams_ctor(), C.CvParams_dtor)
    end
end

--CvLBPFeatureParams

do
    local CvLBPFeatureParams = torch.class('cv.CvLBPFeatureParams', 'cv.CvFeatureParams', cv)

    function CvLBPFeatureParams:__init()
        self.ptr = ffi.gc(C.CvLBPFeatureParams_ctor(), C.CvParams_dtor)
    end
end

--CvFeatureEvaluator

--FeatureParams, type
cv.HAAR = 0
cv.LBP = 1
cv.HOG = 2

do
    local CvFeatureEvaluator = torch.class('cv.CvFeatureEvaluator', cv)

    function CvFeatureEvaluator:__init(t)
        local argRules = {
            {"type", required = true} }
        local type = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.CvFeatureEvaluator_ctor(type), C.CvFeatureEvaluator_dtor)
    end

    function CvFeatureEvaluator:getCls()
        return cv.unwrap_tensor(
                    C.CvFeatureEvaluator_getCls(self.ptr))
    end

    function CvFeatureEvaluator:getCls2(t)
        local argRules = {
            {"si", required = true} }
        local si = cv.argcheck(t, argRules)
        return C.CvFeatureEvaluator_getCls(self.ptr, si)
    end

    function CvFeatureEvaluator:getFeatureSize()
        return C.CvFeatureEvaluator_getFeatureSize(self.ptr)
    end

    function CvFeatureEvaluator:getMaxCatCount()
        return C.CvFeatureEvaluator_getMaxCatCount(self.ptr)
    end

    function CvFeatureEvaluator:getNumFeatures()
        return C.CvFeatureEvaluator_getNumFeatures(self.ptr)
    end

    function CvFeatureEvaluator:init(t)
        local argRules = {
            {"_featureParams", required = true},
            {"_maxSampleCount", required = true},
            {"_winSize", required = true, operator = cv.Size} }
        local _featureParams, _maxSampleCount, _winSize = cv.argcheck(t, argRules)
        C.CvFeatureEvaluator_init(self.ptr, _featureParams.ptr, _maxSampleCount, _winSize)
    end

    function CvFeatureEvaluator:__call(t)
        local argRules = {
            {"featureIdx", required = true},
            {"sampleIdx", required = true} }
        local featureIdx, sampleIdx = cv.argcheck(t, argRules)
        return C.CvFeatureEvaluator_call(self.ptr, featureIdx, sampleIdx)
    end

    function CvFeatureEvaluator:setImage(t)
        local argRules = {
            {"img", required = true},
            {"clsLabel", required = true},
            {"idx", required = true} }
        local img, clsLabel, idx = cv.argcheck(t, argRules)
        C.CvFeatureEvaluator_setImage(self.ptr, cv.wrap_tensor(img), clsLabel, idx)
    end

    function CvFeatureEvaluator:writeFeatures(t)
        local argRules = {
            {"fs", required = true},
            {"featureMap", required = true} }
        local fs, featureMap = cv.argcheck(t, argRules)
        C.CvFeatureEvaluator_writeFeatures(self.ptr, fs.ptr, cv.wrap_tensor(featureMap))
    end
end

--CvHaarEvaluator

do
    local CvHaarEvaluator = torch.class('cv.CvHaarEvaluator', 'cv.CvFeatureEvaluator', cv)

    function CvHaarEvaluator:__init()
        self.ptr = ffi.gc(C.CvHaarEvaluator_ctor(), C.CvHaarEvaluator_dtor)
    end

    function CvHaarEvaluator:generateFeatures()
        C.CvHaarEvaluator_generateFeatures(self.ptr)
    end

    function CvHaarEvaluator:generateFeatures2(t)
        local argRules = {
            {"numFeatures", required = true } }
        local numFeatures = cv.argcheck(t, argRules)
        C.CvHaarEvaluator_generateFeatures(self.ptr, numFeatures)
    end

    function CvHaarEvaluator:init(t)
        local argRules = {
            {"_featureParams", required = true},
            {"_maxSampleCount", required = true},
            {"_winSize", required = true, operator = cv.Size} }
        local _featureParams, _maxSampleCount, _winSize = cv.argcheck(t, argRules)
        C.CvHaarEvaluator_init(self.ptr, _featureParams.ptr, _maxSampleCount, _winSize)
    end

    function CvHaarEvaluator:__call(t)
        local argRules = {
            {"featureIdx", required = true},
            {"sampleIdx", required = true} }
        local featureIdx, sampleIdx = cv.argcheck(t, argRules)
        return C.CvHaarEvaluator_call(self.ptr, featureIdx, sampleIdx)
    end

    function CvHaarEvaluator:setImage(t)
        local argRules = {
            {"img", required = true},
            {"clsLabel", required = true},
            {"idx", required = true} }
        local img, clsLabel, idx = cv.argcheck(t, argRules)
        C.CvHaarEvaluator_setImage(self.ptr, cv.wrap_tensor(img), clsLabel, idx)
    end

    function CvHaarEvaluator:setWinSize(t)
        local argRules = {
            {"patchSize", required = true, operator = cv.Size} }
        local patchSize = cv.argcheck(t, argRules)
        C.CvHaarEvaluator_setWinSize(self.ptr, patchSize)
    end

    function CvHaarEvaluator:setWinSize2()
        return C.CvHaarEvaluator_setWinSize(self.ptr)
    end

    function CvHaarEvaluator:writeFeature(t)
        local argRules = {
            {"fs", required = true} }
        local fs = cv.argcheck(t, argRules)
        C.CvHaarEvaluator_writeFeature(self.ptr, fs.ptr)
    end

    function CvHaarEvaluator:writeFeatures(t)
        local argRules = {
            {"fs", required = true},
            {"featureMap", required = true} }
        local fs, featureMap = cv.argcheck(t, argRules)
        C.CvHaarEvaluator_writeFeatures(self.ptr, fs.ptr, cv.wrap_tensor(featureMap))
    end
end

--CvHOGEvaluator

do
    local CvHOGEvaluator = torch.class('cv.CvHOGEvaluator', 'cv.CvFeatureEvaluator', cv)

    function CvHOGEvaluator:__init(t)
        self.ptr = ffi.gc(C.CvHOGEvaluator_ctor(), C.CvHOGEvaluator_dtor)
    end

    function CvHOGEvaluator:init(t)
        local argRules = {
            {"_featureParams", required = true},
            {"_maxSampleCount", required = true},
            {"_winSize", required = true, operator = cv.Size} }
        local _featureParams, _maxSampleCount, _winSize = cv.argcheck(t, argRules)
        C.CvHOGEvaluator_init(self.ptr, _featureParams.ptr, _maxSampleCount, _winSize)
    end

    function CvHOGEvaluator:__call(t)
        local argRules = {
            {"featureIdx", required = true},
            {"sampleIdx", required = true} }
        local featureIdx, sampleIdx = cv.argcheck(t, argRules)
        return C.CvHOGEvaluator_call(self.ptr, featureIdx, sampleIdx)
    end

    function CvHOGEvaluator:setImage(t)
        local argRules = {
            {"img", required = true},
            {"clsLabel", required = true},
            {"idx", required = true} }
        local img, clsLabel, idx = cv.argcheck(t, argRules)
        C.CvHOGEvaluator_setImage(self.ptr, cv.wrap_tensor(img), clsLabel, idx)
    end

    function CvHOGEvaluator:writeFeatures(t)
        local argRules = {
            {"fs", required = true},
            {"featureMap", required = true} }
        local fs, featureMap = cv.argcheck(t, argRules)
        C.CvHOGEvaluator_writeFeatures(self.ptr, fs.ptr, cv.wrap_tensor(featureMap))
    end
end

--CvLBPEvaluator

do
    local CvLBPEvaluator = torch.class('cv.CvLBPEvaluator', 'cv.CvFeatureEvaluator', cv)

    function CvLBPEvaluator:__init(t)
        self.ptr = ffi.gc(C.CvLBPEvaluator_ctor(), C.CvLBPEvaluator_dtor)
    end

    function CvLBPEvaluator:init(t)
        local argRules = {
            {"_featureParams", required = true},
            {"_maxSampleCount", required = true},
            {"_winSize", required = true, operator = cv.Size} }
        local _featureParams, _maxSampleCount, _winSize = cv.argcheck(t, argRules)
        C.CvLBPEvaluator_init(self.ptr, _featureParams.ptr, _maxSampleCount, _winSize)
    end

    function CvLBPEvaluator:__call(t)
        local argRules = {
            {"featureIdx", required = true},
            {"sampleIdx", required = true} }
        local featureIdx, sampleIdx = cv.argcheck(t, argRules)
        return C.CvLBPEvaluator_call(self.ptr, featureIdx, sampleIdx)
    end

    function CvLBPEvaluator:setImage(t)
        local argRules = {
            {"img", required = true},
            {"clsLabel", required = true},
            {"idx", required = true} }
        local img, clsLabel, idx = cv.argcheck(t, argRules)
        C.CvLBPEvaluator_setImage(self.ptr, cv.wrap_tensor(img), clsLabel, idx)
    end

    function CvLBPEvaluator:writeFeatures(t)
        local argRules = {
            {"fs", required = true},
            {"featureMap", required = true} }
        local fs, featureMap = cv.argcheck(t, argRules)
        C.CvLBPEvaluator_writeFeatures(self.ptr, fs.ptr, cv.wrap_tensor(featureMap))
    end
end

--StrongClassifierDirectSelection

do
    local StrongClassifierDirectSelection = torch.class('cv.StrongClassifierDirectSelection', cv)

    function StrongClassifierDirectSelection:__init(t)
        local argRules = {
            {"numBaseClf", required = true},
            {"numWeakClf", required = true},
            {"patchSz", required = true, operator = cv.Size},
            {"sampleROI", required = true, operator = cv.Rect},
            {"useFeatureEx", default = false},
            {"iterationInit", default = 0} }
        local numBaseClf, numWeakClf, patchSz, sampleROI,
              useFeatureEx, iterationInit = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(
                        C.StrongClassifierDirectSelection_ctor(
                                numBaseClf, numWeakClf, patchSz, sampleROI,
                                useFeatureEx, iterationInit),
                        C.StrongClassifierDirectSelection_dtor)
    end

    function StrongClassifierDirectSelection:classifySmooth(t)
        local argRules = {
            {"images", required = true},
            {"sampleROI", required = true} }
        local images, sampleROI = cv.argcheck(t, argRules)
        local result = C.StrongClassifierDirectSelection_classifySmooth(
                                self.ptr, cv.wrap_tensors(images), sampleROI)
        return result.v1, result.v2
    end

    function StrongClassifierDirectSelection:eval(t)
        local argRules = {
            {"response", required = true} }
        local response = cv.argcheck(t, argRules)
        return C.StrongClassifierDirectSelection_eval(self.ptr, cv.wrap_tensor(response))
    end

    function StrongClassifierDirectSelection:getNumBaseClassifier()
        return C.StrongClassifierDirectSelection_getNumBaseClassifier(self.ptr)
    end

    function StrongClassifierDirectSelection:getPatchSize()
        return C.StrongClassifierDirectSelection_getPatchSize(self.ptr)
    end

    function StrongClassifierDirectSelection:getReplacedClassifier()
        return C.StrongClassifierDirectSelection_getReplacedClassifier(self.ptr)
    end

    function StrongClassifierDirectSelection:getROI()
        return C.StrongClassifierDirectSelection_getROI(self.ptr)
    end

    function StrongClassifierDirectSelection:getSelectedWeakClassifier()
        return cv.gcarray(
                    C.StrongClassifierDirectSelection_getSelectedWeakClassifier(self.ptr))
    end

    function StrongClassifierDirectSelection:getSwappedClassifier()
        return C.StrongClassifierDirectSelection_getSwappedClassifier(self.ptr)
    end

    function StrongClassifierDirectSelection:getUseFeatureExchange()
        return C.StrongClassifierDirectSelection_getUseFeatureExchange(self.ptr)
    end

    function StrongClassifierDirectSelection:initBaseClassifier()
        C.StrongClassifierDirectSelection_initBaseClassifier(self.ptr)
    end

    function StrongClassifierDirectSelection:replaceWeakClassifier(t)
        local argRules = {
            {"idx", requierd = true} }
        local idx = cv.argcheck(t, argRules)
        C.StrongClassifierDirectSelection_replaceWeakClassifier(self.ptr, idx)
    end

    function StrongClassifierDirectSelection:update(t)
        local argRules = {
            {"image", required = true},
            {"target", required = true},
            {"importance", default = 1} }
        local image, target, importance = cv.argcheck(t, argRules)
        return C.StrongClassifierDirectSelection_update(self.ptr, cv.wrap_tensor(image), target, importance)
    end
end

--Detector

do
    local Detector = torch.class('cv.Detector', cv)

    function Detector:__init(t)
        local argRules = {
            {"classifier", required = true} }
        local classifier = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.Detector_ctor(classifier.ptr), C.Detector_dtor)
    end

    function Detector:classifySmooth(t)
        local argRules = {
            {"image", required = true},
            {"minMargin", default = 0} }
        local image, minMargin = cv.argcheck(t, argRules)
        C.Detector_classifySmooth(self.ptr, cv.wrap_tensors(image), minMargin)
    end

    function Detector:getConfidence(t)
        local argRules = {
            {"patchIdx", required = true} }
        local patchIdx = cv.argcheck(t, argRules)
        return C.Detector_getConfidence(self.ptr, patchIdx)
    end

    function Detector:getConfidenceOfBestDetection()
        return C.Detector_getConfidenceOfBestDetection(self.ptr)
    end

    function Detector:getConfidenceOfDetection(t)
        local argRules = {
            {"detectionIdx", required = true} }
        local detectionIdx = cv.argcheck(t, argRules)
        return C.Detector:getConfidenceOfDetection(self.ptr, detectionIdx)
    end

    function Detector:getConfidences()
        return cv.gcarray(C.Detector_getConfidences(self.ptr))
    end

    function Detector:getConfImageDisplay()
        return cv.unwrap_tensors(
                    C.Detector_getConfImageDisplay(self.ptr))
    end

    function Detector:getIdxDetections()
        return cv.gcarray(C.Detector_getIdxDetections(self.ptr))
    end

    function Detector:getNumDetections()
        return C.Detector_getNumDetections(self.ptr)
    end

    function Detector:getPatchIdxOfDetection(t)
        local argRules = {
            {"detectionIdx", required = true} }
        local detectionIdx = cv.argcheck(t, argRules)
        return C.Detector_getPatchIdxOfDetection(self.ptr, detectionIdx)
    end
end

--MultiTracker

do
    local MultiTracker = torch.class('cv.MultiTracker', cv)

    function MultiTracker:__init(t)
        local argRules = {
            {"trackerType", default = ""} }
        local trackerType = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.MultiTracker_ctor(trackerType), C.MultiTracker_dtor)
    end

    function MultiTracker:add(t)
        local argRules = {
            {"image", required = true},
            {"boundingBox", required = true, operator = cv.Rect2d} }
        local image, boundingBox = cv.argcheck(t, argRules)
        return C.MultiTracker_add(self.ptr, cv.wrap_tensor(image), boundingBox)
    end

    function MultiTracker:add2(t)
        local argRules = {
            {"trackerType", required = true},
            {"image", required = true},
            {"boundingBox", required = true, operator = cv.Rect2d} }
        local trackerType, image, boundingBox = cv.argcheck(t, argRules)
        return C.MultiTracker_add2(self.ptr, trackerType, cv.wrap_tensor(image), boundingBox)
    end

    function MultiTracker:add3(t)
        local argRules = {
            {"trackerType", required = true},
            {"image", required = true},
            {"boundingBox", required = true} }
        local trackerType, image, boundingBox = cv.argcheck(t, argRules)
        return C.MultiTracker_add3(self.ptr, trackerType, cv.wrap_tensor(image), boundingBox)
    end

    function MultiTracker:add4(t)
        local argRules = {
            {"image", required = true},
            {"boundingBox", required = true} }
        local image, boundingBox = cv.argcheck(t, argRules)
        return C.MultiTracker_add4(self.ptr, cv.wrap_tensor(image), boundingBox)
    end

    function MultiTracker:update(t)
        local argRules = {
            {"image", required = true} }
        local image = cv.argcheck(t, argRules)
        return C.MultiTracker_update(self.ptr, cv.wrap_tensor(image))
    end

    function MultiTracker:update2(t)
        local argRules = {
            {"image", required = true} }
        local image = cv.argcheck(t, argRules)
        local result = C.MultiTracker_update2(self.ptr, cv.wrap_tensor(image))
        return result.val, result.rect2d
    end
end