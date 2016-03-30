local cv = require 'cv._env'
local ffi = require 'ffi'
require 'cv.Classes'

local Classes = ffi.load(cv.libPath('Classes'))
local C = ffi.load(cv.libPath('tracking'))

--- ***************** Classes *****************

ffi.cdef[[
struct ConfidenceMapArray test(
    struct ConfidenceMapArray val);

struct ConfidenceMap {
    struct ClassArray class_array;
    struct FloatArray float_array;
    int size;
};

struct ConfidenceMapArray {
    struct ConfidenceMap *array;
    int size;
};

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

struct PtrWrapper FeatureHaar_ctor(
        struct SizeWrapper patchSize);

void FeatureHaar_dtor(
        struct PtrWrapper ptr);

struct FloatPlusBool {
    bool v1;
    float v2;
};

struct FloatPlusBool FeatureHaar_eval(
        struct PtrWrapper ptr, struct TensorWrapper image, struct RectWrapper ROI);

struct RectArray FeatureHaar_getAreas(
    struct PtrWrapper ptr);

float FeatureHaar_getInitMean(
        struct PtrWrapper ptr);

float FeatureHaar_getInitSigma(
        struct PtrWrapper ptr);

int FeatureHaar_getNumAreas(
        struct PtrWrapper ptr);

struct FloatArray FeatureHaar_getWeights(
        struct PtrWrapper ptr);

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
        struct PtrWrapper ptr, struct TensorWrapper image);

struct PtrWrapper MultiTracker_Alt_ctor();

void MultiTracker_Alt_dtor(
        struct PtrWrapper ptr);

bool MultiTracker_Alt_addTarget(
        struct PtrWrapper ptr, struct TensorWrapper image,
        struct Rect2dWrapper boundingBox, const char *tracker_algorithm_name);

bool MultiTracker_Alt_update(
        struct PtrWrapper ptr, struct TensorWrapper image);

struct PtrWrapper MultiTrackerTLD_ctor();

void MultiTrackerTLD_dtor(
        struct PtrWrapper ptr);

bool MultiTrackerTLD_update_opt(
        struct PtrWrapper ptr, struct TensorWrapper image);

struct PtrWrapper ROISelector_ctor();

void ROISelector_dtor(
        struct PtrWrapper ptr);

struct Rect2dWrapper ROISelector_select(
        struct PtrWrapper ptr, struct TensorWrapper image, bool fromCenter);

struct Rect2dWrapper ROISelector_select2(
        struct PtrWrapper ptr, const char *windowName,
        struct TensorWrapper img, bool showCrossair, bool fromCenter);

void ROISelector_select3(
        struct PtrWrapper ptr, const char *windowName, struct TensorWrapper img,
        struct Rect2dArray boundingBox, bool fromCenter);

struct PtrWrapper TrackerTargetState_ctor();

void TrackerTargetState_dtor(
        struct PtrWrapper ptr);

int TrackerTargetState_getTargetHeight(
        struct PtrWrapper ptr);

struct Point2fWrapper TrackerTargetState_getTargetPosition(
        struct PtrWrapper ptr);

int TrackerTargetState_getTargetWidth(
        struct PtrWrapper ptr);

void TrackerTargetState_setTargetHeight(
        struct PtrWrapper ptr, int height);

void TrackerTargetState_setTargetPosition(
        struct PtrWrapper ptr, struct Point2fWrapper position);

void TrackerTargetState_setTargetWidth(
        struct PtrWrapper ptr, int width);

struct PtrWrapper TrackerStateEstimator_ctor(
        const char *trackeStateEstimatorType);

void TrackerStateEstimator_dtor(
        struct PtrWrapper ptr);

struct PtrWrapper TrackerStateEstimator_estimate(
        struct PtrWrapper ptr, struct ConfidenceMapArray confidenceMaps);

const char* TrackerStateEstimator_getClassName(
        struct PtrWrapper ptr);

void TrackerStateEstimator_update(
        struct PtrWrapper ptr, struct ConfidenceMapArray confidenceMaps);

struct PtrWrapper TrackerStateEstimatorAdaBoosting_ctor(
        int numClassifer, int initIterations, int nFeatures,
        struct SizeWrapper patchSize, struct RectWrapper ROI);

void TrackerStateEstimatorAdaBoosting_dtor(
        struct PtrWrapper ptr);

struct IntArray TrackerStateEstimatorAdaBoosting_computeReplacedClassifier(
        struct PtrWrapper ptr);

struct IntArray TrackerStateEstimatorAdaBoosting_computeSelectedWeakClassifier(
        struct PtrWrapper ptr);

struct IntArray TrackerStateEstimatorAdaBoosting_computeSwappedClassifier(
        struct PtrWrapper ptr);

void TrackerStateEstimatorAdaBoosting_setCurrentConfidenceMap(
        struct PtrWrapper ptr, struct ConfidenceMap confidenceMap);

struct RectWrapper TrackerStateEstimatorAdaBoosting_getSampleROI(
        struct PtrWrapper ptr);

void TrackerStateEstimatorAdaBoosting_setSampleROI(
        struct PtrWrapper ptr, struct RectWrapper ROI);

struct PtrWrapper TrackerStateEstimatorMILBoosting_ctor(
        int nFeatures);

void TrackerStateEstimatorMILBoosting_dtor(
        struct PtrWrapper ptr);

void TrackerStateEstimatorMILBoosting_setCurrentConfidenceMap(
        struct PtrWrapper ptr, struct ConfidenceMap confidenceMap);

struct PtrWrapper TrackerStateEstimatorSVM_ctor();

void TrackerStateEstimatorSVM_dtor(
        struct PtrWrapper ptr);

void TrackerModel_dtor(
        struct PtrWrapper ptr);

struct ConfidenceMapArray TrackerModel_getConfidenceMaps(
        struct PtrWrapper ptr);

struct ConfidenceMap TrackerModel_getLastConfidenceMap(
        struct PtrWrapper ptr);

struct PtrWrapper TrackerModel_getLastTargetState(
        struct PtrWrapper ptr);

struct PtrWrapper TrackerModel_getTrackerStateEstimator(
        struct PtrWrapper ptr);

void TrackerModel_modelEstimation(
        struct PtrWrapper ptr, struct TensorArray responses);

void TrackerModel_modelUpdate(
        struct PtrWrapper ptr);

bool TrackerModel_runStateEstimator(
        struct PtrWrapper ptr);

void TrackerModel_setLastTargetState(
        struct PtrWrapper ptr, struct PtrWrapper lastTargetState);

bool TrackerModel_setTrackerStateEstimator(
        struct PtrWrapper ptr, struct PtrWrapper trackerStateEstimator);

void Tracker_dtor(
        struct PtrWrapper ptr);

struct PtrWrapper Tracker_getModel(
        struct PtrWrapper ptr);

bool Tracker_init(
        struct PtrWrapper ptr, struct TensorWrapper image, struct Rect2dWrapper boundingBox);

void Tracker_read(
        struct PtrWrapper ptr, struct PtrWrapper fn);

bool Tracker_update(
        struct PtrWrapper ptr, struct TensorWrapper image, struct Rect2dWrapper boundingBox);

void Tracker_write(
        struct PtrWrapper ptr, struct PtrWrapper fn);

struct TrackerBoosting_Params {
    int numClassifiers;
    float samplerOverlap;
    float samplerSearchFactor;
    int iterationInit;
    int featureSetNumFeatures;
};

struct PtrWrapper TrackerBoosting_ctor(
        struct TrackerBoosting_Params parameters);

void TrackerBoosting_dtor(
        struct PtrWrapper ptr);

struct TrackerKCF_Params {
    double sigma;
    double lambda;
    double interp_factor;
    double output_sigma_factor;
    double pca_learning_rate;
    bool resize;
    bool split_coeff;
    bool wrap_kernel;
    bool compress_feature;
    int max_patch_size;
    int compressed_size;
    unsigned int desc_pca;
    unsigned int desc_npca;
};

struct PtrWrapper TrackerKCF_ctor(
        struct TrackerKCF_Params parameters);

void TrackerKCF_dtor(
        struct PtrWrapper ptr);

struct TrackerMedianFlow_Params {
    int pointsInGrid;
};

struct PtrWrapper TrackerMedianFlow_ctor(
        struct TrackerMedianFlow_Params parameters);

void TrackerMedianFlow_dtor(
        struct PtrWrapper ptr);

struct TrackerMIL_Params {
    float samplerInitInRadius;
    int samplerInitMaxNegNum;
    float samplerSearchWinSize;
    float samplerTrackInRadius;
    int samplerTrackMaxPosNum;
    int samplerTrackMaxNegNum;
    int featureSetNumFeatures;
};

struct PtrWrapper TrackerMIL_ctor(
        struct TrackerMIL_Params parameters);

void TrackerMIL_dtor(
        struct PtrWrapper ptr);

struct PtrWrapper TrackerTLD_ctor();

void TrackerTLD_dtor(
        struct PtrWrapper ptr);

struct PtrWrapper TrackerAdaBoostingTargetState_ctor(
        struct Point2fWrapper position, int width, int height,
        bool foreground, struct TensorWrapper responses);

void TrackerAdaBoostingTargetState_dtor(
        struct PtrWrapper ptr);

struct TensorWrapper TrackerAdaBoostingTargetState_getTargetResponses(
        struct PtrWrapper ptr);

bool TrackerAdaBoostingTargetState_isTargetFg(
        struct PtrWrapper ptr);

void TrackerAdaBoostingTargetState_setTargetFg(
        struct PtrWrapper ptr, bool foreground);

void TrackerAdaBoostingTargetState_setTargetResponses(
        struct PtrWrapper ptr, struct TensorWrapper responses);

struct PtrWrapper TrackerMILTargetState_ctor(
        struct Point2fWrapper position, int width, int height, bool foreground, struct TensorWrapper features);

void TrackerMILTargetState_dtor(
        struct PtrWrapper ptr);

struct TensorWrapper TrackerMILTargetState_getFeatures(
        struct PtrWrapper ptr);

bool TrackerMILTargetState_isTargetFg(
        struct PtrWrapper ptr);

void TrackerMILTargetState_setFeatures(
        struct PtrWrapper ptr, struct TensorWrapper features);

void TrackerMILTargetState_setTargetFg(
        struct PtrWrapper ptr, bool foreground);

struct PtrWrapper TrackerFeature_ctor(
        const char *rackerFeatureType);

void TrackerFeature_dtor(
        struct PtrWrapper ptr);

struct TensorWrapper TrackerFeature_compute(
        struct PtrWrapper ptr, struct TensorArray images);

const char* TrackerFeature_getClassName(
        struct PtrWrapper ptr);

void TrackerFeature_selection(
        struct PtrWrapper ptr, struct TensorWrapper response, int npoints);

struct PtrWrapper TrackerFeatureFeature2d_ctor(
        const char *detectorType, const char *descriptorType);

void TrackerFeatureFeature2d_dtor(
        struct PtrWrapper ptr);

void TrackerFeatureFeature2d_selection(
        struct PtrWrapper ptr, struct TensorWrapper response, int npoints);

struct TrackerFeatureHAAR_Params
{
    int numFeatures;
    struct SizeWrapper rectSize;
    bool isIntegral;
};

struct PtrWrapper TrackerFeatureHAAR_ctor(
        struct TrackerFeatureHAAR_Params parameters);

void TrackerFeatureHAAR_dtor(
        struct PtrWrapper ptr);

struct TensorWrapper TrackerFeatureHAAR_extractSelected(
        struct PtrWrapper ptr, struct IntArray selFeatures,
        struct TensorArray images);

struct PtrWrapper TrackerFeatureHAAR_getFeatureAt(
        struct PtrWrapper ptr, int id);

void TrackerFeatureHAAR_selection(
    struct PtrWrapper ptr, struct TensorWrapper response, int npoints);

bool TrackerFeatureHAAR_swapFeature(
        struct PtrWrapper ptr, int source, int target);

bool TrackerFeatureHAAR_swapFeature2(
        struct PtrWrapper ptr, int id,
        struct PtrWrapper feature);

struct PtrWrapper TrackerFeatureHOG_ctor();

void TrackerFeatureHOG_dtor(
        struct PtrWrapper ptr);

void TrackerFeatureHOG_selection(
        struct PtrWrapper ptr, struct TensorWrapper response, int npoints);

struct PtrWrapper TrackerFeatureLBP_ctor();

void TrackerFeatureLBP_dtor(
        struct PtrWrapper ptr);

void TrackerFeatureLBP_selection(
        struct PtrWrapper ptr, struct TensorWrapper response, int npoints);

struct PtrWrapper TrackerFeatureSet_ctor();

void TrackerFeatureSet_dtor(
        struct PtrWrapper ptr);

bool TrackerFeatureSet_addTrackerFeature(
        struct PtrWrapper ptr, const char *trackerFeatureType);

bool TrackerFeatureSet_addTrackerFeature2(
        struct PtrWrapper ptr, struct PtrWrapper feature);

void TrackerFeatureSet_extraction(
        struct PtrWrapper ptr, struct TensorArray images);

struct TensorArray TrackerFeatureSet_getResponses(
        struct PtrWrapper ptr);
]]

function cv.test(val)
    return cv.unwrap_ConfidenceMapArray(C.test(val))
end

function cv.unwrap_ConfidenceMap(confMap)
    local retval = {}
    retval.size = confMap.size
    retval.float_array = cv.gcarray(confMap.float_array)
    retval.class_array = cv.unwrap_class("TrackerTargetState", confMap.class_array)
    return retval
end

function cv.unwrap_ConfidenceMapArray(confArray)
    local retval_array = {}
    for i = 1,confArray.size do
        retval_array[i] = cv.unwrap_ConfidenceMap(confArray.array[i-1])
    end
    return retval_array
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

--FeatureHaar

do
    local FeatureHaar = torch.class('cv.FeatureHaar', cv)

    function FeatureHaar:__init(t)
        local argRules = {
            {"patchSize", required = true, operator = cv.Size} }
        local patchSize = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.FeatureHaar_ctor(patchSize), C.FeatureHaar_dtor)
    end

    function FeatureHaar:eval(t)
        local argRules = {
            {"image", required = true},
            {"ROI", required = true, operator = cv.Rect} }
        local image, ROI = cv.argcheck(t, argRules)
        local result = C.FeatureHaar_eval(self.ptr, cv.wrap_tensor(image), ROI)
        return result.v1, result.v2
    end

    function FeatureHaar:getAreas()
        return cv.gcarray(C.FeatureHaar_getAreas(self.ptr))
    end

    function FeatureHaar:getInitMean()
        return C.FeatureHaar_getInitMean(self.ptr)
    end

    function FeatureHaar:getInitSigma()
        return C.FeatureHaar_getInitSigma(self.ptr)
    end

    function FeatureHaar:getNumAreas()
        return C.FeatureHaar_getNumAreas(self.ptr)
    end

    function FeatureHaar:getWeights()
        return cv.gcarray(C.FeatureHaar_getWeights(self.ptr))
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

--MultiTracker_Alt

do
    local MultiTracker_Alt = torch.class('cv.MultiTracker_Alt', cv)

    function MultiTracker_Alt:__init()
        self.ptr = ffi.gc(C.MultiTracker_Alt_ctor(), C.MultiTracker_Alt_dtor)
    end

    function MultiTracker_Alt:addTarget(t)
        local argRules = {
            {"image", required = true},
            {"boundingBox", required = true, operator = cv.Rect2d},
            {"tracker_algorithm_name", required = true} }
        local image, boundingBox, tracker_algorithm_name = cv.argcheck(t, argRules)
        return C.MultiTracker_Alt_addTarget(self.ptr, cv.wrap_tensor(image), boundingBox, tracker_algorithm_name)
    end

    function MultiTracker_Alt:update(t)
        local argRules = {
            {"image", required = true} }
        local image = cv.argcheck(t, argRules)
        return C.MultiTracker_Alt_update(self.ptr, cv.wrap_tensor(image))
    end
end

--MultiTrackerTLD

do
    local MultiTrackerTLD = torch.class('cv.MultiTrackerTLD', 'cv.MultiTracker_Alt', cv)

    function MultiTrackerTLD:__init()
        self.ptr = ffi.gc(C.MultiTrackerTLD_ctor(), C.MultiTrackerTLD_dtor)
    end

    function MultiTrackerTLD:update_opt(t)
        local argRules = {
            {"image", required = true} }
        local image = cv.argcheck(t, argRules)
        return C.MultiTrackerTLD_update_opt(self.ptr, cv.wrap_tensor(image))
    end
end

--ROISelector

do
    local ROISelector = torch.class('cv.ROISelector', cv)

    function ROISelector:__init(t)
        self.ptr = ffi.gc(C.ROISelector_ctor(), C.ROISelector_dtor)
    end

    function ROISelector:select(t)
        local argRules = {
            {"img", requierd = true},
            {"fromCenter", default = true} }
        local img, fromCenter = cv.argcheck(t, argRules)
        return C.ROISelector_select(self.ptr, cv.wrap_tensor(img), fromCenter)
    end

    function ROISelector:select2(t)
        local argRules = {
            {"windowName", required = true},
            {"img", required = true},
            {"showCrossair", default = true},
            {"fromCenter", default = true} }
        local windowName, img, showCrossair, fromCenter = cv.argcheck(t, argRules)
        return C.ROISelector_select2(self.ptr, windowName, cv.wrap_tensor(img), showCrossair, fromCenter)
    end

    function ROISelector:select3(t)
        local argRules = {
            {"windowName", required = true},
            {"img", required = true},
            {"boundingBox", required = true},
            {"fromCenter", default = true} }
        local windowName, img, boundingBox, fromCenter = cv.argcheck(t, argRules)
        return C.ROISelector_select3(self.ptr, windowName, cv.wrap_tensor(img), boundingBox, fromCenter)
    end
end

--TrackerTargetState

do
    local TrackerTargetState = torch.class('cv.TrackerTargetState', cv)

    function TrackerTargetState:__init()
        self.ptr = ffi.gc(C.TrackerTargetState_ctor(), C.TrackerTargetState_dtor)
    end

    function TrackerTargetState:getTargetHeight()
        return C.TrackerTargetState_getTargetHeight(self.ptr)
    end

    function TrackerTargetState:getTargetPosition()
        return C.TrackerTargetState_getTargetPosition(self.ptr)
    end

    function TrackerTargetState:getTargetWidth()
        return C.TrackerTargetState_getTargetWidth(self.ptr)
    end

    function TrackerTargetState:setTargetHeight(t)
        local argRules = {
            {"height", required = true} }
        local height = cv.argcheck(t, argRules)
        C.TrackerTargetState_setTargetHeight(self.ptr, height)
    end

    function TrackerTargetState:setTargetPosition(t)
        local argRules = {
            {"position", required = true, operator = cv.Point2f} }
        local position = cv.argcheck(t, argRules)
        C.TrackerTargetState_setTargetPosition(self.ptr, position)
    end

    function TrackerTargetState:setTargetWidth(t)
        local argRules = {
            {"width", required = true} }
        local width = cv.argcheck(t, argRules)
        C.TrackerTargetState_setTargetWidth(self.ptr, width)
    end
end

--TrackerStateEstimator

do
    local TrackerStateEstimator = torch.class('cv.TrackerStateEstimator', cv)

    function TrackerStateEstimator:__init(t)
        local argRules = {
            {"trackeStateEstimatorType", required = true} }
        local trackeStateEstimatorType = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.TrackerStateEstimator_ctor(trackeStateEstimatorType), C.TrackerStateEstimator_dtor)
    end

    function TrackerStateEstimator:estimate(t)
        local argRules = {
            {"confidenceMaps", required = true} }
        local confidenceMaps = cv.argcheck(t, argRules)
        local retval = torch.factory('cv.TrackerTargetState')()
        retval.ptr = ffi.gc(
                        C.TrackerStateEstimator_estimate(self.ptr, confidenceMaps),
                        C.TrackerTargetState_dtor)
        return retval
    end

    function TrackerStateEstimator:getClassName()
        return ffi.string(C.TrackerStateEstimator_getClassName(self.ptr))
    end

    function TrackerStateEstimator:update(t)
        local argRules = {
            {"confidenceMaps", required = true} }
        local confidenceMaps = cv.argcheck(t, argRules)
        C.TrackerStateEstimator_update(self.ptr, confidenceMaps)
    end
end

--TrackerStateEstimatorAdaBoosting

do
    local TrackerStateEstimatorAdaBoosting = torch.class('cv.TrackerStateEstimatorAdaBoosting', 'cv.TrackerStateEstimator', cv)

    function TrackerStateEstimatorAdaBoosting:__init(t)
        local argRules = {
            {"numClassifer", required = true},
            {"initIterations", required = true},
            {"nFeatures", required = true},
            {"patchSize", required = true, operator = cv.Size},
            {"ROI", required = true, operator = cv.Rect} }
        local numClassifer, initIterations, nFeatures, patchSize, ROI = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(
                        C.TrackerStateEstimatorAdaBoosting_ctor(
                            numClassifer, initIterations, nFeatures, patchSize, ROI),
                        C.TrackerStateEstimatorAdaBoosting_dtor)
    end

    function TrackerStateEstimatorAdaBoosting:computeReplacedClassifier()
        return cv.gcarray(C.TrackerStateEstimatorAdaBoosting_computeReplacedClassifier(self.ptr))
    end

    function TrackerStateEstimatorAdaBoosting:computeSelectedWeakClassifier()
        return C.TrackerStateEstimatorAdaBoosting_computeSelectedWeakClassifier(self.ptr)
    end

    function TrackerStateEstimatorAdaBoosting:computeSwappedClassifier()
        return C.TrackerStateEstimatorAdaBoosting_computeSwappedClassifier(self.ptr)
    end

    function TrackerStateEstimatorAdaBoosting:setCurrentConfidenceMap(t)
        local argRules = {
            {"confidenceMap", required = true} }
        local confidenceMap = cv.argcheck(t, argRules)
        C.TrackerStateEstimatorAdaBoosting_setCurrentConfidenceMap(self.ptr, confidenceMap)
    end

    function TrackerStateEstimatorAdaBoosting:getSampleROI()
        return C.TrackerStateEstimatorAdaBoosting_getSampleROI(self.ptr)
    end

    function TrackerStateEstimatorAdaBoosting:setSampleROI(t)
        local argRules = {
            {"ROI", required = true, operator = cv.Rect} }
        local ROI = cv.argcheck(t, argRules)
        C.TrackerStateEstimatorAdaBoosting_setSampleROI(self.ptr, ROI)
    end
end

--TrackerStateEstimatorMILBoosting

do
    local TrackerStateEstimatorMILBoosting = torch.class('cv.TrackerStateEstimatorMILBoosting', 'cv.TrackerStateEstimator', cv)

    function TrackerStateEstimatorMILBoosting:__init(t)
        local argRules = {
            {"nFeatures", required = true} }
        local nFeatures = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.TrackerStateEstimatorMILBoosting_ctor(nFeatures), C.TrackerStateEstimatorMILBoosting_dtor)
    end

    function TrackerStateEstimatorMILBoosting:setCurrentConfidenceMap(t)
        local argRules = {
            {"confidenceMap", required = true} }
        local confidenceMap = cv.argcheck(t, argRules)
        C.TrackerStateEstimatorMILBoosting_setCurrentConfidenceMap(self.ptr, confidenceMap)
    end
end

--TrackerStateEstimatorSVM

do
    local TrackerStateEstimatorSVM = torch.class('cv.TrackerStateEstimatorSVM', 'cv.TrackerStateEstimator', cv)

    function TrackerStateEstimatorSVM:__init(t)
        self.ptr = ffi.gc(C.TrackerStateEstimatorSVM_ctor(), C.TrackerStateEstimatorSVM_dtor)
    end
end

--TrackerModel

do
    local TrackerModel = torch.class('cv.TrackerModel', cv)

    function TrackerModel:getConfidenceMaps()
        return C.TrackerModel_getConfidenceMaps(self.ptr)
    end

    function TrackerModel:getLastConfidenceMap()
        return C.TrackerModel_getLastConfidenceMap(self.ptr)
    end

    function TrackerModel:getLastTargetState()
        local retval = torch.factory('cv.TrackerTargetState')()
        retval.ptr = ffi.gc(C.TrackerModel_getLastTargetState(self.ptr), C.TrackerTargetState_dtor)
        return retval
    end

    function TrackerModel:getTrackerStateEstimator()
        local retval = torch.factory('cv.TrackerStateEstimator')()
        retval.ptr = ffi.gc(C.TrackerModel_getTrackerStateEstimator(self.ptr), C.TrackerStateEstimator_dtor)
        return retval
    end

    function TrackerModel:modelEstimation(t)
        local argRules = {
            {"responses", required = true} }
        local responses = cv.argcheck(t, argRules)
        C.TrackerModel_modelEstimation(self.ptr, cv.wrap_tensors(responses))
    end

    function TrackerModel:modelUpdate()
        C.TrackerModel_modelUpdate(self.ptr)
    end

    function TrackerModel:runStateEstimator()
        return C.TrackerModel_runStateEstimator(self.ptr)
    end

    function TrackerModel:setLastTargetState(t)
        local argRules = {
            {"lastTargetState", required = true} }
        local lastTargetState = cv.argcheck(t, argRules)
        C.TrackerModel_setLastTargetState(self.ptr, lastTargetState.ptr)
    end

    function TrackerModel:setTrackerStateEstimator(t)
        local argRules = {
            {"trackerStateEstimator", required = true} }
        local trackerStateEstimator = cv.argcheck(t, argRules)
        return C.TrackerModel_setTrackerStateEstimator(self.ptr, trackerStateEstimator.ptr)
    end
end

--Tracker

do
    local Tracker = torch.class('cv.Tracker', 'cv.Algorithm', cv)

    function Tracker:getModel()
        local retval = torch.factory('cv.TrackerModel')()
        retval.ptr = ffi.gc(C.Tracker_getModel(self.ptr), C.TrackerModel_dtor)
        return retval
    end

    function Tracker:init(t)
        local argRules = {
            {"image", required = true},
            {"boundingBox", required = true, operator = cv.Rect2d } }
        local image, boundingBox = cv.argcheck(t, argRules)
        return C.Tracker_init(self.ptr, cv.wrap_tensor(image), boundingBox)
    end

    function Tracker:read(t)
        local argRules = {
            {"fn", required = true} }
        local fn = cv.argcheck(t, argRules)
        C.Tracker_read(self.ptr, fn.ptr)
    end

    function Tracker:update(t)
        local argRules = {
            {"image", required = true},
            {"boundingBox", required = true, operator = cv.Rect2d } }
        local image, boundingBox = cv.argcheck(t, argRules)
        return C.Tracker_update(self.ptr, cv.wrap_tensor(image), boundingBox)
    end

    function Tracker:write(t)
        local argRules = {
            {"fs", required = true} }
        local fs = cv.argcheck(t, argRules)
        C.Tracker_write(self.ptr, fs.ptr)
    end
end

--TrackerBoosting

do
    local TrackerBoosting = torch.class('cv.TrackerBoosting', 'cv.Tracker', cv)

    function TrackerBoosting:__init(t)
        local argRules = {
            {"parameters", default = ffi.new('struct TrackerBoosting_Params') } }
        local parameters = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.TrackerBoosting_ctor(parameters), C.TrackerBoosting_dtor)
    end
end

--TrackerKCF

do
    local TrackerKCF = torch.class('cv.TrackerKCF', 'cv.Tracker', cv)

    function TrackerKCF:__init(t)
        local argRules = {
            {"parameters", default = ffi.new('struct TrackerKCF_Params')} }
        local parameters = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.TrackerKCF_ctor(parameters), C.TrackerKCF_dtor)
    end
end

--TrackerMedianFlow

do
    local TrackerMedianFlow = torch.class('cv.TrackerMedianFlow', 'cv.Tracker', cv)

    function TrackerMedianFlow:__init(t)
        local argRules = {
            {"parameters", default = ffi.new('struct TrackerMedianFlow_Params')} }
        local parameters = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.TrackerMedianFlow_ctor(parameters), C.TrackerMedianFlow_dtor)
    end
end

--TrackerMIL

do
    local TrackerMIL = torch.class('cv.TrackerMIL', 'cv.Tracker', cv)

    function TrackerMIL:__init(t)
        local argRules = {
            {"parameters", default = ffi.new('struct TrackerMIL_Params')} }
        local parameters = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.TrackerMIL_ctor(parameters), C.TrackerMIL_dtor)
    end
end

--TrackerTLD

do
    local TrackerTLD = torch.class('cv.TrackerTLD', 'cv.Tracker', cv)

    function TrackerTLD:__init()
        self.ptr = ffi.gc(C.TrackerTLD_ctor(), C.TrackerTLD_dtor)
    end
end

--TrackerAdaBoostingTargetState

do
    local TrackerAdaBoostingTargetState = torch.class('cv.TrackerAdaBoostingTargetState', 'cv.TrackerTargetState', cv)

    function TrackerAdaBoostingTargetState:__init(t)
        local argRules = {
            {"position", required = true, operator = cv.Point2f},
            {"width", required = true},
            {"height", required = true},
            {"foreground", required = true},
            {"responses", required = true} }
        local position, width, height, foreground, responses = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(
            C.TrackerAdaBoostingTargetState_ctor(position, width, height, foreground, cv.wrap_tensor(responses)),
            C.TrackerAdaBoostingTargetState_dtor)
    end

    function TrackerAdaBoostingTargetState:getTargetResponses()
        return cv.unwrap_tensors(C.TrackerAdaBoostingTargetState_getTargetResponses(self.ptr))
    end

    function TrackerAdaBoostingTargetState:isTargetFg()
        return C.TrackerAdaBoostingTargetState_isTargetFg(self.ptr)
    end

    function TrackerAdaBoostingTargetState:setTargetFg(t)
        local argRules = {
            {"foreground", required = true} }
        local foreground = cv.argcheck(t, argRules)
        C.TrackerAdaBoostingTargetState_setTargetFg(self.ptr, foreground)
    end

    function TrackerAdaBoostingTargetState:setTargetResponses(t)
        local argRules = {
            {"responses", required = true} }
        local responses = cv.argcheck(t, argRules)
        C.TrackerAdaBoostingTargetState_setTargetResponses(self.ptr, cv.wrap_tensor(responses))
    end
end

--TrackerMILTargetState

do
    local TrackerMILTargetState = torch.class('cv.TrackerMILTargetState', 'cv.TrackerTargetState', cv)

    function TrackerMILTargetState:__init(t)
        local argRules = {
            {"position", required = true, operator = cv.Point2f},
            {"width", required = true},
            {"height", required = true},
            {"foreground", required = true},
            {"features", required = true} }
        local position, width, height, foreground, features = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(
            C.TrackerMILTargetState_ctor(position, width, height, foreground, cv.wrap_tensor(features)),
            C.TrackerMILTargetState_dtor)
    end

    function TrackerMILTargetState:getFeatures()
        return cv.unwrap_tensors(C.TrackerMILTargetState_getFeatures(self.ptr))
    end

    function TrackerMILTargetState:isTargetFg()
        return C.TrackerMILTargetState_isTargetFg(self.ptr)
    end

    function TrackerMILTargetState:setFeatures(t)
        local argRules = {
            {"features", required = true} }
        local features = cv.argcheck(t, argRules)
        C.TrackerMILTargetState_setFeatures(self.ptr, cv.wrap_tensor(features))
    end

    function TrackerMILTargetState:setTargetFg(t)
        local argRules = {
            {"foreground", required = true} }
        local foreground = cv.argcheck(t, argRules)
        C.TrackerMILTargetState_setTargetFg(self.ptr, foreground)
    end
end

--TrackerFeature

do
    local TrackerFeature = torch.class('cv.TrackerFeature', cv)

    function TrackerFeature:__init(t)
        local argRules = {
            {"trackerFeatureType", required = true} }
        local trackerFeatureType = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.TrackerFeature_ctor(trackerFeatureType), C.TrackerFeature_dtor)
    end

    function TrackerFeature:compute(t)
        local argRules = {
            {"images", required = true} }
        local images = cv.argcheck(t, argRules)
        return cv.unwrap_tensors(C.TrackerFeature_compute(self.ptr, cv.wrap_tensors(images)))
    end

    function TrackerFeature:getClassName()
        return ffi.string(C.TrackerFeature_getClassName())
    end

    function TrackerFeature:selection(t)
        local argRules = {
            {"response", required = true},
            {"npoints", required = true} }
        local response, npoints = cv.argcheck(t, argRules)
        C.TrackerFeature_selection(self.ptr, cv.wrap_tensor(response), npoints)
    end
end

--TrackerFeatureFeature2d

do
    local TrackerFeatureFeature2d = torch.class('cv.TrackerFeatureFeature2d', 'cv.TrackerFeature', cv)

    function TrackerFeatureFeature2d:__init(t)
        local argRules = {
            {"detectorType", required = true},
            {"descriptorType", requied = true} }
        local detectorType, descriptorType = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(
            C.TrackerFeatureFeature2d_ctor(detectorType, descriptorType), C.TrackerFeatureFeature2d_dtor)
    end

    function TrackerFeatureFeature2d:selection(t)
        local argRules = {
            {"response", required = true},
            {"npoints", required = true} }
        local response, npoints = cv.argcheck(t, argRules)
        C.TrackerFeatureFeature2d_selection(self.ptr, cv.wrap_tensor(response), npoints)
    end
end

--

do
    local TrackerFeatureHAAR = torch.class('cv.TrackerFeatureHAAR', 'cv.TrackerFeature', cv)

    function TrackerFeatureHAAR:__init(t)

        local def_parameters = ffi.new("struct TrackerFeatureHAAR_Params")
        def_parameters.numFeatures = 250
        def_parameters.rectSize = cv.Size(100, 100)
        def_parameters.isIntegral = false;

        local argRules = {
            {"parameters", default = def_parameters} }
        local parameters = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.TrackerFeatureHAAR_ctor(parameters), C.TrackerFeatureHAAR_dtor)
    end

    function TrackerFeatureHAAR:extractSelected(t)
        local argRules = {
            {"selFeatures", required = true},
            {"images", required = true} }
        local selFeatures, images = cv.argcheck(t, argRules)
        return cv.unwrap_tensors(C.TrackerFeatureHAAR_extractSelected(self.ptr, selFeatures, cv.wrap_tensors(images)))
    end

    function TrackerFeatureHAAR:getFeatureAt(t)
        local argRules = {
            {"id", required = true} }
        local id = cv.argcheck(t, argRules)
        local retval = torch.factory('cv.FeatureHaar')()
        retval.ptr = ffi.gc(
                        C.TrackerFeatureHAAR_getFeatureAt(self.ptr, id),
                        C.FeatureHaar_dtor)
    end

    function TrackerFeatureHAAR:selection(t)
        local argRules = {
            {"response", required = true},
            {"npoints", required = true} }
        local response, npoints = cv.argcheck(t, argRules)
        C.TrackerFeatureHAAR_selection(self.ptr, cv.wrap_tensor(response), npoints)
    end

    function TrackerFeatureHAAR:swapFeature(t)
        if type(t.target) == "number"  then
            local argRules = {
                {"source", required = true},
                {"target", required = true} }
            local source, target = cv.argcheck(t, argRules)
            return C.TrackerFeatureHAAR_swapFeature(self.ptr, source, target)
        else
            local argRules = {
                {"id", required = true},
                {"feature", required = true} }
            local id, feature = cv.argcheck(t, argRules)
            return C.TrackerFeatureHAAR_swapFeature2(self.ptr, id, feature.ptr)
        end
    end
end

--TrackerFeatureHOG

do
    local TrackerFeatureHOG = torch.class('cv.TrackerFeatureHOG', 'cv.TrackerFeature', cv)

    function TrackerFeatureHOG:__init(t)
        self.ptr = ffi.gc(C.TrackerFeatureHOG_ctor(), C.TrackerFeatureHOG_dtor)
    end

    function TrackerFeatureHOG:selection(t)
        local argRules = {
            {"response", required = true},
            {"npoints", required = true} }
        local response, npoints = cv.argcheck(t, argRules)
        C.TrackerFeatureHOG_selection(self.ptr, cv.wrap_tensor(response), npoints)
    end
end

--TrackerFeatureLBP

do
    local TrackerFeatureLBP = torch.class('cv.TrackerFeatureLBP', 'cv.TrackerFeature', cv)

    function TrackerFeatureLBP:__init(t)
        self.ptr = ffi.gc(C.TrackerFeatureLBP_ctor(), C.TrackerFeatureLBP_dtor)
    end

    function TrackerFeatureLBP:selection(t)
        local argRules = {
            {"response", required = true},
            {"npoints", required = true} }
        local response, npoints = cv.argcheck(t, argRules)
        C.TrackerFeatureLBP_selection(self.ptr, cv.wrap_tensor(response), npoints)
    end
end

--TrackerFeatureSet

do
    local TrackerFeatureSet = torch.class('cv.TrackerFeatureSet', cv)

    function TrackerFeatureSet:__init()
        self.ptr = ffi.gc(C.TrackerFeatureSet_ctor(), C.TrackerFeatureSet_dtor)
    end

    function TrackerFeatureSet:addTrackerFeature(t)
        if type(t.trackerFeatureType) == "string" then
            local argRules = {
                {"trackerFeatureType", required = true} }
            local trackerFeatureType = cv.argcheck(t, argRules)
            return C.TrackerFeatureSet_addTrackerFeature(self.ptr, trackerFeatureType)
        else
            local argRules = {
                {"feature", required = true} }
            local feature = cv.argcheck(t, argRules)
            return C.TrackerFeatureSet_addTrackerFeature2(self.ptr, feature.ptr)
        end
    end

    function TrackerFeatureSet:extraction(t)
        local argRules = {
            {"images", required = true} }
        local images = cv.argcheck(t, argRules)
        C.TrackerFeatureSet_extraction(self.ptr, cv.wrap_tensors(images))
    end

    function TrackerFeatureSet:getResponses()
        return cv.unwrap_tensors(
                    C.TrackerFeatureSet_getResponses(self.ptr))
    end
end