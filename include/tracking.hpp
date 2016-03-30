#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/tracking.hpp>

struct ClassArray {
    void *data;
    int size;

    ClassArray() {}

    ClassArray(const std::vector<cv::TrackerTargetState> & vec);

    operator std::vector<cv::TrackerTargetState>();
};

struct ConfidenceMap {
    struct ClassArray class_array;
    struct FloatArray float_array;
    int size;

    ConfidenceMap() {}
    ConfidenceMap(std::vector<std::pair<cv::Ptr<cv::TrackerTargetState>, float>> vec);

    operator std::vector<std::pair<cv::Ptr<cv::TrackerTargetState>, float>>();
};

struct ConfidenceMapArray {
    struct ConfidenceMap *array;
    int size;

    ConfidenceMapArray(){}
    ConfidenceMapArray(std::vector<std::vector<std::pair<cv::Ptr<cv::TrackerTargetState>, float>>> vec);

    operator std::vector<std::vector<std::pair<cv::Ptr<cv::TrackerTargetState>, float>>>();
};

struct FloatArrayPlusInt {
    struct FloatArray array;
    int val;
};

extern "C"
struct ConfidenceMapArray test(
        struct ConfidenceMapArray val);

//WeakClassifierHaarFeature

extern "C"
struct WeakClassifierHaarFeaturePtr {
    void *ptr;

    inline cv::WeakClassifierHaarFeature * operator->() {
        return static_cast<cv::WeakClassifierHaarFeature *>(ptr);
    }
    inline WeakClassifierHaarFeaturePtr(cv::WeakClassifierHaarFeature *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct WeakClassifierHaarFeaturePtr WeakClassifierHaarFeature_ctor();

extern "C"
void WeakClassifierHaarFeature_dtor(
        struct WeakClassifierHaarFeaturePtr ptr);

extern "C"
int WeakClassifierHaarFeature_eval(
        struct WeakClassifierHaarFeaturePtr ptr, float value);

extern "C"
bool WeakClassifierHaarFeature_update(
        struct WeakClassifierHaarFeaturePtr ptr, float value, int target);

//BaseClassifier

extern "C"
struct BaseClassifierPtr {
    void *ptr;

    inline cv::BaseClassifier * operator->() {
        return static_cast<cv::BaseClassifier *>(ptr);
    }
    inline BaseClassifierPtr(cv::BaseClassifier *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct BaseClassifierPtr BaseClassifier_ctor(
        int numWeakClassifier, int iterationInit);

extern "C"
void BaseClassifier_dtor(
        struct BaseClassifierPtr ptr);

extern "C"
int BaseClassifier_computeReplaceWeakestClassifier(
        struct BaseClassifierPtr ptr, struct FloatArray errors);

extern "C"
int BaseClassifier_eval(
        struct BaseClassifierPtr ptr, struct TensorWrapper image);

extern "C"
float BaseClassifier_getError(
        struct BaseClassifierPtr ptr, int curWeakClassifier);

extern "C"
struct FloatArray BaseClassifier_getErrors(
        struct BaseClassifierPtr ptr, struct FloatArray errors);

extern "C"
int BaseClassifier_getIdxOfNewWeakClassifier(
        struct BaseClassifierPtr ptr);

//TODO cv::BaseClassifier::getReferenceWeakClassifier

extern "C"
int BaseClassifier_getSelectedClassifier(
        struct BaseClassifierPtr ptr);

extern "C"
void BaseClassifier_replaceClassifierStatistic(
        struct BaseClassifierPtr ptr, int sourceIndex, int targetIndex);

extern "C"
void BaseClassifier_replaceWeakClassifier(
        struct BaseClassifierPtr ptr, int index);

extern "C"
struct FloatArrayPlusInt BaseClassifier_selectBestClassifier(
        struct BaseClassifierPtr ptr, struct BoolArray errorMask,
        float importance, struct FloatArray errors);

extern "C"
struct BoolArray BaseClassifier_trainClassifier(
        struct BaseClassifierPtr ptr, struct TensorWrapper image,
        int target, float importance, struct BoolArray errorMask);


//EstimatedGaussDistribution


extern "C"
struct EstimatedGaussDistributionPtr {
    void *ptr;

    inline cv::EstimatedGaussDistribution * operator->() {
        return static_cast<cv::EstimatedGaussDistribution *>(ptr);
    }
    inline EstimatedGaussDistributionPtr(cv::EstimatedGaussDistribution *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct EstimatedGaussDistributionPtr EstimatedGaussDistribution_ctor();

extern "C"
struct EstimatedGaussDistributionPtr EstimatedGaussDistribution_ctor2(
        float P_mean, float R_mean, float P_sigma, float R_sigma);

extern "C"
void EstimatedGaussDistribution_dtor(
        struct EstimatedGaussDistributionPtr ptr);

extern "C"
float EstimatedGaussDistribution_getMean(
        struct EstimatedGaussDistributionPtr ptr);

extern "C"
float EstimatedGaussDistribution_getSigma(
        struct EstimatedGaussDistributionPtr ptr);

extern "C"
void EstimatedGaussDistribution_setValues(
        struct EstimatedGaussDistributionPtr ptr, float mean, float sigma);

extern "C"
void EstimatedGaussDistribution_update(
        struct EstimatedGaussDistributionPtr ptr, float value);

//ClassifierThreshold

extern "C"
struct ClassifierThresholdPtr {
    void *ptr;

    inline cv::ClassifierThreshold * operator->() {
        return static_cast<cv::ClassifierThreshold *>(ptr);
    }
    inline ClassifierThresholdPtr(cv::ClassifierThreshold *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct ClassifierThresholdPtr ClassifierThreshold_ctor(
        struct EstimatedGaussDistributionPtr posSamples,
        struct EstimatedGaussDistributionPtr negSamples);

extern "C"
void ClassifierThreshold_dtor(
        struct ClassifierThresholdPtr ptr);

extern "C"
int ClassifierThreshold_eval(
        struct ClassifierThresholdPtr ptr, float value);

//TODO void* cv::ClassifierThreshold::getDistribution(int target)

extern "C"
void ClassifierThreshold_update(
        struct ClassifierThresholdPtr ptr,
        float value, int target);

//ClfMilBoost

struct Params {
    Params();
    int _numSel;
    int _numFeat;
    float _lRate;
};

extern "C"
struct ClfMilBoostPtr {
    void *ptr;

    inline cv::ClfMilBoost * operator->() {
        return static_cast<cv::ClfMilBoost *>(ptr);
    }
    inline ClfMilBoostPtr(cv::ClfMilBoost *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct ClfMilBoostPtr ClfMilBoost_ctor();

extern "C"
void ClfMilBoost_dtor(
        struct ClfMilBoostPtr ptr);

extern "C"
struct FloatArray ClfMilBoost_classify(
        struct ClfMilBoostPtr ptr, struct TensorWrapper x, bool logR);

extern "C"
void ClfMilBoost_init(
        struct ClfMilBoostPtr ptr, struct Params parameters);

extern "C"
float ClfMilBoost_sigmoid(
        struct ClfMilBoostPtr ptr, float x);

extern "C"
void ClfMilBoost_update(
        struct ClfMilBoostPtr ptr, struct TensorWrapper posx,
        struct TensorWrapper negx);

//ClfOnlineStump

extern "C"
struct ClfOnlineStumpPtr {
    void *ptr;

    inline cv::ClfOnlineStump * operator->() {
        return static_cast<cv::ClfOnlineStump *>(ptr);
    }
    inline ClfOnlineStumpPtr(cv::ClfOnlineStump *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct ClfOnlineStumpPtr ClfOnlineStump_ctor();

extern "C"
void ClfOnlineStump_dtor(
        struct ClfOnlineStumpPtr ptr);

extern "C"
bool ClfOnlineStump_classify(
        struct ClfOnlineStumpPtr ptr, struct TensorWrapper x, int i);

extern "C"
float ClfOnlineStump_classifyF(
        struct ClfOnlineStumpPtr ptr, struct TensorWrapper x, int i);

extern "C"
struct FloatArray ClfOnlineStump_classifySetF(
        struct ClfOnlineStumpPtr ptr, struct TensorWrapper x);

extern "C"
void ClfOnlineStump_init(
        struct ClfOnlineStumpPtr ptr);

extern "C"
void ClfOnlineStump_update(
        struct ClfOnlineStumpPtr ptr, struct TensorWrapper posx,
        struct TensorWrapper negx);

//CvParams

extern "C"
struct CvParamsPtr {
    void *ptr;

    inline cv::CvParams * operator->() {
        return static_cast<cv::CvParams *>(ptr);
    }
    inline CvParamsPtr(cv::CvParams *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
void CvParams_dtor(
        struct CvParamsPtr ptr);

extern "C"
void CvParams_printAttrs(
        struct CvParamsPtr ptr);

extern "C"
void CvParams_printDefaults(
        struct CvParamsPtr ptr);

extern "C"
bool CvParams_read(
        struct CvParamsPtr ptr, struct FileNodePtr node);

extern "C"
bool CvParams_scanAttr(
        struct CvParamsPtr ptr, const char* prmName, const char* val);

extern "C"
void CvParams_write(
        struct CvParamsPtr ptr, struct FileStoragePtr fs);

//CvFeatureParams

extern "C"
struct CvFeatureParamsPtr {
    void *ptr;

    inline cv::CvFeatureParams * operator->() {
        return static_cast<cv::CvFeatureParams *>(ptr);
    }
    inline CvFeatureParamsPtr(cv::CvFeatureParams *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct CvFeatureParamsPtr CvFeatureParams_ctor(
        int featureType);

extern "C"
void CvFeatureParams_init(
        struct CvFeatureParamsPtr ptr, struct CvFeatureParamsPtr fp);

extern "C"
bool CvFeatureParams_read(
        struct CvFeatureParamsPtr ptr, struct FileNodePtr node);

extern "C"
void CvFeatureParams_write(
        struct CvFeatureParamsPtr ptr, struct FileStoragePtr fs);

//CvHaarFeatureParams

extern "C"
struct CvHaarFeatureParamsPtr {
    void *ptr;

    inline cv::CvHaarFeatureParams * operator->() {
        return static_cast<cv::CvHaarFeatureParams *>(ptr);
    }
    inline CvHaarFeatureParamsPtr(cv::CvHaarFeatureParams *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct CvHaarFeatureParamsPtr CvHaarFeatureParams_ctor();

extern "C"
void CvHaarFeatureParams_init(
        struct CvHaarFeatureParamsPtr ptr, struct CvFeatureParamsPtr fp);

extern "C"
void CvHaarFeatureParams_printAttrs(
        struct CvHaarFeatureParamsPtr ptr);

extern "C"
void CvHaarFeatureParams_printDefaults(
        struct CvHaarFeatureParamsPtr ptr);

extern "C"
bool CvHaarFeatureParams_read(
        struct CvHaarFeatureParamsPtr ptr, struct FileNodePtr node);

extern "C"
bool CvHaarFeatureParams_scanAttr(
        struct CvHaarFeatureParamsPtr ptr, const char* prm, const char* val);

extern "C"
void CvHaarFeatureParams_write(
        struct CvHaarFeatureParamsPtr ptr, struct FileStoragePtr fs);

//CvHOGFeatureParams

extern "C"
struct CvHOGFeatureParamsPtr {
    void *ptr;

    inline cv::CvHOGFeatureParams * operator->() {
        return static_cast<cv::CvHOGFeatureParams *>(ptr);
    }
    inline CvHOGFeatureParamsPtr(cv::CvHOGFeatureParams *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct CvHOGFeatureParamsPtr CvHOGFeatureParams_ctor();

//CvLBPFeatureParams

extern "C"
struct CvLBPFeatureParamsPtr {
    void *ptr;

    inline cv::CvLBPFeatureParams * operator->() {
        return static_cast<cv::CvLBPFeatureParams *>(ptr);
    }
    inline CvLBPFeatureParamsPtr(cv::CvLBPFeatureParams *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct CvLBPFeatureParamsPtr CvLBPFeatureParams_ctor();

//CvFeatureEvaluator

extern "C"
struct CvFeatureEvaluatorPtr {
    void *ptr;

    inline cv::CvFeatureEvaluator * operator->() {
        return static_cast<cv::CvFeatureEvaluator *>(ptr);
    }
    inline CvFeatureEvaluatorPtr(cv::CvFeatureEvaluator *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct CvFeatureEvaluatorPtr CvFeatureEvaluator_ctor(
        int type);

extern "C"
void CvFeatureEvaluator_dtor(
        struct CvFeatureEvaluatorPtr ptr);

extern "C"
struct TensorWrapper CvFeatureEvaluator_getCls(
        struct CvFeatureEvaluatorPtr ptr);

extern "C"
float CvFeatureEvaluator_getCls2(
        struct CvFeatureEvaluatorPtr ptr, int si);

extern "C"
int CvFeatureEvaluator_getFeatureSize(
        struct CvFeatureEvaluatorPtr ptr);

extern "C"
int CvFeatureEvaluator_getMaxCatCount(
        struct CvFeatureEvaluatorPtr ptr);

extern "C"
int CvFeatureEvaluator_getNumFeatures(
        struct CvFeatureEvaluatorPtr ptr);

extern "C"
void CvFeatureEvaluator_init(
        struct CvFeatureEvaluatorPtr ptr, struct CvFeatureParamsPtr _featureParams,
        int _maxSampleCount, struct SizeWrapper _winSize);

extern "C"
float CvFeatureEvaluator_call(
        struct CvFeatureEvaluatorPtr ptr, int featureIdx, int sampleIdx);

extern "C"
void CvFeatureEvaluator_setImage(
        struct CvFeatureEvaluatorPtr ptr, struct TensorWrapper img,
        unsigned char clsLabel, int idx);

extern "C"
void CvFeatureEvaluator_writeFeatures(
        struct CvFeatureEvaluatorPtr ptr, struct FileStoragePtr fs,
        struct TensorWrapper featureMap);

//FeatureHaar

extern "C"
struct FeatureHaarPtr {
    void *ptr;

    inline cv::CvHaarEvaluator::FeatureHaar * operator->() {
        return static_cast<cv::CvHaarEvaluator::FeatureHaar *>(ptr);
    }
    inline FeatureHaarPtr(cv::CvHaarEvaluator::FeatureHaar *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct FeatureHaarPtr FeatureHaar_ctor(
        struct SizeWrapper patchSize);

extern "C"
void FeatureHaar_dtor(
        struct FeatureHaarPtr ptr);

struct FloatPlusBool {
    bool v1;
    float v2;
};

extern "C"
struct FloatPlusBool FeatureHaar_eval(
        struct FeatureHaarPtr ptr, struct TensorWrapper image, struct RectWrapper ROI);

extern "C"
struct RectArray FeatureHaar_getAreas(
    struct FeatureHaarPtr ptr);

extern "C"
float FeatureHaar_getInitMean(
        struct FeatureHaarPtr ptr);

extern "C"
float FeatureHaar_getInitSigma(
        struct FeatureHaarPtr ptr);

extern "C"
int FeatureHaar_getNumAreas(
        struct FeatureHaarPtr ptr);

extern "C"
struct FloatArray FeatureHaar_getWeights(
        struct FeatureHaarPtr ptr);

//CvHaarEvaluator

extern "C"
struct CvHaarEvaluatorPtr {
    void *ptr;

    inline cv::CvHaarEvaluator * operator->() {
        return static_cast<cv::CvHaarEvaluator *>(ptr);
    }
    inline CvHaarEvaluatorPtr(cv::CvHaarEvaluator *ptr) {
        this->ptr = ptr;
    }
};



extern "C"
struct CvHaarEvaluatorPtr CvHaarEvaluator_ctor();

extern "C"
void CvHaarEvaluator_dtor(
        struct CvHaarEvaluatorPtr ptr);

extern "C"
void CvHaarEvaluator_generateFeatures(
        struct CvHaarEvaluatorPtr ptr);

extern "C"
void CvHaarEvaluator_generateFeatures2(
        struct CvHaarEvaluatorPtr ptr, int numFeatures);

//TODO need to do
//const std::vector< CvHaarEvaluator::FeatureHaar > & getFeatures () const
//CvHaarEvaluator::FeatureHaar & getFeatures (int idx)

extern "C"
void CvHaarEvaluator_init(
        struct CvHaarEvaluatorPtr ptr, struct CvFeatureParamsPtr _featureParams,
        int _maxSampleCount, struct SizeWrapper _winSize);

extern "C"
float CvHaarEvaluator_call(
        struct CvHaarEvaluatorPtr ptr, int featureIdx, int sampleIdx);

extern "C"
void CvHaarEvaluator_setImage(
        struct CvHaarEvaluatorPtr ptr, struct TensorWrapper img,
        unsigned char clsLabel, int idx);

extern "C"
void CvHaarEvaluator_setWinSize(
        struct CvHaarEvaluatorPtr ptr, struct SizeWrapper patchSize);

extern "C"
struct SizeWrapper CvHaarEvaluator_setWinSize2(
        struct CvHaarEvaluatorPtr ptr);

extern "C"
void CvHaarEvaluator_writeFeature(
        struct CvHaarEvaluatorPtr ptr, struct FileStoragePtr fs);

extern "C"
void CvHaarEvaluator_writeFeatures(
        struct CvHaarEvaluatorPtr ptr, struct FileStoragePtr fs,
        struct TensorWrapper featureMap);

//CvHOGEvaluator

extern "C"
struct CvHOGEvaluatorPtr {
    void *ptr;

    inline cv::CvHOGEvaluator * operator->() {
        return static_cast<cv::CvHOGEvaluator *>(ptr);
    }
    inline CvHOGEvaluatorPtr(cv::CvHOGEvaluator *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct CvHOGEvaluatorPtr CvHOGEvaluator_ctor();

extern "C"
void CvHOGEvaluator_dtor(
        struct CvHOGEvaluatorPtr ptr);

extern "C"
void CvHOGEvaluator_init(
        struct CvHOGEvaluatorPtr ptr, struct CvFeatureParamsPtr _featureParams,
        int _maxSampleCount, struct SizeWrapper _winSize);

extern "C"
float CvHOGEvaluator_call(
        struct CvHOGEvaluatorPtr ptr, int featureIdx, int sampleIdx);

extern "C"
void CvHOGEvaluator_setImage(
        struct CvHOGEvaluatorPtr ptr, struct TensorWrapper img,
        unsigned char clsLabel, int idx);

extern "C"
void CvHOGEvaluator_writeFeatures(
        struct CvHOGEvaluatorPtr ptr, struct FileStoragePtr fs,
        struct TensorWrapper featureMap);

//CvLBPEvaluator

extern "C"
struct CvLBPEvaluatorPtr {
    void *ptr;

    inline cv::CvLBPEvaluator * operator->() {
        return static_cast<cv::CvLBPEvaluator *>(ptr);
    }
    inline CvLBPEvaluatorPtr(cv::CvLBPEvaluator *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct CvLBPEvaluatorPtr CvLBPEvaluator_ctor();

extern "C"
void CvLBPEvaluator_dtor(
        struct CvLBPEvaluatorPtr ptr);

extern "C"
void CvLBPEvaluator_init(
        struct CvLBPEvaluatorPtr ptr, struct CvFeatureParamsPtr _featureParams,
        int _maxSampleCount, struct SizeWrapper _winSize);

extern "C"
float CvLBPEvaluator_call(
        struct CvLBPEvaluatorPtr ptr, int featureIdx, int sampleIdx);

extern "C"
void CvLBPEvaluator_setImage(
        struct CvLBPEvaluatorPtr ptr, struct TensorWrapper img,
        unsigned char clsLabel, int idx);

extern "C"
void CvLBPEvaluator_writeFeatures(
        struct CvLBPEvaluatorPtr ptr, struct FileStoragePtr fs,
        struct TensorWrapper featureMap);

//StrongClassifierDirectSelection

extern "C"
struct StrongClassifierDirectSelectionPtr {
    void *ptr;

    inline cv::StrongClassifierDirectSelection * operator->() {
        return static_cast<cv::StrongClassifierDirectSelection *>(ptr);
    }
    inline StrongClassifierDirectSelectionPtr(cv::StrongClassifierDirectSelection *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct StrongClassifierDirectSelectionPtr StrongClassifierDirectSelection_ctor(
        int numBaseClf, int numWeakClf, struct SizeWrapper patchSz,
        struct RectWrapper sampleROI, bool useFeatureEx, int iterationInit);

extern "C"
void StrongClassifierDirectSelection_dtor(
        struct StrongClassifierDirectSelectionPtr ptr);

struct FloatPlusInt {
    float v1;
    int v2;
};

extern "C"
struct FloatPlusInt StrongClassifierDirectSelection_classifySmooth(
        struct StrongClassifierDirectSelectionPtr ptr, struct TensorArray images,
        struct RectWrapper sampleROI);

extern "C"
float StrongClassifierDirectSelection_eval(
        struct StrongClassifierDirectSelectionPtr ptr, struct TensorWrapper response);

extern "C"
int StrongClassifierDirectSelection_getNumBaseClassifier(
        struct StrongClassifierDirectSelectionPtr ptr);

extern "C"
struct SizeWrapper StrongClassifierDirectSelection_getPatchSize(
    struct StrongClassifierDirectSelectionPtr ptr);

extern "C"
int StrongClassifierDirectSelection_getReplacedClassifier(
        struct StrongClassifierDirectSelectionPtr ptr);

extern "C"
struct RectWrapper StrongClassifierDirectSelection_getROI(
        struct StrongClassifierDirectSelectionPtr ptr);

extern "C"
struct IntArray StrongClassifierDirectSelection_getSelectedWeakClassifier(
        struct StrongClassifierDirectSelectionPtr ptr);

extern "C"
int StrongClassifierDirectSelection_getSwappedClassifier(
        struct StrongClassifierDirectSelectionPtr ptr);

extern "C"
bool StrongClassifierDirectSelection_getUseFeatureExchange(
        struct StrongClassifierDirectSelectionPtr ptr);

extern "C"
void StrongClassifierDirectSelection_initBaseClassifier(
        struct StrongClassifierDirectSelectionPtr ptr);

extern "C"
void StrongClassifierDirectSelection_replaceWeakClassifier(
        struct StrongClassifierDirectSelectionPtr ptr, int idx);

extern "C"
bool StrongClassifierDirectSelection_update(
        struct StrongClassifierDirectSelectionPtr ptr, struct TensorWrapper image,
        int target, float importance);

//Detector

extern "C"
struct DetectorPtr {
    void *ptr;

    inline cv::Detector * operator->() {
        return static_cast<cv::Detector *>(ptr);
    }
    inline DetectorPtr(cv::Detector *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct DetectorPtr Detector_ctor(
        struct StrongClassifierDirectSelectionPtr ptr);

extern "C"
void Detector_dtor(
        struct DetectorPtr ptr);

extern "C"
void Detector_classifySmooth(
        struct DetectorPtr ptr, struct TensorArray image, float minMargin);

extern "C"
float Detector_getConfidence(
        struct DetectorPtr ptr, int patchIdx);

extern "C"
float Detector_getConfidenceOfBestDetection(
        struct DetectorPtr ptr);

extern "C"
float Detector_getConfidenceOfDetection(
        struct DetectorPtr ptr, int detectionIdx);

extern "C"
struct FloatArray Detector_getConfidences(
        struct DetectorPtr ptr);

extern "C"
struct TensorWrapper Detector_getConfImageDisplay(
        struct DetectorPtr ptr);

extern "C"
struct IntArray Detector_getIdxDetections(
        struct DetectorPtr ptr);

extern "C"
int Detector_getNumDetections(
        struct DetectorPtr ptr);

extern "C"
int Detector_getPatchIdxOfDetection(
        struct DetectorPtr ptr, int detectionIdx);

//MultiTracker

extern "C"
struct MultiTrackerPtr {
    void *ptr;

    inline cv::MultiTracker * operator->() {
        return static_cast<cv::MultiTracker *>(ptr);
    }
    inline MultiTrackerPtr(cv::MultiTracker *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct MultiTrackerPtr MultiTracker_ctor(
        const char *trackerType);

extern "C"
void MultiTracker_dtor(
        struct MultiTrackerPtr ptr);

extern "C"
bool MultiTracker_add(
        struct MultiTrackerPtr ptr, struct TensorWrapper image,
        struct Rect2dWrapper boundingBox);

extern "C"
bool MultiTracker_add2(
        struct MultiTrackerPtr ptr, const char *trackerType,
        struct TensorWrapper image, struct Rect2dWrapper boundingBox);

extern "C"
bool MultiTracker_add3(
        struct MultiTrackerPtr ptr, const char *trackerType,
        struct TensorWrapper image, struct Rect2dArray boundingBox);

extern "C"
bool MultiTracker_add4(
        struct MultiTrackerPtr ptr, struct TensorWrapper image,
        struct Rect2dArray boundingBox);

extern "C"
bool MultiTracker_update(
        struct MultiTrackerPtr ptr, struct TensorWrapper image);

struct Rect2dArrayPlusBool {
    struct Rect2dArray rect2d;
    bool val;

};
extern "C"
struct Rect2dArrayPlusBool MultiTracker_update2(
        struct MultiTrackerPtr ptr, struct TensorWrapper image);

//MultiTracker_Alt

extern "C"
struct MultiTracker_AltPtr {
    void *ptr;

    inline cv::MultiTracker_Alt * operator->() {
        return static_cast<cv::MultiTracker_Alt *>(ptr);
    }
    inline MultiTracker_AltPtr(cv::MultiTracker_Alt *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct MultiTracker_AltPtr MultiTracker_Alt_ctor();

extern "C"
void MultiTracker_Alt_dtor(
        struct MultiTracker_AltPtr ptr);

extern "C"
bool MultiTracker_Alt_addTarget(
        struct MultiTracker_AltPtr ptr, struct TensorWrapper image,
        struct Rect2dWrapper boundingBox, const char *tracker_algorithm_name);

extern "C"
bool MultiTracker_Alt_update(
        struct MultiTracker_AltPtr ptr, struct TensorWrapper image);

//MultiTrackerTLD

extern "C"
struct MultiTrackerTLDPtr {
    void *ptr;

    inline cv::MultiTrackerTLD * operator->() {
        return static_cast<cv::MultiTrackerTLD *>(ptr);
    }
    inline MultiTrackerTLDPtr(cv::MultiTrackerTLD *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct MultiTrackerTLDPtr MultiTrackerTLD_ctor();

extern "C"
void MultiTrackerTLD_dtor(
        struct MultiTrackerTLDPtr ptr);

extern "C"
bool MultiTrackerTLD_update_opt(
        struct MultiTrackerTLDPtr ptr, struct TensorWrapper image);

//ROISelector

extern "C"
struct ROISelectorPtr {
    void *ptr;

    inline cv::ROISelector * operator->() {
        return static_cast<cv::ROISelector *>(ptr);
    }
    inline ROISelectorPtr(cv::ROISelector *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct ROISelectorPtr ROISelector_ctor();

extern "C"
void ROISelector_dtor(
        struct ROISelectorPtr ptr);

extern "C"
struct Rect2dWrapper ROISelector_select(
        struct ROISelectorPtr ptr, struct TensorWrapper image, bool fromCenter);

extern "C"
struct Rect2dWrapper ROISelector_select2(
        struct ROISelectorPtr ptr, const char *windowName,
        struct TensorWrapper img, bool showCrossair, bool fromCenter);

extern "C"
void ROISelector_select3(
        struct ROISelectorPtr ptr, const char *windowName, struct TensorWrapper img,
        struct Rect2dArray boundingBox, bool fromCenter);

//TrackerTargetState

extern "C"
struct TrackerTargetStatePtr {
    void *ptr;

    TrackerTargetStatePtr() {}

    inline cv::TrackerTargetState * operator->() {
        return static_cast<cv::TrackerTargetState *>(ptr);
    }
    inline TrackerTargetStatePtr(cv::TrackerTargetState *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct TrackerTargetStatePtr TrackerTargetState_ctor();

extern "C"
void TrackerTargetState_dtor(
        struct TrackerTargetStatePtr ptr);

extern "C"
int TrackerTargetState_getTargetHeight(
        struct TrackerTargetStatePtr ptr);

extern "C"
struct Point2fWrapper TrackerTargetState_getTargetPosition(
        struct TrackerTargetStatePtr ptr);

extern "C"
int TrackerTargetState_getTargetWidth(
        struct TrackerTargetStatePtr ptr);

extern "C"
void TrackerTargetState_setTargetHeight(
        struct TrackerTargetStatePtr ptr, int height);

extern "C"
void TrackerTargetState_setTargetPosition(
        struct TrackerTargetStatePtr ptr, struct Point2fWrapper position);

extern "C"
void TrackerTargetState_setTargetWidth(
        struct TrackerTargetStatePtr ptr, int width);

//TrackerStateEstimator

extern "C"
struct TrackerStateEstimatorPtr {
    void *ptr;

    inline cv::TrackerStateEstimator * operator->() {
        return static_cast<cv::TrackerStateEstimator *>(ptr);
    }
    inline TrackerStateEstimatorPtr(cv::TrackerStateEstimator *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct TrackerStateEstimatorPtr TrackerStateEstimator_ctor(
        const char *trackeStateEstimatorType);

extern "C"
void TrackerStateEstimator_dtor(
        struct TrackerStateEstimatorPtr ptr);

extern "C"
struct TrackerTargetStatePtr TrackerStateEstimator_estimate(
        struct TrackerStateEstimatorPtr ptr, struct ConfidenceMapArray confidenceMaps);

extern "C"
const char* TrackerStateEstimator_getClassName(
        struct TrackerStateEstimatorPtr ptr);

extern "C"
void TrackerStateEstimator_update(
        struct TrackerStateEstimatorPtr ptr, struct ConfidenceMapArray confidenceMaps);

//TrackerStateEstimatorAdaBoosting

extern "C"
struct TrackerStateEstimatorAdaBoostingPtr {
    void *ptr;

    inline cv::TrackerStateEstimatorAdaBoosting * operator->() {
        return static_cast<cv::TrackerStateEstimatorAdaBoosting *>(ptr);
    }
    inline TrackerStateEstimatorAdaBoostingPtr(cv::TrackerStateEstimatorAdaBoosting *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct TrackerStateEstimatorAdaBoostingPtr TrackerStateEstimatorAdaBoosting_ctor(
        int numClassifer, int initIterations, int nFeatures,
        struct SizeWrapper patchSize, struct RectWrapper ROI);

extern "C"
void TrackerStateEstimatorAdaBoosting_dtor(
        struct TrackerStateEstimatorAdaBoostingPtr ptr);

extern "C"
struct IntArray TrackerStateEstimatorAdaBoosting_computeReplacedClassifier(
        struct TrackerStateEstimatorAdaBoostingPtr ptr);

extern "C"
struct IntArray TrackerStateEstimatorAdaBoosting_computeSelectedWeakClassifier(
        struct TrackerStateEstimatorAdaBoostingPtr ptr);

extern "C"
struct IntArray TrackerStateEstimatorAdaBoosting_computeSwappedClassifier(
        struct TrackerStateEstimatorAdaBoostingPtr ptr);

extern "C"
void TrackerStateEstimatorAdaBoosting_setCurrentConfidenceMap(
        struct TrackerStateEstimatorAdaBoostingPtr ptr, struct ConfidenceMap confidenceMap);

extern "C"
struct RectWrapper TrackerStateEstimatorAdaBoosting_getSampleROI(
        struct TrackerStateEstimatorAdaBoostingPtr ptr);

extern "C"
void TrackerStateEstimatorAdaBoosting_setSampleROI(
        struct TrackerStateEstimatorAdaBoostingPtr ptr, struct RectWrapper ROI);

//TrackerStateEstimatorMILBoosting

extern "C"
struct TrackerStateEstimatorMILBoostingPtr {
    void *ptr;

    inline cv::TrackerStateEstimatorMILBoosting * operator->() {
        return static_cast<cv::TrackerStateEstimatorMILBoosting *>(ptr);
    }
    inline TrackerStateEstimatorMILBoostingPtr(cv::TrackerStateEstimatorMILBoosting *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct TrackerStateEstimatorMILBoostingPtr TrackerStateEstimatorMILBoosting_ctor(
        int nFeatures);

extern "C"
void TrackerStateEstimatorMILBoosting_dtor(
        struct TrackerStateEstimatorMILBoostingPtr ptr);

extern "C"
void TrackerStateEstimatorMILBoosting_setCurrentConfidenceMap(
        struct TrackerStateEstimatorMILBoostingPtr ptr, struct ConfidenceMap confidenceMap);

//TrackerStateEstimatorSVM

extern "C"
struct TrackerStateEstimatorSVMPtr {
    void *ptr;

    inline cv::TrackerStateEstimatorSVM * operator->() {
        return static_cast<cv::TrackerStateEstimatorSVM *>(ptr);
    }
    inline TrackerStateEstimatorSVMPtr(cv::TrackerStateEstimatorSVM *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct TrackerStateEstimatorSVMPtr TrackerStateEstimatorSVM_ctor();

extern "C"
void TrackerStateEstimatorSVM_dtor(
        struct TrackerStateEstimatorSVMPtr ptr);

//TrackerModel

extern "C"
struct TrackerModelPtr {
    void *ptr;

    inline cv::TrackerModel * operator->() {
        return static_cast<cv::TrackerModel *>(ptr);
    }
    inline TrackerModelPtr(cv::TrackerModel *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
void TrackerModel_dtor(
        struct TrackerModelPtr ptr);

extern "C"
struct ConfidenceMapArray TrackerModel_getConfidenceMaps(
        struct TrackerModelPtr ptr);

extern "C"
struct ConfidenceMap TrackerModel_getLastConfidenceMap(
        struct TrackerModelPtr ptr);

extern "C"
struct TrackerTargetStatePtr TrackerModel_getLastTargetState(
        struct TrackerModelPtr ptr);

extern "C"
struct TrackerStateEstimatorPtr TrackerModel_getTrackerStateEstimator(
        struct TrackerModelPtr ptr);

extern "C"
void TrackerModel_modelEstimation(
        struct TrackerModelPtr ptr, struct TensorArray responses);

extern "C"
void TrackerModel_modelUpdate(
        struct TrackerModelPtr ptr);

extern "C"
bool TrackerModel_runStateEstimator(
        struct TrackerModelPtr ptr);

extern "C"
void TrackerModel_setLastTargetState(
        struct TrackerModelPtr ptr, struct TrackerTargetStatePtr lastTargetState);

extern "C"
bool TrackerModel_setTrackerStateEstimator(
        struct TrackerModelPtr ptr, struct TrackerStateEstimatorPtr trackerStateEstimator);

//Tracker

extern "C"
struct TrackerPtr {
    void *ptr;

    inline cv::Tracker * operator->() {
        return static_cast<cv::Tracker *>(ptr);
    }
    inline TrackerPtr(cv::Tracker *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
void Tracker_dtor(
        struct TrackerPtr ptr);

extern "C"
struct TrackerModelPtr Tracker_getModel(
        struct TrackerPtr ptr);

extern "C"
bool Tracker_init(
        struct TrackerPtr ptr, struct TensorWrapper image, struct Rect2dWrapper boundingBox);

extern "C"
void Tracker_read(
        struct TrackerPtr ptr, struct FileNodePtr fn);

extern "C"
bool Tracker_update(
        struct TrackerPtr ptr, struct TensorWrapper image, struct Rect2dWrapper boundingBox);

extern "C"
void Tracker_write(
        struct TrackerPtr ptr, struct FileStoragePtr fn);

//TrackerBoosting

extern "C"
struct TrackerBoostingPtr {
    void *ptr;

    inline cv::TrackerBoosting * operator->() {
        return static_cast<cv::TrackerBoosting *>(ptr);
    }
    inline TrackerBoostingPtr(cv::TrackerBoosting *ptr) {
        this->ptr = ptr;
    }
};

struct TrackerBoosting_Params {
    int numClassifiers;
    float samplerOverlap;
    float samplerSearchFactor;
    int iterationInit;
    int featureSetNumFeatures;
};

extern "C"
struct TrackerBoostingPtr TrackerBoosting_ctor(
        struct TrackerBoosting_Params parameters);

extern "C"
void TrackerBoosting_dtor(
        struct TrackerBoostingPtr ptr);

//TrackerKCF

extern "C"
struct TrackerKCFPtr {
    void *ptr;

    inline cv::TrackerKCF * operator->() {
        return static_cast<cv::TrackerKCF *>(ptr);
    }
    inline TrackerKCFPtr(cv::TrackerKCF *ptr) {
        this->ptr = ptr;
    }
};

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

extern "C"
struct TrackerKCFPtr TrackerKCF_ctor(
        struct TrackerKCF_Params parameters);

extern "C"
void TrackerKCF_dtor(
        struct TrackerKCFPtr ptr);

//TrackerMedianFlow

extern "C"
struct TrackerMedianFlowPtr {
    void *ptr;

    inline cv::TrackerMedianFlow * operator->() {
        return static_cast<cv::TrackerMedianFlow *>(ptr);
    }
    inline TrackerMedianFlowPtr(cv::TrackerMedianFlow *ptr) {
        this->ptr = ptr;
    }
};

struct TrackerMedianFlow_Params {
    int pointsInGrid;
};

extern "C"
struct TrackerMedianFlowPtr TrackerMedianFlow_ctor(
        struct TrackerMedianFlow_Params parameters);

extern "C"
void TrackerMedianFlow_dtor(
        struct TrackerMedianFlowPtr ptr);

//TrackerMIL

extern "C"
struct TrackerMILPtr {
    void *ptr;

    inline cv::TrackerMIL * operator->() {
        return static_cast<cv::TrackerMIL *>(ptr);
    }
    inline TrackerMILPtr(cv::TrackerMIL *ptr) {
        this->ptr = ptr;
    }
};

struct TrackerMIL_Params {
    float samplerInitInRadius;
    int samplerInitMaxNegNum;
    float samplerSearchWinSize;
    float samplerTrackInRadius;
    int samplerTrackMaxPosNum;
    int samplerTrackMaxNegNum;
    int featureSetNumFeatures;
};

extern "C"
struct TrackerMILPtr TrackerMIL_ctor(
        struct TrackerMIL_Params parameters);

extern "C"
void TrackerMIL_dtor(
        struct TrackerMILPtr ptr);

//TrackerTLD

extern "C"
struct TrackerTLDPtr {
    void *ptr;

    inline cv::TrackerTLD * operator->() {
        return static_cast<cv::TrackerTLD *>(ptr);
    }
    inline TrackerTLDPtr(cv::TrackerTLD *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct TrackerTLDPtr TrackerTLD_ctor();

extern "C"
void TrackerTLD_dtor(
        struct TrackerTLDPtr ptr);

//TrackerAdaBoostingTargetState

extern "C"
struct TrackerAdaBoostingTargetStatePtr {
    void *ptr;

    inline cv::TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState * operator->() {
        return static_cast<cv::TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState *>(ptr);
    }
    inline TrackerAdaBoostingTargetStatePtr(
            cv::TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct TrackerAdaBoostingTargetStatePtr TrackerAdaBoostingTargetState_ctor(
        struct Point2fWrapper position, int width, int height,
        bool foreground, struct TensorWrapper responses);

extern "C"
void TrackerAdaBoostingTargetState_dtor(
        struct TrackerAdaBoostingTargetStatePtr ptr);

extern "C"
struct TensorWrapper TrackerAdaBoostingTargetState_getTargetResponses(
        struct TrackerAdaBoostingTargetStatePtr ptr);

extern "C"
bool TrackerAdaBoostingTargetState_isTargetFg(
        struct TrackerAdaBoostingTargetStatePtr ptr);

extern "C"
void TrackerAdaBoostingTargetState_setTargetFg(
        struct TrackerAdaBoostingTargetStatePtr ptr, bool foreground);

extern "C"
void TrackerAdaBoostingTargetState_setTargetResponses(
        struct TrackerAdaBoostingTargetStatePtr ptr, struct TensorWrapper responses);

//TrackerMILTargetState

extern "C"
struct TrackerMILTargetStatePtr {
    void *ptr;

    inline cv::TrackerStateEstimatorMILBoosting::TrackerMILTargetState * operator->() {
        return static_cast<cv::TrackerStateEstimatorMILBoosting::TrackerMILTargetState *>(ptr);
    }
    inline TrackerMILTargetStatePtr(
            cv::TrackerStateEstimatorMILBoosting::TrackerMILTargetState *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct TrackerMILTargetStatePtr TrackerMILTargetState_ctor(
        struct Point2fWrapper position, int width, int height, bool foreground, struct TensorWrapper features);

extern "C"
void TrackerMILTargetState_dtor(
        struct TrackerMILTargetStatePtr ptr);

extern "C"
struct TensorWrapper TrackerMILTargetState_getFeatures(
        struct TrackerMILTargetStatePtr ptr);

extern "C"
bool TrackerMILTargetState_isTargetFg(
        struct TrackerMILTargetStatePtr ptr);

extern "C"
void TrackerMILTargetState_setFeatures(
        struct TrackerMILTargetStatePtr ptr, struct TensorWrapper features);

extern "C"
void TrackerMILTargetState_setTargetFg(
        struct TrackerMILTargetStatePtr ptr, bool foreground);

//TrackerFeature

extern "C"
struct TrackerFeaturePtr {
    void *ptr;

    inline cv::TrackerFeature * operator->() {
        return static_cast<cv::TrackerFeature *>(ptr);
    }
    inline TrackerFeaturePtr(cv::TrackerFeature *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct TrackerFeaturePtr TrackerFeature_ctor(
        const char *rackerFeatureType);

extern "C"
void TrackerFeature_dtor(
        struct TrackerFeaturePtr ptr);

extern "C"
struct TensorWrapper TrackerFeature_compute(
        struct TrackerFeaturePtr ptr, struct TensorArray images);

extern "C"
const char* TrackerFeature_getClassName(
        struct TrackerFeaturePtr ptr);

extern "C"
void TrackerFeature_selection(
        struct TrackerFeaturePtr ptr, struct TensorWrapper response, int npoints);

//TrackerFeatureFeature2d

extern "C"
struct TrackerFeatureFeature2dPtr {
    void *ptr;

    inline cv::TrackerFeatureFeature2d * operator->() {
        return static_cast<cv::TrackerFeatureFeature2d *>(ptr);
    }
    inline TrackerFeatureFeature2dPtr(cv::TrackerFeatureFeature2d *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct TrackerFeatureFeature2dPtr TrackerFeatureFeature2d_ctor(
        const char *detectorType, const char *descriptorType);

extern "C"
void TrackerFeatureFeature2d_dtor(
        struct TrackerFeatureFeature2dPtr ptr);

extern "C"
void TrackerFeatureFeature2d_selection(
        struct TrackerFeatureFeature2dPtr ptr, struct TensorWrapper response, int npoints);

//TrackerFeatureHAAR

extern "C"
struct TrackerFeatureHAARPtr {
    void *ptr;

    inline cv::TrackerFeatureHAAR * operator->() {
        return static_cast<cv::TrackerFeatureHAAR *>(ptr);
    }
    inline TrackerFeatureHAARPtr(cv::TrackerFeatureHAAR *ptr) {
        this->ptr = ptr;
    }
};

struct TrackerFeatureHAAR_Params
{
    int numFeatures;
    struct SizeWrapper rectSize;
    bool isIntegral;
};

extern "C"
struct TrackerFeatureHAARPtr TrackerFeatureHAAR_ctor(
        struct TrackerFeatureHAAR_Params parameters);

extern "C"
void TrackerFeatureHAAR_dtor(
        struct TrackerFeatureHAARPtr ptr);

extern "C"
struct TensorWrapper TrackerFeatureHAAR_extractSelected(
        struct TrackerFeatureHAARPtr ptr, struct IntArray selFeatures,
        struct TensorArray images);

extern "C"
struct FeatureHaarPtr TrackerFeatureHAAR_getFeatureAt(
        struct TrackerFeatureHAARPtr ptr, int id);

extern "C"
void TrackerFeatureHAAR_selection(
    struct TrackerFeatureHAARPtr ptr, struct TensorWrapper response, int npoints);

extern "C"
bool TrackerFeatureHAAR_swapFeature(
        struct TrackerFeatureHAARPtr ptr, int source, int target);

extern "C"
bool TrackerFeatureHAAR_swapFeature2(
        struct TrackerFeatureHAARPtr ptr, int id,
        struct FeatureHaarPtr feature);

//TrackerFeatureHOG

extern "C"
struct TrackerFeatureHOGPtr {
    void *ptr;

    inline cv::TrackerFeatureHOG * operator->() {
        return static_cast<cv::TrackerFeatureHOG *>(ptr);
    }
    inline TrackerFeatureHOGPtr(cv::TrackerFeatureHOG *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct TrackerFeatureHOGPtr TrackerFeatureHOG_ctor();

extern "C"
void TrackerFeatureHOG_dtor(
        struct TrackerFeatureHOGPtr ptr);

extern "C"
void TrackerFeatureHOG_selection(
        struct TrackerFeatureHOGPtr ptr, struct TensorWrapper response, int npoints);

//TrackerFeatureLBP

extern "C"
struct TrackerFeatureLBPPtr {
    void *ptr;

    inline cv::TrackerFeatureLBP * operator->() {
        return static_cast<cv::TrackerFeatureLBP *>(ptr);
    }
    inline TrackerFeatureLBPPtr(cv::TrackerFeatureLBP *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct TrackerFeatureLBPPtr TrackerFeatureLBP_ctor();

extern "C"
void TrackerFeatureLBP_dtor(
        struct TrackerFeatureLBPPtr ptr);

extern "C"
void TrackerFeatureLBP_selection(
        struct TrackerFeatureLBPPtr ptr, struct TensorWrapper response, int npoints);

//TrackerFeatureSet

extern "C"
struct TrackerFeatureSetPtr {
    void *ptr;

    inline cv::TrackerFeatureSet * operator->() {
        return static_cast<cv::TrackerFeatureSet *>(ptr);
    }
    inline TrackerFeatureSetPtr(cv::TrackerFeatureSet *ptr) {
        this->ptr = ptr;
    }
};

extern "C"
struct TrackerFeatureSetPtr TrackerFeatureSet_ctor();

extern "C"
void TrackerFeatureSet_dtor(
        struct TrackerFeatureSetPtr ptr);

extern "C"
bool TrackerFeatureSet_addTrackerFeature(
        struct TrackerFeatureSetPtr ptr, const char *trackerFeatureType);


extern "C"
bool TrackerFeatureSet_addTrackerFeature2(
        struct TrackerFeatureSetPtr ptr, struct TrackerFeaturePtr feature);

extern "C"
void TrackerFeatureSet_extraction(
        struct TrackerFeatureSetPtr ptr, struct TensorArray images);

extern "C"
struct TensorArray TrackerFeatureSet_getResponses(
        struct TrackerFeatureSetPtr ptr);