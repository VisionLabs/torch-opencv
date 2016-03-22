#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/tracking.hpp>

struct FloatArrayPlusInt {
    struct FloatArray array;
    int val;
};

extern "C"
void test();

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