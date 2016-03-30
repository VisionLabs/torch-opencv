#include <tracking.hpp>

ClassArray::ClassArray(const std::vector<cv::TrackerTargetState> & vec)
{
    TrackerTargetStatePtr *temp = static_cast<TrackerTargetStatePtr *>(malloc(vec.size() * sizeof(TrackerTargetStatePtr)));

    this->size = vec.size();

    for (int i = 0; i < vec.size(); i++) {
        cv::TrackerTargetState *cls = new cv::TrackerTargetState();
        *cls = vec[i];
        temp[i] = cls;
    }
    this->data = temp;
}

ClassArray::operator std::vector<cv::TrackerTargetState>()
{
    TrackerTargetStatePtr *temp =
            static_cast<TrackerTargetStatePtr *>(this->data);

    std::vector<cv::TrackerTargetState> retval(this->size);

    for(int i = 0; i < this->size; i++) {
        retval[i] = *static_cast<cv::TrackerTargetState *>(temp[i].ptr);
    }
    return retval;
}

ConfidenceMap::ConfidenceMap(std::vector<std::pair<cv::Ptr<cv::TrackerTargetState>, float>> vec)
{
    this->size = vec.size();
    this->float_array.size = vec.size();
    this->float_array.data = static_cast<float *>(malloc(sizeof(float) * this->size));
    this->class_array.size = vec.size();
    TrackerTargetStatePtr *temp = static_cast<TrackerTargetStatePtr *>(malloc(vec.size() * sizeof(TrackerTargetStatePtr)));

    cv::TrackerTargetState *cls;
    for(int i = 0; i < vec.size(); i++) {
        this->float_array.data[i] = vec[i].second;
        cls = new cv::TrackerTargetState();
        *cls = *vec[i].first;
        temp[i] = cls;
    }
    this->class_array.data = temp;
}

ConfidenceMap::operator std::vector<std::pair<cv::Ptr<cv::TrackerTargetState>, float>>()
{
    std::vector<std::pair<cv::Ptr<cv::TrackerTargetState>, float>> vec(this->size);
    std::pair<cv::Ptr<cv::TrackerTargetState>, float> pair;

    TrackerTargetStatePtr *temp =
            static_cast<TrackerTargetStatePtr *>(this->class_array.data);

    for(int i = 0; i < this->size; i++){
        cv::TrackerTargetState * t = static_cast<cv::TrackerTargetState *>(malloc(sizeof(cv::TrackerTargetState)));
        memcpy(t, temp[i].ptr, sizeof(cv::TrackerTargetState));
        pair.first = cv::Ptr<cv::TrackerTargetState>(t);
        pair.second = this->float_array.data[i];
        vec[i] = pair;
    }
    return vec;
}

ConfidenceMapArray::operator std::vector<std::vector<std::pair<cv::Ptr<cv::TrackerTargetState>, float>>>()
{
    std::vector<std::vector<std::pair<cv::Ptr<cv::TrackerTargetState>, float>>> retval(this->size);

    for(int i = 0; i < this->size; i++) {
        retval[i] = this->array[i];
    }
    return retval;
};

ConfidenceMapArray::ConfidenceMapArray(std::vector<std::vector<std::pair<cv::Ptr<cv::TrackerTargetState>, float>>> vec)
{
    this->size = vec.size();
    this->array = static_cast<struct ConfidenceMap *>(malloc(vec.size() * sizeof(struct ConfidenceMap)));
    for(int i = 0; i < vec.size(); i++){
        this->array[i] = vec[i];
    }
}

extern "C"
struct ConfidenceMapArray test(
        struct ConfidenceMapArray val)
{
    std::vector<std::vector<std::pair<cv::Ptr<cv::TrackerTargetState>, float>>> vec = val;

    return ConfidenceMapArray(vec);
}

//WeakClassifierHaarFeature

extern "C"
struct WeakClassifierHaarFeaturePtr WeakClassifierHaarFeature_ctor()
{
    return new cv::WeakClassifierHaarFeature();
}

extern "C"
void WeakClassifierHaarFeature_dtor(
        struct WeakClassifierHaarFeaturePtr ptr)
{
    delete static_cast<cv::WeakClassifierHaarFeature *>(ptr.ptr);
}

extern "C"
int WeakClassifierHaarFeature_eval(
        struct WeakClassifierHaarFeaturePtr ptr, float value)
{
    return ptr->eval(value);
}

extern "C"
bool WeakClassifierHaarFeature_update(
        struct WeakClassifierHaarFeaturePtr ptr, float value, int target)
{
    return ptr->update(value, target);
}

//BaseClassifier

extern "C"
struct BaseClassifierPtr BaseClassifier_ctor(
        int numWeakClassifier, int iterationInit)
{
    return new cv::BaseClassifier(numWeakClassifier, iterationInit);
}

extern "C"
void BaseClassifier_dtor(
        struct BaseClassifierPtr ptr)
{
    delete static_cast<cv::BaseClassifier *>(ptr.ptr);
}

extern "C"
int BaseClassifier_computeReplaceWeakestClassifier(
        struct BaseClassifierPtr ptr, struct FloatArray errors)
{
    return ptr->computeReplaceWeakestClassifier(errors);
}

extern "C"
int BaseClassifier_eval(
        struct BaseClassifierPtr ptr, struct TensorWrapper image)
{
    ptr->eval(image.toMat());
}

extern "C"
float BaseClassifier_getError(
        struct BaseClassifierPtr ptr, int curWeakClassifier)
{
    ptr->getError(curWeakClassifier);
}

extern "C"
extern "C"
struct FloatArray BaseClassifier_getErrors(
        struct BaseClassifierPtr ptr, struct FloatArray errors)
{
    float * errors_vec = errors.data;
    ptr->getErrors(errors_vec);
    return errors;
}

extern "C"
int BaseClassifier_getIdxOfNewWeakClassifier(
        struct BaseClassifierPtr ptr)
{
    ptr->getIdxOfNewWeakClassifier();
}

extern "C"
int BaseClassifier_getSelectedClassifier(
        struct BaseClassifierPtr ptr)
{
    ptr->getSelectedClassifier();
}

extern "C"
void BaseClassifier_replaceClassifierStatistic(
        struct BaseClassifierPtr ptr, int sourceIndex, int targetIndex)
{
    ptr->replaceClassifierStatistic(sourceIndex, targetIndex);
}

extern "C"
void BaseClassifier_replaceWeakClassifier(
        struct BaseClassifierPtr ptr, int index)
{
    ptr->replaceWeakClassifier(index);
}

extern "C"
struct FloatArrayPlusInt BaseClassifier_selectBestClassifier(
        struct BaseClassifierPtr ptr, struct BoolArray errorMask,
        float importance, struct FloatArray errors)
{
    FloatArrayPlusInt result;
    std::vector<float> vec(errors.size);
    std::vector<bool> vec_bool = errorMask;
    memcpy(&vec[0], errors.data, sizeof(float) * errors.size);
    result.val = ptr->selectBestClassifier(vec_bool, importance, vec);
    result.array = FloatArray(vec);
    return result;
}

extern "C"
struct BoolArray BaseClassifier_trainClassifier(
        struct BaseClassifierPtr ptr, struct TensorWrapper image,
        int target, float importance, struct BoolArray errorMask)
{
    std::vector<bool> vec = errorMask;
    ptr->trainClassifier(image.toMat(), target, importance, vec);
    return BoolArray(vec);
}


//EstimatedGaussDistribution


extern "C"
struct EstimatedGaussDistributionPtr EstimatedGaussDistribution()
{
    return new cv::EstimatedGaussDistribution();
}

extern "C"
struct EstimatedGaussDistributionPtr EstimatedGaussDistribution_ctor2(
        float P_mean, float R_mean, float P_sigma, float R_sigma)
{
    return new cv::EstimatedGaussDistribution(P_mean, R_mean, P_sigma, R_sigma);
}

extern "C"
void EstimatedGaussDistribution_dtor(
        struct EstimatedGaussDistributionPtr ptr)
{
    delete static_cast<cv::EstimatedGaussDistribution *>(ptr.ptr);
}

extern "C"
float EstimatedGaussDistribution_getMean(
        struct EstimatedGaussDistributionPtr ptr)
{
    return ptr->getMean();
}

extern "C"
float EstimatedGaussDistribution_getSigma(
        struct EstimatedGaussDistributionPtr ptr)
{
    ptr->getSigma();
}

extern "C"
void EstimatedGaussDistribution_setValues(
        struct EstimatedGaussDistributionPtr ptr, float mean, float sigma)
{
    ptr->setValues(mean, sigma);
}

extern "C"
void EstimatedGaussDistribution_update(
        struct EstimatedGaussDistributionPtr ptr, float value)
{
    ptr->update(value);
}

//ClassifierThreshold

extern "C"
struct ClassifierThresholdPtr ClassifierThreshold_ctor(
        struct EstimatedGaussDistributionPtr posSamples,
        struct EstimatedGaussDistributionPtr negSamples)
{
    return new cv::ClassifierThreshold(
            static_cast<cv::EstimatedGaussDistribution *>(posSamples.ptr),
            static_cast<cv::EstimatedGaussDistribution *>(negSamples.ptr));
}

extern "C"
void ClassifierThreshold_dtor(
        struct ClassifierThresholdPtr ptr)
{
    delete static_cast<cv::ClassifierThreshold *>(ptr.ptr);
}

extern "C"
int ClassifierThreshold_eval(
        struct ClassifierThresholdPtr ptr, float value)
{
    return ptr->eval(value);
}

extern "C"
void ClassifierThreshold_update(
        struct ClassifierThresholdPtr ptr,
        float value, int target)
{
    ptr->update(value, target);
}

//ClfMilBoost

extern "C"
struct ClfMilBoostPtr ClfMilBoost_ctor()
{
    return new cv::ClfMilBoost();
}

extern "C"
void ClfMilBoost_dtor(
        struct ClfMilBoostPtr ptr)
{
    delete static_cast<cv::ClfMilBoost *>(ptr.ptr);
}

extern "C"
struct FloatArray ClfMilBoost_classify(
        struct ClfMilBoostPtr ptr, struct TensorWrapper x, bool logR)
{
    return FloatArray(ptr->classify(x.toMat(), log));
}

extern "C"
void ClfMilBoost_init(
        struct ClfMilBoostPtr ptr, struct Params parameters)
{
    cv::ClfMilBoost::Params* p_par = reinterpret_cast<cv::ClfMilBoost::Params *>(&parameters);
    ptr->init(*p_par);
}

extern "C"
float ClfMilBoost_sigmoid(
        struct ClfMilBoostPtr ptr, float x)
{
    return ptr->sigmoid(x);
}

extern "C"
void ClfMilBoost_update(
        struct ClfMilBoostPtr ptr, struct TensorWrapper posx,
        struct TensorWrapper negx)
{
    ptr->update(posx.toMat(), negx.toMat());
}

//ClfOnlineStump

extern "C"
struct ClfOnlineStumpPtr ClfOnlineStump_ctor()
{
    return new cv::ClfOnlineStump();
}

extern "C"
void ClfOnlineStump_dtor(
        struct ClfOnlineStumpPtr ptr)
{
    delete static_cast<cv::ClfOnlineStump *>(ptr.ptr);
}

extern "C"
bool ClfOnlineStump_classify(
        struct ClfOnlineStumpPtr ptr, struct TensorWrapper x, int i)
{
    return ptr->classify(x.toMat(), i);
}

extern "C"
float ClfOnlineStump_classifyF(
        struct ClfOnlineStumpPtr ptr, struct TensorWrapper x, int i)
{
    return ptr->classifyF(x.toMat(), i);
}

extern "C"
struct FloatArray ClfOnlineStump_classifySetF(
        struct ClfOnlineStumpPtr ptr, struct TensorWrapper x)
{
    return FloatArray(ptr->classifySetF(x.toMat()));
}

extern "C"
void ClfOnlineStump_init(
        struct ClfOnlineStumpPtr ptr)
{
    ptr->init();
}

extern "C"
void ClfOnlineStump_update(
        struct ClfOnlineStumpPtr ptr, struct TensorWrapper posx,
        struct TensorWrapper negx)
{
    ptr->update(posx.toMat(), negx.toMat());
}

//CvParams

extern "C"
void CvParams_dtor(
        struct CvParamsPtr ptr)
{
    delete static_cast<cv::CvParams *>(ptr.ptr);
}

extern "C"
void CvParams_printAttrs(
        struct CvParamsPtr ptr)
{
    ptr->printAttrs();
}

extern "C"
void CvParams_printDefaults(
        struct CvParamsPtr ptr)
{
    ptr->printDefaults();
}

extern "C"
bool CvParams_read(
        struct CvParamsPtr ptr, struct FileNodePtr node)
{
    return ptr->read(*node);
}

extern "C"
bool CvParams_scanAttr(
        struct CvParamsPtr ptr, const char* prmName, const char* val)
{
    ptr->scanAttr(prmName, val);
}

extern "C"
void CvParams_write(
        struct CvParamsPtr ptr, struct FileStoragePtr fs)
{
    ptr->write(*fs);
}

//CvFeatureParams

extern "C"
struct CvFeatureParamsPtr CvFeatureParams_ctor(
        int featureType)
{
    return rescueObjectFromPtr(cv::CvFeatureParams::create(featureType));
}

extern "C"
void CvFeatureParams_init(
        struct CvFeatureParamsPtr ptr, struct CvFeatureParamsPtr fp)
{
    ptr->init(*static_cast<cv::CvFeatureParams *>(fp.ptr));
}

bool CvFeatureParams_read(
        struct CvFeatureParamsPtr ptr, struct FileNodePtr node)
{
    return ptr->read(*node);
}

extern "C"
void CvFeatureParams_write(
        struct CvFeatureParamsPtr ptr, struct FileStoragePtr fs)
{
    ptr->write(*fs);
}

//CvHaarFeatureParams

extern "C"
struct CvHaarFeatureParamsPtr CvHaarFeatureParams_ctor()
{
    return new cv::CvHaarFeatureParams();
}

extern "C"
void CvHaarFeatureParams_init(
        struct CvHaarFeatureParamsPtr ptr, struct CvFeatureParamsPtr fp)
{
    ptr->init(*static_cast<cv::CvFeatureParams *>(fp.ptr));
}

extern "C"
void CvHaarFeatureParams_printAttrs(
        struct CvHaarFeatureParamsPtr ptr)
{
    ptr->printAttrs();
}

extern "C"
void CvHaarFeatureParams_printDefaults(
        struct CvHaarFeatureParamsPtr ptr)
{
    ptr->printDefaults();
}

extern "C"
bool CvHaarFeatureParams_read(
        struct CvHaarFeatureParamsPtr ptr, struct FileNodePtr node)
{
    return ptr->read(*node);
}

extern "C"
bool CvHaarFeatureParams_scanAttr(
        struct CvHaarFeatureParamsPtr ptr, const char* prm, const char* val)
{
    ptr->scanAttr(prm, val);
}

extern "C"
void CvHaarFeatureParams_write(
        struct CvHaarFeatureParamsPtr ptr, struct FileStoragePtr fs)
{
    ptr->write(*fs);
}

//CvHOGFeatureParams

extern "C"
struct CvHOGFeatureParamsPtr CvHOGFeatureParams_ctor()
{
    return new cv::CvHOGFeatureParams();
}

//CvLBPFeatureParams

extern "C"
struct CvLBPFeatureParamsPtr CvLBPFeatureParams_ctor()
{
    return new cv::CvLBPFeatureParams();
}

//CvFeatureEvaluator

extern "C"
struct CvFeatureEvaluatorPtr CvFeatureEvaluator_ctor(
        int type)
{
    return rescueObjectFromPtr(cv::CvFeatureEvaluator::create(type));
}

extern "C"
void CvFeatureEvaluator_dtor(
        struct CvFeatureEvaluatorPtr ptr)
{
    delete static_cast<cv::CvFeatureEvaluator *>(ptr.ptr);
}

extern "C"
struct TensorWrapper CvFeatureEvaluator_getCls(
        struct CvFeatureEvaluatorPtr ptr)
{
    cv::Mat mat = ptr->getCls();
    return TensorWrapper(MatT(mat));
}

extern "C"
float CvFeatureEvaluator_getCls2(
        struct CvFeatureEvaluatorPtr ptr, int si)
{
    return ptr->getCls(si);
}

extern "C"
int CvFeatureEvaluator_getFeatureSize(
        struct CvFeatureEvaluatorPtr ptr)
{
    return ptr->getFeatureSize();
}

extern "C"
int CvFeatureEvaluator_getMaxCatCount(
        struct CvFeatureEvaluatorPtr ptr)
{
    return ptr->getMaxCatCount();
}

extern "C"
int CvFeatureEvaluator_getNumFeatures(
        struct CvFeatureEvaluatorPtr ptr)
{
    ptr->getNumFeatures();
}

extern "C"
void CvFeatureEvaluator_init(
        struct CvFeatureEvaluatorPtr ptr, struct CvFeatureParamsPtr _featureParams,
        int _maxSampleCount, struct SizeWrapper _winSize)
{
    ptr->init(static_cast<cv::CvFeatureParams *>(ptr.ptr), _maxSampleCount, _winSize);
}

extern "C"
float CvFeatureEvaluator_call(
        struct CvFeatureEvaluatorPtr ptr, int featureIdx, int sampleIdx)
{
    ptr->operator()(featureIdx, sampleIdx);
}

extern "C"
void CvFeatureEvaluator_setImage(
        struct CvFeatureEvaluatorPtr ptr, struct TensorWrapper img,
        unsigned char clsLabel, int idx)
{
    ptr->setImage(img.toMat(), clsLabel, idx);
}

extern "C"
void CvFeatureEvaluator_writeFeatures(
        struct CvFeatureEvaluatorPtr ptr, struct FileStoragePtr fs,
        struct TensorWrapper featureMap)
{
    ptr->writeFeatures(*fs, featureMap.toMat());
}

//FeatureHaar

extern "C"
struct FeatureHaarPtr FeatureHaar_ctor(
        struct SizeWrapper patchSize)
{
    return new cv::CvHaarEvaluator::FeatureHaar(patchSize);
}

extern "C"
void FeatureHaar_dtor(
        struct FeatureHaarPtr ptr)
{
    delete static_cast<cv::CvHaarEvaluator::FeatureHaar *>(ptr.ptr);
}

extern "C"
struct FloatPlusBool FeatureHaar_eval(
        struct FeatureHaarPtr ptr, struct TensorWrapper image, struct RectWrapper ROI)
{
    struct FloatPlusBool result;
    result.v1 = ptr->eval(image.toMat(), ROI, &result.v2);
    return result;
}

extern "C"
struct RectArray FeatureHaar_getAreas(
        struct FeatureHaarPtr ptr)
{
    return RectArray(ptr->getAreas());
}

extern "C"
float FeatureHaar_getInitMean(
        struct FeatureHaarPtr ptr)
{
    return ptr->getInitMean();
}

extern "C"
float FeatureHaar_getInitSigma(
        struct FeatureHaarPtr ptr)
{
    return ptr->getInitSigma();
}

extern "C"
int FeatureHaar_getNumAreas(
        struct FeatureHaarPtr ptr)
{
    return ptr->getNumAreas();
}

extern "C"
struct FloatArray FeatureHaar_getWeights(
        struct FeatureHaarPtr ptr)
{
    return FloatArray(ptr->getWeights());
}

//CvHaarEvaluator

extern "C"
struct CvHaarEvaluatorPtr CvHaarEvaluator_ctor()
{
    //TODO undefined symbol: _ZTVN2cv15CvHaarEvaluatorE
    //return new cv::CvHaarEvaluator();
}

extern "C"
void CvHaarEvaluator_dtor(
        struct CvHaarEvaluatorPtr ptr)
{
    delete static_cast<cv::CvHaarEvaluator *>(ptr.ptr);
}

extern "C"
void CvHaarEvaluator_generateFeatures(
        struct CvHaarEvaluatorPtr ptr)
{
    ptr->generateFeatures();
}

extern "C"
void CvHaarEvaluator_generateFeatures2(
        struct CvHaarEvaluatorPtr ptr, int numFeatures)
{
    ptr->generateFeatures(numFeatures);
}

extern "C"
void CvHaarEvaluator_init(
        struct CvHaarEvaluatorPtr ptr, struct CvFeatureParamsPtr _featureParams,
        int _maxSampleCount, struct SizeWrapper _winSize)
{
    ptr->init(static_cast<cv::CvFeatureParams *>(ptr.ptr), _maxSampleCount, _winSize);
}

extern "C"
float CvHaarEvaluator_call(
        struct CvHaarEvaluatorPtr ptr, int featureIdx, int sampleIdx)
{
    ptr->operator()(featureIdx, sampleIdx);
}

extern "C"
void CvHaarEvaluator_setImage(
        struct CvHaarEvaluatorPtr ptr, struct TensorWrapper img,
        unsigned char clsLabel, int idx)
{
    ptr->setImage(img.toMat(), clsLabel, idx);
}

extern "C"
void CvHaarEvaluator_setWinSize(
        struct CvHaarEvaluatorPtr ptr, struct SizeWrapper patchSize)
{
    ptr->setWinSize(patchSize);
}

extern "C"
struct SizeWrapper CvHaarEvaluator_setWinSize2(
        struct CvHaarEvaluatorPtr ptr)
{
    return SizeWrapper(ptr->setWinSize());
}

extern "C"
void CvHaarEvaluator_writeFeature(
        struct CvHaarEvaluatorPtr ptr, struct FileStoragePtr fs)
{
    ptr->writeFeature(*fs);
}

extern "C"
void CvHaarEvaluator_writeFeatures(
        struct CvHaarEvaluatorPtr ptr, struct FileStoragePtr fs,
        struct TensorWrapper featureMap)
{
    ptr->writeFeatures(*fs, featureMap.toMat());
}

//CvHOGEvaluator

extern "C"
struct CvHOGEvaluatorPtr CvHOGEvaluator_ctor()
{
//TODO undefined symbol: _ZTVN2cv14CvHOGEvaluatorE
//    return new cv::CvHOGEvaluator();
}

extern "C"
void CvHOGEvaluator_dtor(
        struct CvHOGEvaluatorPtr ptr)
{
    delete static_cast<cv::CvHOGEvaluator *>(ptr.ptr);
}

extern "C"
void CvHOGEvaluator_init(
        struct CvHOGEvaluatorPtr ptr, struct CvFeatureParamsPtr _featureParams,
        int _maxSampleCount, struct SizeWrapper _winSize)
{
    ptr->init(static_cast<cv::CvFeatureParams *>(ptr.ptr), _maxSampleCount, _winSize);
}

extern "C"
float CvHOGEvaluator_call(
        struct CvHOGEvaluatorPtr ptr, int featureIdx, int sampleIdx)
{
    ptr->operator()(featureIdx, sampleIdx);
}

extern "C"
void CvHOGEvaluator_setImage(
        struct CvHOGEvaluatorPtr ptr, struct TensorWrapper img,
        unsigned char clsLabel, int idx)
{
    ptr->setImage(img.toMat(), clsLabel, idx);
}

extern "C"
void CvHOGEvaluator_writeFeatures(
        struct CvHOGEvaluatorPtr ptr, struct FileStoragePtr fs,
        struct TensorWrapper featureMap)
{
    ptr->writeFeatures(*fs, featureMap.toMat());
}

//CvLBPEvaluator

extern "C"
struct CvLBPEvaluatorPtr CvLBPEvaluator_ctor()
{
//TODO undefined symbol
//    return new cv::CvLBPEvaluator();
}

extern "C"
void CvLBPEvaluator_dtor(
        struct CvLBPEvaluatorPtr ptr)
{
    delete static_cast<cv::CvLBPEvaluator *>(ptr.ptr);
}

extern "C"
void CvLBPEvaluator_init(
        struct CvLBPEvaluatorPtr ptr, struct CvFeatureParamsPtr _featureParams,
        int _maxSampleCount, struct SizeWrapper _winSize)
{
    ptr->init(static_cast<cv::CvFeatureParams *>(ptr.ptr), _maxSampleCount, _winSize);
}

extern "C"
float CvLBPEvaluator_call(
        struct CvLBPEvaluatorPtr ptr, int featureIdx, int sampleIdx)
{
    ptr->operator()(featureIdx, sampleIdx);
}

extern "C"
void CvLBPEvaluator_setImage(
        struct CvLBPEvaluatorPtr ptr, struct TensorWrapper img,
        unsigned char clsLabel, int idx)
{
    ptr->setImage(img.toMat(), clsLabel, idx);
}

extern "C"
void CvLBPEvaluator_writeFeatures(
        struct CvLBPEvaluatorPtr ptr, struct FileStoragePtr fs,
        struct TensorWrapper featureMap)
{
    ptr->writeFeatures(*fs, featureMap.toMat());
}

extern "C"
struct StrongClassifierDirectSelectionPtr StrongClassifierDirectSelection_ctor(
        int numBaseClf, int numWeakClf, struct SizeWrapper patchSz,
        struct RectWrapper sampleROI, bool useFeatureEx, int iterationInit)
{
    return new cv::StrongClassifierDirectSelection(
                        numBaseClf, numWeakClf, patchSz,
                        sampleROI, useFeatureEx, iterationInit);
}

extern "C"
void StrongClassifierDirectSelection_dtor(
        struct StrongClassifierDirectSelectionPtr ptr)
{
    delete static_cast<cv::StrongClassifierDirectSelection *>(ptr.ptr);
}

extern "C"
struct FloatPlusInt StrongClassifierDirectSelection_classifySmooth(
        struct StrongClassifierDirectSelectionPtr ptr, struct TensorArray images,
        struct RectWrapper sampleROI)
{
    struct FloatPlusInt result;
    result.v1 = ptr->classifySmooth(images.toMatList(), sampleROI, result.v2);
    return result;
}

extern "C"
float StrongClassifierDirectSelection_eval(
        struct StrongClassifierDirectSelectionPtr ptr, struct TensorWrapper response)
{
    return ptr->eval(response.toMat());
}

extern "C"
int StrongClassifierDirectSelection_getNumBaseClassifier(
        struct StrongClassifierDirectSelectionPtr ptr)
{
    return ptr->getNumBaseClassifier();
}

extern "C"
struct SizeWrapper StrongClassifierDirectSelection_getPatchSize(
        struct StrongClassifierDirectSelectionPtr ptr)
{
    return SizeWrapper(ptr->getPatchSize());
}

extern "C"
int StrongClassifierDirectSelection_getReplacedClassifier(
        struct StrongClassifierDirectSelectionPtr ptr)
{
    return ptr->getReplacedClassifier();
}

extern "C"
struct RectWrapper StrongClassifierDirectSelection_getROI(
        struct StrongClassifierDirectSelectionPtr ptr)
{
    return RectWrapper(ptr->getROI());
}

extern "C"
struct IntArray StrongClassifierDirectSelection_getSelectedWeakClassifier(
        struct StrongClassifierDirectSelectionPtr ptr)
{
    return IntArray(ptr->getSelectedWeakClassifier());
}

extern "C"
int StrongClassifierDirectSelection_getSwappedClassifier(
        struct StrongClassifierDirectSelectionPtr ptr)
{
    return ptr->getSwappedClassifier();
}

extern "C"
bool StrongClassifierDirectSelection_getUseFeatureExchange(
        struct StrongClassifierDirectSelectionPtr ptr)
{
    return ptr->getUseFeatureExchange();
}

extern "C"
void StrongClassifierDirectSelection_initBaseClassifier(
        struct StrongClassifierDirectSelectionPtr ptr)
{
    ptr->initBaseClassifier();
}

extern "C"
void StrongClassifierDirectSelection_replaceWeakClassifier(
        struct StrongClassifierDirectSelectionPtr ptr, int idx)
{
    ptr->replaceWeakClassifier(idx);
}

extern "C"
bool StrongClassifierDirectSelection_update(
        struct StrongClassifierDirectSelectionPtr ptr, struct TensorWrapper image,
        int target, float importance)
{
    return ptr->update(image.toMat(), target, importance);
}

extern "C"
struct DetectorPtr Detector_ctor(
        struct StrongClassifierDirectSelectionPtr ptr)
{
    return new cv::Detector(static_cast<cv::StrongClassifierDirectSelection *>(ptr.ptr));
}

extern "C"
void Detector_dtor(
        struct DetectorPtr ptr)
{
    delete static_cast<cv::Detector *>(ptr.ptr);
}

extern "C"
void Detector_classifySmooth(
        struct DetectorPtr ptr, struct TensorArray image, float minMargin)
{
    ptr->classifySmooth(image.toMatList(), minMargin);
}

extern "C"
float Detector_getConfidence(
        struct DetectorPtr ptr, int patchIdx)
{
    return ptr->getConfidence(patchIdx);
}

extern "C"
float Detector_getConfidenceOfBestDetection(
        struct DetectorPtr ptr)
{
    return ptr->getConfidenceOfBestDetection();
}

extern "C"
float Detector_getConfidenceOfDetection(
        struct DetectorPtr ptr, int detectionIdx)
{
    return ptr->getConfidenceOfDetection(detectionIdx);
}

extern "C"
struct FloatArray Detector_getConfidences(
        struct DetectorPtr ptr)
{
    return FloatArray(ptr->getConfidences());
}

extern "C"
struct TensorWrapper Detector_getConfImageDisplay(
        struct DetectorPtr ptr)
{
    cv::Mat mat = ptr->getConfImageDisplay();
    return TensorWrapper(MatT(mat));
}

extern "C"
struct IntArray Detector_getIdxDetections(
        struct DetectorPtr ptr)
{
    ptr->getIdxDetections();
}

extern "C"
int Detector_getNumDetections(
        struct DetectorPtr ptr)
{
    return ptr->getNumDetections();
}

extern "C"
int Detector_getPatchIdxOfDetection(
        struct DetectorPtr ptr, int detectionIdx)
{
    return ptr->getPatchIdxOfDetection(detectionIdx);
}

//MultiTracker

extern "C"
struct MultiTrackerPtr MultiTracker_ctor(
        const char *trackerType)
{
    return new cv::MultiTracker(trackerType);
}

extern "C"
void MultiTracker_dtor(
        struct MultiTrackerPtr ptr)
{
    delete static_cast<cv::MultiTracker *>(ptr.ptr);
}

extern "C"
bool MultiTracker_add(
        struct MultiTrackerPtr ptr, struct TensorWrapper image,
        struct Rect2dWrapper boundingBox)
{
    cv::Mat img = cv::imread("/home/ainur/torch-opencv/demo/data/lena.jpg");
    return ptr->add(img, cv::Rect2d(2,2,20,20));
}

extern "C"
bool MultiTracker_add2(
        struct MultiTrackerPtr ptr, const char *trackerType,
        struct TensorWrapper image, struct Rect2dWrapper boundingBox)
{
    return ptr->add(trackerType, image.toMat(), boundingBox);
}

extern "C"
bool MultiTracker_add3(
        struct MultiTrackerPtr ptr, const char *trackerType,
        struct TensorWrapper image, struct Rect2dArray boundingBox)
{
    return ptr->add(trackerType, image.toMat(), boundingBox);
}

extern "C"
bool MultiTracker_add4(
        struct MultiTrackerPtr ptr, struct TensorWrapper image,
        struct Rect2dArray boundingBox)
{
    return ptr->add(image.toMat(), boundingBox);
}

extern "C"
bool MultiTracker_update(
        struct MultiTrackerPtr ptr, struct TensorWrapper image)
{
    return ptr->update(image.toMat());
}

extern "C"
struct Rect2dArrayPlusBool MultiTracker_update2(
        struct MultiTrackerPtr ptr, struct TensorWrapper image)
{
    struct Rect2dArrayPlusBool result;
    std::vector<cv::Rect2d> vec;
    result.val = ptr->update(image.toMat(), vec);
    new(&result.rect2d) Rect2dArray(vec);
    return result;
}

//MultiTracker_Alt

extern "C"
struct MultiTracker_AltPtr MultiTracker_Alt_ctor()
{
    return new cv::MultiTracker_Alt();
}

extern "C"
void MultiTracker_Alt_dtor(
        struct MultiTracker_AltPtr ptr)
{
    delete static_cast<cv::MultiTracker_Alt *>(ptr.ptr);
}

extern "C"
bool MultiTracker_Alt_addTarget(
        struct MultiTracker_AltPtr ptr, struct TensorWrapper image,
        struct Rect2dWrapper boundingBox, const char *tracker_algorithm_name)
{
    return ptr->addTarget(image.toMat(), boundingBox, tracker_algorithm_name);
}

extern "C"
bool MultiTracker_Alt_update(
        struct MultiTracker_AltPtr ptr, struct TensorWrapper image)
{
    return ptr->update(image.toMat());
}

extern "C"
extern "C"
struct MultiTrackerTLDPtr MultiTrackerTLD_ctor()
{
    return new cv::MultiTrackerTLD();
}

extern "C"
void MultiTrackerTLD_dtor(
        struct MultiTrackerTLDPtr ptr)
{
    delete static_cast<cv::MultiTrackerTLD *>(ptr.ptr);
}

extern "C"
bool MultiTrackerTLD_update_opt(
        struct MultiTrackerTLDPtr ptr, struct TensorWrapper image)
{
    ptr->update_opt(image.toMat());
}

//ROISelector

extern "C"
struct ROISelectorPtr ROISelector_ctor()
{
    return new cv::ROISelector();
}

extern "C"
void ROISelector_dtor(
        struct ROISelectorPtr ptr)
{
    delete static_cast<cv::ROISelector *>(ptr.ptr);
}

extern "C"
struct Rect2dWrapper ROISelector_select(
        struct ROISelectorPtr ptr, struct TensorWrapper image, bool fromCenter)
{
    return Rect2dWrapper(ptr->select(image.toMat(), fromCenter));
}

extern "C"
struct Rect2dWrapper ROISelector_select2(
        struct ROISelectorPtr ptr, const char *windowName,
        struct TensorWrapper img, bool showCrossair, bool fromCenter)
{
    return Rect2dWrapper(ptr->select(windowName, img.toMat(), showCrossair, fromCenter));
}

extern "C"
void ROISelector_select3(
        struct ROISelectorPtr ptr, const char *windowName, struct TensorWrapper img,
        struct Rect2dArray boundingBox, bool fromCenter)
{
    std::vector<cv::Rect2d> vec = boundingBox;
    return ptr->select(windowName, img.toMat(), vec, fromCenter);
}

//TrackerTargetState

extern "C"
struct TrackerTargetStatePtr TrackerTargetState_ctor()
{
    return new cv::TrackerTargetState();
}

extern "C"
void TrackerTargetState_dtor(
        struct TrackerTargetStatePtr ptr)
{
    delete static_cast<cv::TrackerTargetState *>(ptr.ptr);
}

extern "C"
int TrackerTargetState_getTargetHeight(
        struct TrackerTargetStatePtr ptr)
{
    return ptr->getTargetHeight();
}

extern "C"
struct Point2fWrapper TrackerTargetState_getTargetPosition(
        struct TrackerTargetStatePtr ptr)
{
    return Point2fWrapper(ptr->getTargetPosition());
}

extern "C"
int TrackerTargetState_getTargetWidth(
        struct TrackerTargetStatePtr ptr)
{
    return ptr->getTargetWidth();
}

extern "C"
void TrackerTargetState_setTargetHeight(
        struct TrackerTargetStatePtr ptr, int height)
{
    ptr->setTargetHeight(height);
}

extern "C"
void TrackerTargetState_setTargetPosition(
        struct TrackerTargetStatePtr ptr, struct Point2fWrapper position)
{
    ptr->setTargetPosition(position);
}

extern "C"
void TrackerTargetState_setTargetWidth(
        struct TrackerTargetStatePtr ptr, int width)
{
    ptr->setTargetWidth(width);
}

//TrackerStateEstimator

extern "C"
struct TrackerStateEstimatorPtr TrackerStateEstimator_ctor(
        const char *trackeStateEstimatorType)
{
    return rescueObjectFromPtr(cv::TrackerStateEstimator::create(trackeStateEstimatorType));
}

extern "C"
void TrackerStateEstimator_dtor(
        struct TrackerStateEstimatorPtr ptr)
{
    delete static_cast<cv::TrackerStateEstimator *>(ptr.ptr);
}

extern "C"
struct TrackerTargetStatePtr TrackerStateEstimator_estimate(
        struct TrackerStateEstimatorPtr ptr, struct ConfidenceMapArray confidenceMaps)
{
    std::vector<std::vector<std::pair<cv::Ptr<cv::TrackerTargetState>, float>>> vec = confidenceMaps;
    return rescueObjectFromPtr(ptr->estimate(vec));
}

extern "C"
const char* TrackerStateEstimator_getClassName(
        struct TrackerStateEstimatorPtr ptr)
{
    cv::String str = ptr->getClassName();
    return str.c_str();
}

extern "C"
void TrackerStateEstimator_update(
        struct TrackerStateEstimatorPtr ptr, struct ConfidenceMapArray confidenceMaps)
{
    std::vector<std::vector<std::pair<cv::Ptr<cv::TrackerTargetState>, float>>> vec = confidenceMaps;
    ptr->update(vec);
}

//TrackerStateEstimatorAdaBoosting

extern "C"
struct TrackerStateEstimatorAdaBoostingPtr TrackerStateEstimatorAdaBoosting_ctor(
        int numClassifer, int initIterations, int nFeatures,
        struct SizeWrapper patchSize, struct RectWrapper ROI)
{
    return new cv::TrackerStateEstimatorAdaBoosting(
                        numClassifer, initIterations, nFeatures, patchSize, ROI);
}

extern "C"
void TrackerStateEstimatorAdaBoosting_dtor(
        struct TrackerStateEstimatorAdaBoostingPtr ptr)
{
    delete static_cast<cv::TrackerStateEstimatorAdaBoosting *>(ptr.ptr);
}

extern "C"
struct IntArray TrackerStateEstimatorAdaBoosting_computeReplacedClassifier(
        struct TrackerStateEstimatorAdaBoostingPtr ptr)
{
    return IntArray(ptr->computeReplacedClassifier());
}

extern "C"
struct IntArray TrackerStateEstimatorAdaBoosting_computeSelectedWeakClassifier(
        struct TrackerStateEstimatorAdaBoostingPtr ptr)
{
    return IntArray(ptr->computeSelectedWeakClassifier());
}

extern "C"
struct IntArray TrackerStateEstimatorAdaBoosting_computeSwappedClassifier(
        struct TrackerStateEstimatorAdaBoostingPtr ptr)
{
    return IntArray(ptr->computeSwappedClassifier());
}

extern "C"
void TrackerStateEstimatorAdaBoosting_setCurrentConfidenceMap(
        struct TrackerStateEstimatorAdaBoostingPtr ptr, struct ConfidenceMap confidenceMap)
{
    std::vector<std::pair<cv::Ptr<cv::TrackerTargetState>, float>> vec = confidenceMap;
    ptr->setCurrentConfidenceMap(vec);
}

extern "C"
struct RectWrapper TrackerStateEstimatorAdaBoosting_getSampleROI(
        struct TrackerStateEstimatorAdaBoostingPtr ptr)
{
    return RectWrapper(ptr->getSampleROI());
}

extern "C"
void TrackerStateEstimatorAdaBoosting_setSampleROI(
        struct TrackerStateEstimatorAdaBoostingPtr ptr, struct RectWrapper ROI)
{
    ptr->setSampleROI(ROI);
}

//TrackerStateEstimatorMILBoosting

extern "C"
struct TrackerStateEstimatorMILBoostingPtr TrackerStateEstimatorMILBoosting_ctor(
        int nFeatures)
{
    return new cv::TrackerStateEstimatorMILBoosting(nFeatures);
}

extern "C"
void TrackerStateEstimatorMILBoosting_dtor(
        struct TrackerStateEstimatorMILBoostingPtr ptr)
{
    delete static_cast<cv::TrackerStateEstimatorMILBoosting *>(ptr.ptr);
}

extern "C"
void TrackerStateEstimatorMILBoosting_setCurrentConfidenceMap(
        struct TrackerStateEstimatorMILBoostingPtr ptr, struct ConfidenceMap confidenceMap)
{
    std::vector<std::pair<cv::Ptr<cv::TrackerTargetState>, float>> vec = confidenceMap;
    ptr->setCurrentConfidenceMap(vec);
}

//TrackerStateEstimatorSVM

extern "C"
struct TrackerStateEstimatorSVMPtr TrackerStateEstimatorSVM_ctor()
{
    return new cv::TrackerStateEstimatorSVM();
}

extern "C"
void TrackerStateEstimatorSVM_dtor(
        struct TrackerStateEstimatorSVMPtr ptr)
{
    delete static_cast<cv::TrackerStateEstimatorSVM *>(ptr.ptr);
}

//TrackerModel

extern "C"
void TrackerModel_dtor(
        struct TrackerModelPtr ptr);

extern "C"
struct ConfidenceMapArray TrackerModel_getConfidenceMaps(
        struct TrackerModelPtr ptr)
{
    return ConfidenceMapArray(ptr->getConfidenceMaps());
}

extern "C"
struct ConfidenceMap TrackerModel_getLastConfidenceMap(
        struct TrackerModelPtr ptr)
{
    return ConfidenceMap(ptr->getLastConfidenceMap());
}

extern "C"
struct TrackerTargetStatePtr TrackerModel_getLastTargetState(
        struct TrackerModelPtr ptr)
{
    return rescueObjectFromPtr(ptr->getLastTargetState());
}

extern "C"
struct TrackerStateEstimatorPtr TrackerModel_getTrackerStateEstimator(
        struct TrackerModelPtr ptr)
{
    return rescueObjectFromPtr(ptr->getTrackerStateEstimator());
}

extern "C"
void TrackerModel_modelEstimation(
        struct TrackerModelPtr ptr, struct TensorArray responses)
{
    ptr->modelEstimation(responses.toMatList());
}

extern "C"
void TrackerModel_modelUpdate(
        struct TrackerModelPtr ptr)
{
    ptr->modelUpdate();
}

extern "C"
bool TrackerModel_runStateEstimator(
        struct TrackerModelPtr ptr)
{
    return ptr->runStateEstimator();
}

extern "C"
void TrackerModel_setLastTargetState(
        struct TrackerModelPtr ptr, struct TrackerTargetStatePtr lastTargetState)
{
    ptr->setLastTargetState(static_cast<cv::TrackerTargetState *>(lastTargetState.ptr));
}

extern "C"
bool TrackerModel_setTrackerStateEstimator(
        struct TrackerModelPtr ptr, struct TrackerStateEstimatorPtr trackerStateEstimator)
{
    return ptr->setTrackerStateEstimator(static_cast<cv::TrackerStateEstimator *>(trackerStateEstimator.ptr));
}

//Tracker

extern "C"
void Tracker_dtor(
        struct TrackerPtr ptr)
{
    delete static_cast<cv::Tracker *>(ptr.ptr);
}

extern "C"
struct TrackerModelPtr Tracker_getModel(
        struct TrackerPtr ptr)
{
    return rescueObjectFromPtr(ptr->getModel());
}

extern "C"
bool Tracker_init(
        struct TrackerPtr ptr, struct TensorWrapper image, struct Rect2dWrapper boundingBox)
{
    return ptr->init(image.toMat(), boundingBox);
}

extern "C"
void Tracker_read(
        struct TrackerPtr ptr, struct FileNodePtr fn)
{
    ptr->read(*fn);
}

extern "C"
bool Tracker_update(
        struct TrackerPtr ptr, struct TensorWrapper image, struct Rect2dWrapper boundingBox)
{
    cv::Rect2d rect = boundingBox;
    return ptr->update(image.toMat(), rect);
}

extern "C"
void Tracker_write(
        struct TrackerPtr ptr, struct FileStoragePtr fn)
{

    ptr->write(*fn);
}

//TrackerBoosting

extern "C"
struct TrackerBoostingPtr TrackerBoosting_ctor(
        struct TrackerBoosting_Params parameters)
{
    cv::TrackerBoosting::Params p;
    p.featureSetNumFeatures = parameters.featureSetNumFeatures;
    p.iterationInit = parameters.iterationInit;
    p.numClassifiers = parameters.numClassifiers;
    p.samplerOverlap = parameters.samplerOverlap;
    p.samplerSearchFactor = parameters.samplerSearchFactor;
    return rescueObjectFromPtr(cv::TrackerBoosting::createTracker(p));
}

extern "C"
void TrackerBoosting_dtor(
        struct TrackerBoostingPtr ptr)
{
    delete static_cast<cv::TrackerBoosting *>(ptr.ptr);
}

//TrackerKCF

extern "C"
struct TrackerKCFPtr TrackerKCF_ctor(
        struct TrackerKCF_Params parameters)
{
    cv::TrackerKCF::Params p;
    p.sigma = parameters.sigma;
    p.lambda = parameters.lambda;
    p.interp_factor = parameters.interp_factor;
    p.output_sigma_factor = parameters.output_sigma_factor;
    p.pca_learning_rate = parameters.pca_learning_rate;
    p.resize = parameters.resize;
    p.split_coeff = parameters.split_coeff;
    p.wrap_kernel = parameters.wrap_kernel;
    p.compress_feature = parameters.compress_feature;
    p.max_patch_size = parameters.max_patch_size;
    p.compressed_size = parameters.compressed_size;
    p.desc_pca = parameters.desc_pca;
    p.desc_npca = parameters.desc_npca;
    return rescueObjectFromPtr(cv::TrackerKCF::createTracker(p));
}

extern "C"
void TrackerKCF_dtor(
        struct TrackerKCFPtr ptr)
{
    delete static_cast<cv::TrackerKCF *>(ptr.ptr);
}

//TrackerMedianFlow

extern "C"
struct TrackerMedianFlowPtr TrackerMedianFlow_ctor(
        struct TrackerMedianFlow_Params parameters)
{
    cv::TrackerMedianFlow::Params p;
    p.pointsInGrid = parameters.pointsInGrid;
    return rescueObjectFromPtr(cv::TrackerMedianFlow::createTracker(p));
}

extern "C"
void TrackerMedianFlow_dtor(
        struct TrackerMedianFlowPtr ptr)
{
    delete static_cast<cv::TrackerMedianFlow *>(ptr.ptr);
}

extern "C"
struct TrackerMILPtr TrackerMIL_ctor(
        struct TrackerMIL_Params parameters)
{
    cv::TrackerMIL::Params p;
    p.samplerInitInRadius = parameters.samplerInitInRadius;
    p.samplerInitMaxNegNum = parameters.samplerInitMaxNegNum;
    p.samplerSearchWinSize = parameters.samplerSearchWinSize;
    p.samplerTrackInRadius = parameters.samplerTrackInRadius;
    p.samplerTrackMaxPosNum = parameters.samplerTrackMaxPosNum;
    p.samplerTrackMaxNegNum = parameters.samplerTrackMaxNegNum;
    p.featureSetNumFeatures = parameters.featureSetNumFeatures;
    return rescueObjectFromPtr(cv::TrackerMIL::createTracker(p));
}

extern "C"
void TrackerMIL_dtor(
        struct TrackerMILPtr ptr)
{
    delete static_cast<cv::TrackerMIL *>(ptr.ptr);
}

//TrackerTLD

extern "C"
struct TrackerTLDPtr TrackerTLD_ctor()
{
    return rescueObjectFromPtr(cv::TrackerTLD::createTracker());
}

extern "C"
void TrackerTLD_dtor(
        struct TrackerTLDPtr ptr)
{
    delete static_cast<cv::TrackerTLD *>(ptr.ptr);
}

//TrackerAdaBoostingTargetState

extern "C"
struct TrackerAdaBoostingTargetStatePtr TrackerAdaBoostingTargetState_ctor(
        struct Point2fWrapper position, int width, int height,
        bool foreground, struct TensorWrapper responses)
{
    return new cv::TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState(
            position, width, height, foreground, responses.toMat());
}

extern "C"
void TrackerAdaBoostingTargetState_dtor(
        struct TrackerAdaBoostingTargetStatePtr ptr)
{
    delete static_cast<cv::TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState *>(ptr.ptr);
}

extern "C"
struct TensorWrapper TrackerAdaBoostingTargetState_getTargetResponses(
        struct TrackerAdaBoostingTargetStatePtr ptr)
{
    TensorWrapper(MatT(ptr->getTargetResponses()));
}

extern "C"
bool TrackerAdaBoostingTargetState_isTargetFg(
        struct TrackerAdaBoostingTargetStatePtr ptr)
{
    return ptr->isTargetFg();;
}

extern "C"
void TrackerAdaBoostingTargetState_setTargetFg(
        struct TrackerAdaBoostingTargetStatePtr ptr, bool foreground)
{
    ptr->setTargetFg(foreground);
}

extern "C"
void TrackerAdaBoostingTargetState_setTargetResponses(
        struct TrackerAdaBoostingTargetStatePtr ptr, struct TensorWrapper responses)
{
    ptr->setTargetResponses(responses.toMat());
}

//TrackerMILTargetState

extern "C"
struct TrackerMILTargetStatePtr TrackerMILTargetState_ctor(
        struct Point2fWrapper position, int width, int height, bool foreground, struct TensorWrapper features)
{
    return new cv::TrackerStateEstimatorMILBoosting::TrackerMILTargetState(
                            position, width, height, foreground, features.toMat());
}

extern "C"
void TrackerMILTargetState_dtor(
        struct TrackerMILTargetStatePtr ptr)
{
    delete static_cast<cv::TrackerStateEstimatorMILBoosting::TrackerMILTargetState *>(ptr.ptr);
}

extern "C"
struct TensorWrapper TrackerMILTargetState_getFeatures(
        struct TrackerMILTargetStatePtr ptr)
{
    return TensorWrapper(MatT(ptr->getFeatures()));
}

extern "C"
bool TrackerMILTargetState_isTargetFg(
        struct TrackerMILTargetStatePtr ptr)
{
    return ptr->isTargetFg();;
}

extern "C"
void TrackerMILTargetState_setFeatures(
        struct TrackerMILTargetStatePtr ptr, struct TensorWrapper features)
{
    ptr->setFeatures(features.toMat());
}

extern "C"
void TrackerMILTargetState_setTargetFg(
        struct TrackerMILTargetStatePtr ptr, bool foreground)
{
    ptr->setTargetFg(foreground);
}

//TrackerFeature

extern "C"
struct TrackerFeaturePtr TrackerFeature_ctor(
        const char *rackerFeatureType)
{
    return rescueObjectFromPtr(cv::TrackerFeature::create(rackerFeatureType));
}

extern "C"
void TrackerFeature_dtor(
        struct TrackerFeaturePtr ptr)
{
    delete static_cast<cv::TrackerFeature *>(ptr.ptr);
}

extern "C"
struct TensorWrapper TrackerFeature_compute(
        struct TrackerFeaturePtr ptr, struct TensorArray images)
{
    MatT response;
    ptr->compute(images.toMatList(), response.mat);
    return TensorWrapper(response);
}

extern "C"
const char* TrackerFeature_getClassName(
        struct TrackerFeaturePtr ptr)
{
    return ptr->getClassName().c_str();
}

extern "C"
void TrackerFeature_selection(
        struct TrackerFeaturePtr ptr, struct TensorWrapper response, int npoints)
{
    cv::Mat mat = response.toMat();
    ptr->selection(mat, npoints);
}

//TrackerFeatureFeature2d

extern "C"
struct TrackerFeatureFeature2dPtr TrackerFeatureFeature2d_ctor(
        const char *detectorType, const char *descriptorType)
{
    return new cv::TrackerFeatureFeature2d(detectorType, descriptorType);
}

extern "C"
void TrackerFeatureFeature2d_dtor(
        struct TrackerFeatureFeature2dPtr ptr)
{
    delete static_cast<cv::TrackerFeatureFeature2d *>(ptr.ptr);
}

extern "C"
void TrackerFeatureFeature2d_selection(
        struct TrackerFeatureFeature2dPtr ptr, struct TensorWrapper response, int npoints)
{
    cv::Mat mat = response.toMat();
    ptr->selection(mat, npoints);
}

//TrackerFeatureHAAR

extern "C"
struct TrackerFeatureHAARPtr TrackerFeatureHAAR_ctor(
        struct TrackerFeatureHAAR_Params parameters)
{
    cv::TrackerFeatureHAAR::Params p;
    p.numFeatures = parameters.numFeatures;
    p.rectSize = parameters.rectSize;
    p.isIntegral = parameters.isIntegral;

    return new cv::TrackerFeatureHAAR(p);
}

extern "C"
void TrackerFeatureHAAR_dtor(
        struct TrackerFeatureHAARPtr ptr)
{
    delete static_cast<cv::TrackerFeatureHAAR *>(ptr.ptr);
}

extern "C"
struct TensorWrapper TrackerFeatureHAAR_extractSelected(
        struct TrackerFeatureHAARPtr ptr, struct IntArray selFeatures,
        struct TensorArray images)
{
    MatT response;
    ptr->extractSelected(selFeatures, images.toMatList(), response.mat);
    return TensorWrapper(response);
}

extern "C"
struct FeatureHaarPtr TrackerFeatureHAAR_getFeatureAt(
        struct TrackerFeatureHAARPtr ptr, int id)
{
    cv::CvHaarEvaluator::FeatureHaar *retval = new cv::CvHaarEvaluator::FeatureHaar(cv::Size(0,0));
    *retval = ptr->getFeatureAt(id);
    return retval;
}

extern "C"
void TrackerFeatureHAAR_selection(
        struct TrackerFeatureHAARPtr ptr, struct TensorWrapper response, int npoints)
{
    cv::Mat mat = response.toMat();
    ptr->selection(mat, npoints);
}

extern "C"
bool TrackerFeatureHAAR_swapFeature(
        struct TrackerFeatureHAARPtr ptr, int source, int target)
{
    return ptr->swapFeature(source, target);
}

extern "C"
bool TrackerFeatureHAAR_swapFeature2(
        struct TrackerFeatureHAARPtr ptr, int id,
        struct FeatureHaarPtr feature)
{
    return ptr->swapFeature(id, *static_cast<cv::CvHaarEvaluator::FeatureHaar *>(feature.ptr));
}

//TrackerFeatureHOG

extern "C"
struct TrackerFeatureHOGPtr TrackerFeatureHOG_ctor()
{
    return new cv::TrackerFeatureHOG();
}

extern "C"
void TrackerFeatureHOG_dtor(
        struct TrackerFeatureHOGPtr ptr)
{
    delete static_cast<cv::TrackerFeatureHOG *>(ptr.ptr);
}

extern "C"
void TrackerFeatureHOG_selection(
        struct TrackerFeatureHOGPtr ptr, struct TensorWrapper response, int npoints)
{
    cv::Mat mat = response.toMat();
    ptr->selection(mat, npoints);
}

//TrackerFeatureLBP

extern "C"
struct TrackerFeatureLBPPtr TrackerFeatureLBP_ctor()
{
    return new cv::TrackerFeatureLBP();
}

extern "C"
void TrackerFeatureLBP_dtor(
        struct TrackerFeatureLBPPtr ptr)
{
    delete static_cast<cv::TrackerFeatureLBP *>(ptr.ptr);
}

extern "C"
void TrackerFeatureLBP_selection(
        struct TrackerFeatureLBPPtr ptr, struct TensorWrapper response, int npoints)
{
    cv::Mat mat = response.toMat();
    ptr->selection(mat, npoints);
}

//TrackerFeatureSet

extern "C"
struct TrackerFeatureSetPtr TrackerFeatureSet_ctor()
{
    return new cv::TrackerFeatureSet();
}

extern "C"
void TrackerFeatureSet_dtor(
        struct TrackerFeatureSetPtr ptr)
{
    delete static_cast<cv::TrackerFeatureSet *>(ptr.ptr);
}

extern "C"
bool TrackerFeatureSet_addTrackerFeature(
        struct TrackerFeatureSetPtr ptr, const char *trackerFeatureType)
{
    return ptr->addTrackerFeature(trackerFeatureType);
}

extern "C"
bool TrackerFeatureSet_addTrackerFeature2(
        struct TrackerFeatureSetPtr ptr, struct TrackerFeaturePtr feature)
{
    cv::Ptr<cv::TrackerFeature> p(static_cast<cv::TrackerFeature *>(feature.ptr));
    rescueObjectFromPtr(p);
    return ptr->addTrackerFeature(p);
}

extern "C"
void TrackerFeatureSet_extraction(
        struct TrackerFeatureSetPtr ptr, struct TensorArray images)
{
    ptr->extraction(images.toMatList());
}

extern "C"
struct TensorArray TrackerFeatureSet_getResponses(
        struct TrackerFeatureSetPtr ptr)
{
    //need to check
    std::vector<cv::Mat> vec = ptr->getResponses();
    return vec;
}