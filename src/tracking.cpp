#include <tracking.hpp>

extern "C"
void test(){
    cv::Mat img = cv::imread("/home/ainur/torch-opencv/demo/data/lena.jpg");
    cv::MultiTracker temp = cv::MultiTracker();
    bool res = temp.add("TLD", img, cv::Rect2d(2,2,20,20));
    if(res) std::cout << "true" << std::endl;
    else std::cout << "false" << std::endl;
    cv::imshow("1", img);
    cv::waitKey(0);
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
}