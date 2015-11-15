#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/ml.hpp>

namespace ml = cv::ml;

extern "C"
struct TensorWrapper TrainData_getSubVector(struct TensorWrapper vec, struct TensorWrapper idx);

struct ParamGridPtr {
    void *ptr;

    inline ml::ParamGrid * operator->() { return static_cast<ml::ParamGrid *>(ptr); }
    inline ParamGridPtr(ml::ParamGrid *ptr) { this->ptr = ptr; }

    inline operator ml::ParamGrid & () { return *static_cast<ml::ParamGrid *>(ptr); }
};

struct TrainDataPtr {
    void *ptr;

    inline ml::TrainData * operator->() { return static_cast<ml::TrainData *>(ptr); }
    inline TrainDataPtr(ml::TrainData *ptr) { this->ptr = ptr; }
    inline operator ml::TrainData *() { return static_cast<ml::TrainData *>(ptr); }
};

struct StatModelPtr {
    void *ptr;

    inline ml::StatModel * operator->() { return static_cast<ml::StatModel *>(ptr); }
    inline StatModelPtr(ml::StatModel *ptr) { this->ptr = ptr; }
};

extern "C"
struct ParamGridPtr ParamGrid_ctor(double _minVal, double _maxVal, double _logStep);

extern "C"
struct ParamGridPtr ParamGrid_ctor_default();

extern "C"
struct TrainDataPtr TrainData_ctor(
        struct TensorWrapper samples, int layout, struct TensorWrapper responses,
        struct TensorWrapper varIdx, struct TensorWrapper sampleIdx,
        struct TensorWrapper sampleWeights, struct TensorWrapper varType);

extern "C"
int TrainData_getLayout(struct TrainDataPtr ptr);

extern "C"
int TrainData_getNTrainSamples(struct TrainDataPtr ptr);

extern "C"
int TrainData_getNTestSamples(struct TrainDataPtr ptr);

extern "C"
int TrainData_getNSamples(struct TrainDataPtr ptr);

extern "C"
int TrainData_getNVars(struct TrainDataPtr ptr);

extern "C"
int TrainData_getNAllVars(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getSamples(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getMissing(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getTrainResponses(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getTrainNormCatResponses(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getTestResponses(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getTestNormCatResponses(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getResponses(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getNormCatResponses(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getSampleWeights(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getTrainSampleWeights(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getTestSampleWeights(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getVarIdx(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getVarType(struct TrainDataPtr ptr);

extern "C"
int TrainData_getResponseType(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getTrainSampleIdx(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getTestSampleIdx(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getDefaultSubstValues(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getClassLabels(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getCatOfs(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getCatMap(struct TrainDataPtr ptr);

extern "C"
void TrainData_shuffleTrainTest(struct TrainDataPtr ptr);

extern "C"
struct TensorWrapper TrainData_getSample(
        struct TrainDataPtr ptr, struct TensorWrapper varIdx, int sidx);

extern "C"
struct TensorWrapper TrainData_getTrainSamples(
        struct TrainDataPtr ptr, int layout, bool compressSamples, bool compressVars);

extern "C"
struct TensorWrapper TrainData_getValues(
        struct TrainDataPtr ptr, int vi, struct TensorWrapper sidx);

extern "C"
struct TensorWrapper TrainData_getNormCatValues(
        struct TrainDataPtr ptr, int vi, struct TensorWrapper sidx);

extern "C"
void TrainData_setTrainTestSplit(struct TrainDataPtr ptr, int count, bool shuffle);

extern "C"
void TrainData_setTrainTestSplitRatio(struct TrainDataPtr ptr, double ratio, bool shuffle);

extern "C"
int StatModel_getVarCount(struct StatModelPtr ptr);

extern "C"
bool StatModel_empty(struct StatModelPtr ptr);

extern "C"
bool StatModel_isTrained(struct StatModelPtr ptr);

extern "C"
bool StatModel_isClassifier(struct StatModelPtr ptr);

extern "C"
bool StatModel_train(struct StatModelPtr ptr, struct TrainDataPtr trainData, int flags);

extern "C"
bool StatModel_train_Mat(
        struct StatModelPtr ptr, struct TensorWrapper samples, int layout, struct TensorWrapper responses);

extern "C"
struct TensorPlusFloat StatModel_calcError(
        struct StatModelPtr ptr, struct TrainDataPtr data, bool test, struct TensorWrapper resp);

extern "C"
float StatModel_predict(
        struct StatModelPtr ptr, struct TensorWrapper samples, struct TensorWrapper results, int flags);

struct NormalBayesClassifierPtr {
    void *ptr;

    inline ml::NormalBayesClassifier * operator->() { return static_cast<ml::NormalBayesClassifier *>(ptr); }
    inline NormalBayesClassifierPtr(ml::NormalBayesClassifier *ptr) { this->ptr = ptr; }
};

extern "C"
struct NormalBayesClassifierPtr NormalBayesClassifier_ctor();

extern "C"
struct TensorArrayPlusFloat NormalBayesClassifier_predictProb(
        struct NormalBayesClassifierPtr ptr, struct TensorWrapper inputs,
        struct TensorWrapper outputs, struct TensorWrapper outputProbs, int flags);

struct KNearestPtr {
    void *ptr;

    inline ml::KNearest * operator->() { return static_cast<ml::KNearest *>(ptr); }
    inline KNearestPtr(ml::KNearest *ptr) { this->ptr = ptr; }
};

extern "C"
struct KNearestPtr KNearest_ctor();

extern "C"
void KNearest_setDefaultK(struct KNearestPtr ptr, int val);

extern "C"
int KNearest_getDefaultK(struct KNearestPtr ptr);

extern "C"
void KNearest_setIsClassifier(struct KNearestPtr ptr, bool val);

extern "C"
bool KNearest_getIsClassifier(struct KNearestPtr ptr);

extern "C"
void KNearest_setEmax(struct KNearestPtr ptr, int val);

extern "C"
int KNearest_getEmax(struct KNearestPtr ptr);

extern "C"
void KNearest_setAlgorithmType(struct KNearestPtr ptr, int val);

extern "C"
int KNearest_getAlgorithmType(struct KNearestPtr ptr);

extern "C"
float KNearest_findNearest(
        struct KNearestPtr ptr, struct TensorWrapper samples, int k,
        struct TensorWrapper results, struct TensorWrapper neighborResponses,
        struct TensorWrapper dist);

struct SVMPtr {
    void *ptr;

    inline ml::SVM * operator->() { return static_cast<ml::SVM *>(ptr); }
    inline SVMPtr(ml::SVM *ptr) { this->ptr = ptr; }
};

extern "C"
struct SVMPtr SVM_ctor();

extern "C"
void SVM_setType(struct SVMPtr ptr, int val);

extern "C"
int SVM_getType(struct SVMPtr ptr);

extern "C"
void SVM_setGamma(struct SVMPtr ptr, double val);

extern "C"
double SVM_getGamma(struct SVMPtr ptr);

extern "C"
void SVM_setCoef0(struct SVMPtr ptr, double val);

extern "C"
double SVM_getCoef0(struct SVMPtr ptr);

extern "C"
void SVM_setDegree(struct SVMPtr ptr, double val);

extern "C"
double SVM_getDegree(struct SVMPtr ptr);

extern "C"
void SVM_setC(struct SVMPtr ptr, double val);

extern "C"
double SVM_getC(struct SVMPtr ptr);

extern "C"
void SVM_setNu(struct SVMPtr ptr, double val);

extern "C"
double SVM_getNu(struct SVMPtr ptr);

extern "C"
void SVM_setP(struct SVMPtr ptr, double val);

extern "C"
double SVM_getP(struct SVMPtr ptr);

extern "C"
void SVM_setClassWeights(struct SVMPtr ptr, struct TensorWrapper val);

extern "C"
struct TensorWrapper SVM_getClassWeights(struct SVMPtr ptr);

extern "C"
void SVM_setTermCriteria(struct SVMPtr ptr, struct TermCriteriaWrapper val);

extern "C"
struct TermCriteriaWrapper SVM_getTermCriteria(struct SVMPtr ptr);

extern "C"
int SVM_getKernelType(struct SVMPtr ptr);

extern "C"
void SVM_setKernel(struct SVMPtr ptr, int val);

//extern "C"
//void SVM_setCustomKernel(struct SVMPtr ptr, struct KernelPtr val);

extern "C"
bool SVM_trainAuto(
        struct SVMPtr ptr, struct TrainDataPtr data, int kFold, struct ParamGridPtr Cgrid,
        struct ParamGridPtr gammaGrid, struct ParamGridPtr pGrid, struct ParamGridPtr nuGrid,
        struct ParamGridPtr coeffGrid, struct ParamGridPtr degreeGrid, bool balanced);

extern "C"
struct TensorWrapper SVM_getSupportVectors(struct SVMPtr ptr);

extern "C"
struct TensorArrayPlusDouble SVM_getDecisionFunction(
        struct SVMPtr ptr, int i, struct TensorWrapper alpha, struct TensorWrapper svidx);

extern "C"
struct ParamGridPtr SVM_getDefaultGrid(struct SVMPtr ptr, int param_id);

struct EMPtr {
    void *ptr;

    inline ml::EM * operator->() { return static_cast<ml::EM *>(ptr); }
    inline EMPtr(ml::EM *ptr) { this->ptr = ptr; }
};

extern "C"
struct EMPtr EM_ctor();

extern "C"
void EM_setClustersNumber(struct EMPtr ptr, int val);

extern "C"
int EM_getClustersNumber(struct EMPtr ptr);

extern "C"
void EM_setCovarianceMatrixType(struct EMPtr ptr, int val);

extern "C"
int EM_getCovarianceMatrixType(struct EMPtr ptr);

extern "C"
void EM_setTermCriteria(struct EMPtr ptr, struct TermCriteriaWrapper val);

extern "C"
struct TermCriteriaWrapper EM_getTermCriteria(struct EMPtr ptr);

extern "C"
struct TensorWrapper EM_getWeights(struct EMPtr ptr);

extern "C"
struct TensorWrapper EM_getMeans(struct EMPtr ptr);

extern "C"
struct TensorArray EM_getCovs(struct EMPtr ptr);

extern "C"
struct Vec2dWrapper EM_predict2(
        struct EMPtr ptr, struct TensorWrapper sample, struct TensorWrapper probs);

extern "C"
bool EM_trainEM(
        struct EMPtr ptr, struct TensorWrapper samples,
        struct TensorWrapper logLikelihoods,
        struct TensorWrapper labels, struct TensorWrapper probs);

extern "C"
bool EM_trainE(
        struct EMPtr ptr, struct TensorWrapper samples, struct TensorWrapper means0,
        struct TensorWrapper covs0, struct TensorWrapper weights0,
        struct TensorWrapper logLikelihoods, struct TensorWrapper labels,
        struct TensorWrapper probs);

extern "C"
bool EM_trainM(
        struct EMPtr ptr, struct TensorWrapper samples, struct TensorWrapper probs0,
        struct TensorWrapper logLikelihoods, struct TensorWrapper labels,
        struct TensorWrapper probs);

struct DTreesPtr {
    void *ptr;

    inline ml::DTrees * operator->() { return static_cast<ml::DTrees *>(ptr); }
    inline DTreesPtr(ml::DTrees *ptr) { this->ptr = ptr; }
};

struct ConstNodeArray {
    const ml::DTrees::Node *ptr;
    int size;
};

struct ConstSplitArray {
    const ml::DTrees::Split *ptr;
    int size;
};

extern "C"
struct DTreesPtr DTrees_ctor();

extern "C"
void DTrees_setMaxCategories(struct DTreesPtr ptr, int val);

extern "C"
int DTrees_getMaxCategories(struct DTreesPtr ptr);

extern "C"
void DTrees_setMaxDepth(struct DTreesPtr ptr, int val);

extern "C"
int DTrees_getMaxDepth(struct DTreesPtr ptr);

extern "C"
void DTrees_setMinSampleCount(struct DTreesPtr ptr, int val);

extern "C"
int DTrees_getMinSampleCount(struct DTreesPtr ptr);

extern "C"
void DTrees_setCVFolds(struct DTreesPtr ptr, int val);

extern "C"
int DTrees_getCVFolds(struct DTreesPtr ptr);

extern "C"
void DTrees_setUseSurrogates(struct DTreesPtr ptr, bool val);

extern "C"
bool DTrees_getUseSurrogates(struct DTreesPtr ptr);

extern "C"
void DTrees_setUse1SERule(struct DTreesPtr ptr, bool val);

extern "C"
bool DTrees_getUse1SERule(struct DTreesPtr ptr);

extern "C"
void DTrees_setTruncatePrunedTree(struct DTreesPtr ptr, bool val);

extern "C"
bool DTrees_getTruncatePrunedTree(struct DTreesPtr ptr);

extern "C"
void DTrees_setRegressionAccuracy(struct DTreesPtr ptr, float val);

extern "C"
float DTrees_getRegressionAccuracy(struct DTreesPtr ptr);

extern "C"
void DTrees_setPriors(struct DTreesPtr ptr, struct TensorWrapper val);

extern "C"
struct TensorWrapper DTrees_getPriors(struct DTreesPtr ptr);

extern "C"
struct TensorWrapper DTrees_getRoots(struct DTreesPtr ptr);

extern "C"
struct ConstNodeArray DTrees_getNodes(struct DTreesPtr ptr);

extern "C"
struct ConstSplitArray DTrees_getSplits(struct DTreesPtr ptr);

extern "C"
struct TensorWrapper DTrees_getSubsets(struct DTreesPtr ptr);

struct RTreesPtr {
    void *ptr;

    inline ml::RTrees * operator->() { return static_cast<ml::RTrees *>(ptr); }
    inline RTreesPtr(ml::RTrees *ptr) { this->ptr = ptr; }
};

struct BoostPtr {
    void *ptr;

    inline ml::Boost * operator->() { return static_cast<ml::Boost *>(ptr); }
    inline BoostPtr(ml::Boost *ptr) { this->ptr = ptr; }
};

struct ANN_MLPPtr {
    void *ptr;

    inline ml::ANN_MLP * operator->() { return static_cast<ml::ANN_MLP *>(ptr); }
    inline ANN_MLPPtr(ml::ANN_MLP *ptr) { this->ptr = ptr; }
};

struct LogisticRegressionPtr {
    void *ptr;

    inline ml::LogisticRegression * operator->() { return static_cast<ml::LogisticRegression *>(ptr); }
    inline LogisticRegressionPtr(ml::LogisticRegression *ptr) { this->ptr = ptr; }
};
