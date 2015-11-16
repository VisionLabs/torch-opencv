#include <ml.hpp>

struct TensorWrapper randMVNormal(
        struct TensorWrapper mean, struct TensorWrapper cov, int nsamples, struct TensorWrapper samples)
{
    if (samples.isNull()) {
        cv::Mat retval;
        ml::randMVNormal(mean.toMat(), cov.toMat(), nsamples, retval);
        return TensorWrapper(retval);
    } else {
        ml::randMVNormal(mean.toMat(), cov.toMat(), nsamples, samples.toMat());
        return samples;
    }
}

struct TensorArray createConcentricSpheresTestSet(
        int nsamples, int nfeatures, int nclasses, struct TensorWrapper samples, struct TensorWrapper responses)
{
    std::vector<cv::Mat> retval(2);
    if (!samples.isNull()) retval[0] = samples;
    if (!responses.isNull()) retval[1] = responses;

    ml::createConcentricSpheresTestSet(nsamples, nfeatures, nclasses, retval[0], retval[1]);
    return TensorArray(retval);
}

// ParamGrid

extern "C"
struct ParamGridPtr ParamGrid_ctor(double _minVal, double _maxVal, double _logStep)
{
    return new ml::ParamGrid(_minVal, _maxVal, _logStep);
}

extern "C"
struct ParamGridPtr ParamGrid_ctor_default()
{
    return new ml::ParamGrid();
}

// TrainData

extern "C"
struct TrainDataPtr TrainData_ctor(
        struct TensorWrapper samples, int layout, struct TensorWrapper responses,
        struct TensorWrapper varIdx, struct TensorWrapper sampleIdx,
        struct TensorWrapper sampleWeights, struct TensorWrapper varType)
{
    return rescueObjectFromPtr(ml::TrainData::create(
            samples.toMat(), layout, responses.toMat(), varIdx.toMat(),
            sampleIdx.toMat(), sampleWeights.toMat(), varType.toMat()));
}

extern "C"
struct TensorWrapper TrainData_getSubVector(
        struct TensorWrapper vec, struct TensorWrapper idx)
{
    return TensorWrapper(ml::TrainData::getSubVector(vec.toMat(), idx.toMat()));
}

extern "C"
int TrainData_getLayout(struct TrainDataPtr ptr)
{
    return ptr->getLayout();
}

extern "C"
int TrainData_getNTrainSamples(struct TrainDataPtr ptr)
{
    return ptr->getNTrainSamples();
}

extern "C"
int TrainData_getNTestSamples(struct TrainDataPtr ptr)
{
    return ptr->getNTestSamples();
}

extern "C"
int TrainData_getNSamples(struct TrainDataPtr ptr)
{
    return ptr->getNSamples();
}

extern "C"
int TrainData_getNVars(struct TrainDataPtr ptr)
{
    return ptr->getNVars();
}

extern "C"
int TrainData_getNAllVars(struct TrainDataPtr ptr)
{
    return ptr->getNAllVars();
}

extern "C"
struct TensorWrapper TrainData_getSamples(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getSamples());
}

extern "C"
struct TensorWrapper TrainData_getMissing(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getMissing());
}

extern "C"
struct TensorWrapper TrainData_getTrainResponses(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getTrainResponses());
}

extern "C"
struct TensorWrapper TrainData_getTrainNormCatResponses(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getTrainNormCatResponses());
}

extern "C"
struct TensorWrapper TrainData_getTestResponses(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getTestResponses());
}

extern "C"
struct TensorWrapper TrainData_getTestNormCatResponses(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getTestNormCatResponses());
}

extern "C"
struct TensorWrapper TrainData_getResponses(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getResponses());
}

extern "C"
struct TensorWrapper TrainData_getNormCatResponses(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getNormCatResponses());
}

extern "C"
struct TensorWrapper TrainData_getSampleWeights(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getSampleWeights());
}

extern "C"
struct TensorWrapper TrainData_getTrainSampleWeights(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getTrainSampleWeights());
}

extern "C"
struct TensorWrapper TrainData_getTestSampleWeights(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getTestSampleWeights());
}

extern "C"
struct TensorWrapper TrainData_getVarIdx(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getVarIdx());
}

extern "C"
struct TensorWrapper TrainData_getVarType(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getVarType());
}

extern "C"
int TrainData_getResponseType(struct TrainDataPtr ptr)
{
    return ptr->getResponseType();
}

extern "C"
struct TensorWrapper TrainData_getTrainSampleIdx(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getTrainSampleIdx());
}

extern "C"
struct TensorWrapper TrainData_getTestSampleIdx(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getTestSampleIdx());
}

extern "C"
struct TensorWrapper TrainData_getDefaultSubstValues(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getDefaultSubstValues());
}

extern "C"
struct TensorWrapper TrainData_getClassLabels(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getClassLabels());
}

extern "C"
struct TensorWrapper TrainData_getCatOfs(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getCatOfs());
}

extern "C"
struct TensorWrapper TrainData_getCatMap(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getCatMap());
}

extern "C"
void TrainData_shuffleTrainTest(struct TrainDataPtr ptr)
{
    ptr->shuffleTrainTest();
}

extern "C"
struct TensorWrapper TrainData_getSample(
        struct TrainDataPtr ptr, struct TensorWrapper varIdx, int sidx)
{
    cv::Mat varIdxMat = varIdx;
    std::vector<float> output(varIdxMat.rows * varIdxMat.cols);
    ptr->getSample(varIdxMat, sidx, output.data());
    return TensorWrapper(cv::Mat(output));
}

extern "C"
struct TensorWrapper TrainData_getTrainSamples(
        struct TrainDataPtr ptr, int layout, bool compressSamples, bool compressVars)
{
    return TensorWrapper(ptr->getTrainSamples(layout, compressSamples, compressVars));
}

extern "C"
struct TensorWrapper TrainData_getValues(
        struct TrainDataPtr ptr, int vi, struct TensorWrapper sidx)
{
    cv::Mat sidxMat = sidx;
    std::vector<float> output(sidxMat.rows * sidxMat.cols);
    ptr->getValues(vi, sidxMat, output.data());
    return TensorWrapper(cv::Mat(output));
}

extern "C"
struct TensorWrapper TrainData_getNormCatValues(
        struct TrainDataPtr ptr, int vi, struct TensorWrapper sidx)
{
    cv::Mat sidxMat = sidx;
    std::vector<int> output(sidxMat.rows * sidxMat.cols);
    ptr->getNormCatValues(vi, sidxMat, output.data());
    return TensorWrapper(cv::Mat(output));
}

extern "C"
void TrainData_setTrainTestSplit(struct TrainDataPtr ptr, int count, bool shuffle)
{
    ptr->setTrainTestSplit(count, shuffle);
}

extern "C"
void TrainData_setTrainTestSplitRatio(struct TrainDataPtr ptr, double ratio, bool shuffle)
{
    ptr->setTrainTestSplitRatio(ratio, shuffle);
}

// StatModel

extern "C"
int StatModel_getVarCount(struct StatModelPtr ptr)
{
    return ptr->getVarCount();
}

extern "C"
bool StatModel_empty(struct StatModelPtr ptr)
{
    return ptr->empty();
}

extern "C"
bool StatModel_isTrained(struct StatModelPtr ptr)
{
    return ptr->isTrained();
}

extern "C"
bool StatModel_isClassifier(struct StatModelPtr ptr)
{
    return ptr->isClassifier();
}

extern "C"
bool StatModel_train(struct StatModelPtr ptr, struct TrainDataPtr trainData, int flags)
{
    cv::Ptr<ml::TrainData> trainDataPtr(static_cast<ml::TrainData *>(trainData));
    ptr->train(trainDataPtr, flags);
    rescueObjectFromPtr(trainDataPtr);
}

extern "C"
bool StatModel_train_Mat(
        struct StatModelPtr ptr, struct TensorWrapper samples, int layout, struct TensorWrapper responses)
{
    ptr->train(samples.toMat(), layout, responses.toMat());
}

extern "C"
struct TensorPlusFloat StatModel_calcError(
        struct StatModelPtr ptr, struct TrainDataPtr data, bool test, struct TensorWrapper resp)
{
    cv::Ptr<ml::TrainData> dataPtr(static_cast<ml::TrainData *>(data));
    TensorPlusFloat retval;

    if (resp.isNull()) {
        cv::Mat result;
        retval.val = ptr->calcError(dataPtr, test, result);
        new (&retval.tensor) TensorWrapper(result);
    } else {
        retval.val = ptr->calcError(dataPtr, test, resp.toMat());
        retval.tensor = resp;
    }

    rescueObjectFromPtr(dataPtr);
    return retval;
}

extern "C"
float StatModel_predict(
        struct StatModelPtr ptr, struct TensorWrapper samples, struct TensorWrapper results, int flags)
{
    return ptr->predict(samples.toMat(), TO_MAT_OR_NOARRAY(results), flags);
}

// NormalBayesClassifier

extern "C"
struct NormalBayesClassifierPtr NormalBayesClassifier_ctor()
{
    return rescueObjectFromPtr(ml::NormalBayesClassifier::create());
}

extern "C"
struct TensorArrayPlusFloat NormalBayesClassifier_predictProb(
        struct NormalBayesClassifierPtr ptr, struct TensorWrapper inputs,
        struct TensorWrapper outputs, struct TensorWrapper outputProbs, int flags)
{
    TensorArrayPlusFloat retval;
    std::vector<cv::Mat> result(2);
    if (!outputs.isNull())      result[0] = outputs;
    if (!outputProbs.isNull())  result[1] = outputProbs;

    retval.val = ptr->predictProb(
            inputs.toMat(),
            outputs.isNull() ? result[0] : outputs.toMat(),
            outputProbs.isNull() ? result[1] : outputs.toMat(),
            flags);

    new (&retval.tensors) TensorArray(result);

    return retval;
}

// KNearest

extern "C"
struct KNearestPtr KNearest_ctor()
{
    return rescueObjectFromPtr(ml::KNearest::create());
}

extern "C"
void KNearest_setDefaultK(struct KNearestPtr ptr, int val)
{
    ptr->setDefaultK(val);
}

extern "C"
int KNearest_getDefaultK(struct KNearestPtr ptr)
{
    return ptr->getDefaultK();
}

extern "C"
void KNearest_setIsClassifier(struct KNearestPtr ptr, bool val)
{
    ptr->setIsClassifier(val);
}

extern "C"
bool KNearest_getIsClassifier(struct KNearestPtr ptr)
{
    return ptr->getIsClassifier();
}

extern "C"
void KNearest_setEmax(struct KNearestPtr ptr, int val)
{
    ptr->setEmax(val);
}

extern "C"
int KNearest_getEmax(struct KNearestPtr ptr)
{
    return ptr->getEmax();
}

extern "C"
void KNearest_setAlgorithmType(struct KNearestPtr ptr, int val)
{
    ptr->setAlgorithmType(val);
}

extern "C"
int KNearest_getAlgorithmType(struct KNearestPtr ptr)
{
    return ptr->getAlgorithmType();
}

extern "C"
float KNearest_findNearest(
        struct KNearestPtr ptr, struct TensorWrapper samples, int k,
        struct TensorWrapper results, struct TensorWrapper neighborResponses,
        struct TensorWrapper dist)
{
    return ptr->findNearest(
            samples.toMat(), k, TO_MAT_OR_NOARRAY(results),
            TO_MAT_OR_NOARRAY(neighborResponses), TO_MAT_OR_NOARRAY(dist));
}

// SVM

extern "C"
struct SVMPtr SVM_ctor()
{
    return rescueObjectFromPtr(ml::SVM::create());
}

extern "C"
void SVM_setType(struct SVMPtr ptr, int val)
{
    ptr->setType(val);
}

extern "C"
int SVM_getType(struct SVMPtr ptr)
{
    return ptr->getType();
}

extern "C"
void SVM_setGamma(struct SVMPtr ptr, double val)
{
    ptr->setGamma(val);
}

extern "C"
double SVM_getGamma(struct SVMPtr ptr)
{
    return ptr->getGamma();
}

extern "C"
void SVM_setCoef0(struct SVMPtr ptr, double val)
{
    ptr->setCoef0(val);
}

extern "C"
double SVM_getCoef0(struct SVMPtr ptr)
{
    return ptr->getCoef0();
}

extern "C"
void SVM_setDegree(struct SVMPtr ptr, double val)
{
    ptr->setDegree(val);
}

extern "C"
double SVM_getDegree(struct SVMPtr ptr)
{
    return ptr->getDegree();
}

extern "C"
void SVM_setC(struct SVMPtr ptr, double val)
{
    ptr->setC(val);
}

extern "C"
double SVM_getC(struct SVMPtr ptr)
{
    return ptr->getC();
}

extern "C"
void SVM_setNu(struct SVMPtr ptr, double val)
{
    ptr->setNu(val);
}

extern "C"
double SVM_getNu(struct SVMPtr ptr)
{
    return ptr->getNu();
}

extern "C"
void SVM_setP(struct SVMPtr ptr, double val)
{
    ptr->setP(val);
}

extern "C"
double SVM_getP(struct SVMPtr ptr)
{
    return ptr->getP();
}

extern "C"
void SVM_setClassWeights(struct SVMPtr ptr, struct TensorWrapper val)
{
    ptr->setClassWeights(val.toMat());
}

extern "C"
struct TensorWrapper SVM_getClassWeights(struct SVMPtr ptr)
{
    return TensorWrapper(ptr->getClassWeights());
}

extern "C"
void SVM_setTermCriteria(struct SVMPtr ptr, struct TermCriteriaWrapper val)
{
    ptr->setTermCriteria(val);
}

extern "C"
struct TermCriteriaWrapper SVM_getTermCriteria(struct SVMPtr ptr)
{
    return ptr->getTermCriteria();
}

extern "C"
int SVM_getKernelType(struct SVMPtr ptr)
{
    return ptr->getKernelType();
}

extern "C"
void SVM_setKernel(struct SVMPtr ptr, int val)
{
    ptr->setKernel(val);
}

// TODO this
//extern "C"
//void SVM_setCustomKernel(struct SVMPtr ptr, struct KernelPtr val)
//{
//    ptr->setCustomKernel(val);
//}

extern "C"
bool SVM_trainAuto(
        struct SVMPtr ptr, struct TrainDataPtr data, int kFold, struct ParamGridPtr Cgrid,
        struct ParamGridPtr gammaGrid, struct ParamGridPtr pGrid, struct ParamGridPtr nuGrid,
        struct ParamGridPtr coeffGrid, struct ParamGridPtr degreeGrid, bool balanced)
{
    cv::Ptr<ml::TrainData> dataPtr(static_cast<ml::TrainData *>(data));
    rescueObjectFromPtr(dataPtr);

    return ptr->trainAuto(
            dataPtr, kFold, Cgrid, gammaGrid, pGrid,
            nuGrid, coeffGrid, degreeGrid, balanced);
}

extern "C"
struct TensorWrapper SVM_getSupportVectors(struct SVMPtr ptr)
{
    return TensorWrapper(ptr->getSupportVectors());
}

extern "C"
struct TensorArrayPlusDouble SVM_getDecisionFunction(
        struct SVMPtr ptr, int i, struct TensorWrapper alpha, struct TensorWrapper svidx)
{
    TensorArrayPlusDouble retval;
    std::vector<cv::Mat> result;
    if (!alpha.isNull()) result[0] = alpha;
    if (!svidx.isNull()) result[1] = svidx;

    retval.val = ptr->getDecisionFunction(
            i,
            alpha.isNull() ? result[0] : alpha.toMat(),
            svidx.isNull() ? result[1] : svidx.toMat());

    new (&retval.tensors) TensorArray(result);
    return retval;
}

extern "C"
struct ParamGridPtr SVM_getDefaultGrid(struct SVMPtr ptr, int param_id)
{
    ml::ParamGrid *result = new ml::ParamGrid;
    *result = ptr->getDefaultGrid(param_id);
    return result;
}

// EM

extern "C"
struct EMPtr EM_ctor()
{
    return rescueObjectFromPtr(ml::EM::create());
}

extern "C"
void EM_setClustersNumber(struct EMPtr ptr, int val)
{
    ptr->setClustersNumber(val);
}

extern "C"
int EM_getClustersNumber(struct EMPtr ptr)
{
    return ptr->getClustersNumber();
}

extern "C"
void EM_setCovarianceMatrixType(struct EMPtr ptr, int val)
{
    ptr->setCovarianceMatrixType(val);
}

extern "C"
int EM_getCovarianceMatrixType(struct EMPtr ptr)
{
    return ptr->getCovarianceMatrixType();
}

extern "C"
void EM_setTermCriteria(struct EMPtr ptr, struct TermCriteriaWrapper val)
{
    ptr->setTermCriteria(val);
}

extern "C"
struct TermCriteriaWrapper EM_getTermCriteria(struct EMPtr ptr)
{
    return ptr->getTermCriteria();
}

extern "C"
struct TensorWrapper EM_getWeights(struct EMPtr ptr)
{
    return TensorWrapper(ptr->getWeights());
}

extern "C"
struct TensorWrapper EM_getMeans(struct EMPtr ptr)
{
    return TensorWrapper(ptr->getMeans());
}

extern "C"
struct TensorArray EM_getCovs(struct EMPtr ptr)
{
    std::vector<cv::Mat> retval;
    ptr->getCovs(retval);
    return TensorArray(retval);
}

extern "C"
struct Vec2dWrapper EM_predict2(
        struct EMPtr ptr, struct TensorWrapper sample, struct TensorWrapper probs)
{
    return ptr->predict2(sample.toMat(), TO_MAT_OR_NOARRAY(probs));
}

extern "C"
bool EM_trainEM(
        struct EMPtr ptr, struct TensorWrapper samples,
        struct TensorWrapper logLikelihoods,
        struct TensorWrapper labels, struct TensorWrapper probs)
{
    return ptr->trainEM(
            samples.toMat(), TO_MAT_OR_NOARRAY(logLikelihoods),
            TO_MAT_OR_NOARRAY(labels), TO_MAT_OR_NOARRAY(probs));
}

extern "C"
bool EM_trainE(
        struct EMPtr ptr, struct TensorWrapper samples, struct TensorWrapper means0,
        struct TensorWrapper covs0, struct TensorWrapper weights0,
        struct TensorWrapper logLikelihoods, struct TensorWrapper labels,
        struct TensorWrapper probs)
{
    return ptr->trainE(
            samples.toMat(), means0.toMat(), TO_MAT_OR_NOARRAY(covs0),
            TO_MAT_OR_NOARRAY(weights0), TO_MAT_OR_NOARRAY(logLikelihoods),
            TO_MAT_OR_NOARRAY(labels), TO_MAT_OR_NOARRAY(probs));
}

extern "C"
bool EM_trainM(
        struct EMPtr ptr, struct TensorWrapper samples, struct TensorWrapper probs0,
        struct TensorWrapper logLikelihoods, struct TensorWrapper labels,
        struct TensorWrapper probs)
{
    return ptr->trainM(
            samples.toMat(), probs0.toMat(), TO_MAT_OR_NOARRAY(logLikelihoods),
            TO_MAT_OR_NOARRAY(labels), TO_MAT_OR_NOARRAY(probs));
}

// DTrees: Node

// DTrees

extern "C"
struct DTreesPtr DTrees_ctor()
{
    return rescueObjectFromPtr(ml::DTrees::create());
}

extern "C"
void DTrees_setMaxCategories(struct DTreesPtr ptr, int val)
{
    ptr->setMaxCategories(val);
}

extern "C"
int DTrees_getMaxCategories(struct DTreesPtr ptr)
{
    return ptr->getMaxCategories();
}

extern "C"
void DTrees_setMaxDepth(struct DTreesPtr ptr, int val)
{
    ptr->setMaxDepth(val);
}

extern "C"
int DTrees_getMaxDepth(struct DTreesPtr ptr)
{
    return ptr->getMaxDepth();
}

extern "C"
void DTrees_setMinSampleCount(struct DTreesPtr ptr, int val)
{
    ptr->setMinSampleCount(val);
}

extern "C"
int DTrees_getMinSampleCount(struct DTreesPtr ptr)
{
    return ptr->getMinSampleCount();
}

extern "C"
void DTrees_setCVFolds(struct DTreesPtr ptr, int val)
{
    ptr->setCVFolds(val);
}

extern "C"
int DTrees_getCVFolds(struct DTreesPtr ptr)
{
    return ptr->getCVFolds();
}

extern "C"
void DTrees_setUseSurrogates(struct DTreesPtr ptr, bool val)
{
    ptr->setUseSurrogates(val);
}

extern "C"
bool DTrees_getUseSurrogates(struct DTreesPtr ptr)
{
    return ptr->getUseSurrogates();
}

extern "C"
void DTrees_setUse1SERule(struct DTreesPtr ptr, bool val)
{
    ptr->setUse1SERule(val);
}

extern "C"
bool DTrees_getUse1SERule(struct DTreesPtr ptr)
{
    return ptr->getUse1SERule();
}

extern "C"
void DTrees_setTruncatePrunedTree(struct DTreesPtr ptr, bool val)
{
    ptr->setTruncatePrunedTree(val);
}

extern "C"
bool DTrees_getTruncatePrunedTree(struct DTreesPtr ptr)
{
    return ptr->getTruncatePrunedTree();
}

extern "C"
void DTrees_setRegressionAccuracy(struct DTreesPtr ptr, float val)
{
    ptr->setRegressionAccuracy(val);
}

extern "C"
float DTrees_getRegressionAccuracy(struct DTreesPtr ptr)
{
    return ptr->getRegressionAccuracy();
}

extern "C"
void DTrees_setPriors(struct DTreesPtr ptr, struct TensorWrapper val)
{
    ptr->setPriors(val);
}

extern "C"
struct TensorWrapper DTrees_getPriors(struct DTreesPtr ptr)
{
    return TensorWrapper(ptr->getPriors());
}

extern "C"
struct TensorWrapper DTrees_getRoots(struct DTreesPtr ptr)
{
    return TensorWrapper(cv::Mat(ptr->getRoots()));
}

extern "C"
struct ConstNodeArray DTrees_getNodes(struct DTreesPtr ptr)
{
    const std::vector<ml::DTrees::Node> & result = ptr->getNodes();
    ConstNodeArray retval;
    retval.ptr = result.data();
    retval.size = result.size();
    return retval;
}

extern "C"
struct ConstSplitArray DTrees_getSplits(struct DTreesPtr ptr)
{
    const std::vector<ml::DTrees::Split> & result = ptr->getSplits();
    ConstSplitArray retval;
    retval.ptr = result.data();
    retval.size = result.size();
    return retval;
}

extern "C"
struct TensorWrapper DTrees_getSubsets(struct DTreesPtr ptr)
{
    return TensorWrapper(cv::Mat(ptr->getSubsets()));
}

// RTrees

extern "C"
struct RTreesPtr RTrees_ctor()
{
    return rescueObjectFromPtr(ml::RTrees::create());
}

extern "C"
void RTrees_setCalculateVarImportance(struct RTreesPtr ptr, bool val)
{
    ptr->setCalculateVarImportance(val);
}

extern "C"
bool RTrees_getCalculateVarImportance(struct RTreesPtr ptr)
{
    return ptr->getCalculateVarImportance();
}

extern "C"
void RTrees_setActiveVarCount(struct RTreesPtr ptr, int val)
{
    ptr->setActiveVarCount(val);
}

extern "C"
int RTrees_getActiveVarCount(struct RTreesPtr ptr)
{
    return ptr->getActiveVarCount();
}

extern "C"
void RTrees_setTermCriteria(struct RTreesPtr ptr, struct TermCriteriaWrapper val)
{
    ptr->setTermCriteria(val);
}

extern "C"
struct TermCriteriaWrapper RTrees_getTermCriteria(struct RTreesPtr ptr)
{
    return ptr->getTermCriteria();
}

extern "C"
struct TensorWrapper RTrees_getVarImportance(struct RTreesPtr ptr)
{
    return TensorWrapper(ptr->getVarImportance());
}

// Boost

extern "C"
struct BoostPtr Boost_ctor()
{
    return rescueObjectFromPtr(ml::Boost::create());
}

extern "C"
void Boost_setBoostType(struct BoostPtr ptr, int val)
{
    ptr->setBoostType(val);
}

extern "C"
int Boost_getBoostType(struct BoostPtr ptr)
{
    return ptr->getBoostType();
}

extern "C"
void Boost_setWeakCount(struct BoostPtr ptr, int val)
{
    ptr->setWeakCount(val);
}

extern "C"
int Boost_getWeakCount(struct BoostPtr ptr)
{
    return ptr->getWeakCount();
}

extern "C"
void Boost_setWeightTrimRate(struct BoostPtr ptr, double val)
{
    ptr->setWeightTrimRate(val);
}

extern "C"
double Boost_getWeightTrimRate(struct BoostPtr ptr)
{
    return ptr->getWeightTrimRate();
}

// ANN_MLP

extern "C"
struct ANN_MLPPtr ANN_MLP_ctor()
{
    return rescueObjectFromPtr(ml::ANN_MLP::create());
}

extern "C"
void ANN_MLP_setTrainMethod(struct ANN_MLPPtr ptr, int method, double param1, double param2)
{
    ptr->setTrainMethod(method, param1, param2);
}

extern "C"
int ANN_MLP_getTrainMethod(struct ANN_MLPPtr ptr)
{
    return ptr->getTrainMethod();
}

extern "C"
void ANN_MLP_setActivationFunction(struct ANN_MLPPtr ptr, int type, double param1, double param2)
{
    ptr->setActivationFunction(type, param1, param2);
}

extern "C"
void ANN_MLP_setLayerSizes(struct ANN_MLPPtr ptr, struct TensorWrapper val)
{
    ptr->setLayerSizes(val.toMat());
}

extern "C"
struct TensorWrapper ANN_MLP_getLayerSizes(struct ANN_MLPPtr ptr)
{
    return TensorWrapper(ptr->getLayerSizes());
}

extern "C"
void ANN_MLP_setTermCriteria(struct ANN_MLPPtr ptr, struct TermCriteriaWrapper val)
{
    ptr->setTermCriteria(val);
}

extern "C"
struct TermCriteriaWrapper ANN_MLP_getTermCriteria(struct ANN_MLPPtr ptr)
{
    return ptr->getTermCriteria();
}

extern "C"
void ANN_MLP_setBackpropWeightScale(struct ANN_MLPPtr ptr, double val)
{
    ptr->setBackpropWeightScale(val);
}

extern "C"
double ANN_MLP_getBackpropWeightScale(struct ANN_MLPPtr ptr)
{
    return ptr->getBackpropWeightScale();
}

extern "C"
void ANN_MLP_setBackpropMomentumScale(struct ANN_MLPPtr ptr, double val)
{
    ptr->setBackpropMomentumScale(val);
}

extern "C"
double ANN_MLP_getBackpropMomentumScale(struct ANN_MLPPtr ptr)
{
    return ptr->getBackpropMomentumScale();
}

extern "C"
void ANN_MLP_setRpropDW0(struct ANN_MLPPtr ptr, double val)
{
    ptr->setRpropDW0(val);
}

extern "C"
double ANN_MLP_getRpropDW0(struct ANN_MLPPtr ptr)
{
    return ptr->getRpropDW0();
}

extern "C"
void ANN_MLP_setRpropDWPlus(struct ANN_MLPPtr ptr, double val)
{
    ptr->setRpropDWPlus(val);
}

extern "C"
double ANN_MLP_getRpropDWPlus(struct ANN_MLPPtr ptr)
{
    return ptr->getRpropDWPlus();
}

extern "C"
void ANN_MLP_setRpropDWMinus(struct ANN_MLPPtr ptr, double val)
{
    ptr->setRpropDWMinus(val);
}

extern "C"
double ANN_MLP_getRpropDWMinus(struct ANN_MLPPtr ptr)
{
    return ptr->getRpropDWMinus();
}

extern "C"
void ANN_MLP_setRpropDWMin(struct ANN_MLPPtr ptr, double val)
{
    ptr->setRpropDWMin(val);
}

extern "C"
double ANN_MLP_getRpropDWMin(struct ANN_MLPPtr ptr)
{
    return ptr->getRpropDWMin();
}

extern "C"
void ANN_MLP_setRpropDWMax(struct ANN_MLPPtr ptr, double val)
{
    ptr->setRpropDWMax(val);
}

extern "C"
double ANN_MLP_getRpropDWMax(struct ANN_MLPPtr ptr)
{
    return ptr->getRpropDWMax();
}

extern "C"
struct TensorWrapper ANN_MLP_getWeights(struct ANN_MLPPtr ptr, int layerIdx)
{
    return TensorWrapper(ptr->getWeights(layerIdx));
}

// LogisticRegression

extern "C"
struct LogisticRegressionPtr LogisticRegression_ctor()
{
    return rescueObjectFromPtr(ml::LogisticRegression::create());
}

extern "C"
void LogisticRegression_setLearningRate(struct LogisticRegressionPtr ptr, double val)
{
    ptr->setLearningRate(val);
}

extern "C"
double LogisticRegression_getLearningRate(struct LogisticRegressionPtr ptr)
{
    return ptr->getLearningRate();
}

extern "C"
void LogisticRegression_setIterations(struct LogisticRegressionPtr ptr, int val)
{
    ptr->setIterations(val);
}

extern "C"
int LogisticRegression_getIterations(struct LogisticRegressionPtr ptr)
{
    return ptr->getIterations();
}

extern "C"
void LogisticRegression_setRegularization(struct LogisticRegressionPtr ptr, int val)
{
    ptr->setRegularization(val);
}

extern "C"
int LogisticRegression_getRegularization(struct LogisticRegressionPtr ptr)
{
    return ptr->getRegularization();
}

extern "C"
void LogisticRegression_setTrainMethod(struct LogisticRegressionPtr ptr, int val)
{
    ptr->setTrainMethod(val);
}

extern "C"
int LogisticRegression_getTrainMethod(struct LogisticRegressionPtr ptr)
{
    return ptr->getTrainMethod();
}

extern "C"
void LogisticRegression_setMiniBatchSize(struct LogisticRegressionPtr ptr, int val)
{
    ptr->setMiniBatchSize(val);
}

extern "C"
int LogisticRegression_getMiniBatchSize(struct LogisticRegressionPtr ptr)
{
    return ptr->getMiniBatchSize();
}

extern "C"
void LogisticRegression_setTermCriteria(struct LogisticRegressionPtr ptr, struct TermCriteriaWrapper val)
{
    ptr->setTermCriteria(val);
}

extern "C"
struct TermCriteriaWrapper LogisticRegression_getTermCriteria(struct LogisticRegressionPtr ptr)
{
    return ptr->getTermCriteria();
}

extern "C"
struct TensorWrapper LogisticRegression_get_learnt_thetas(struct LogisticRegressionPtr ptr)
{
    return TensorWrapper(ptr->get_learnt_thetas());
}
