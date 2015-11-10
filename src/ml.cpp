#include <ml.hpp>

extern "C"
struct TensorWrapper TrainData_getSubVector(
        struct TensorWrapper vec, struct TensorWrapper idx)
{
    return TensorWrapper(ml::TrainData::getSubVector(vec.toMat(), idx.toMat()));
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
struct TensorPlusFloat StatModel_predict(
        struct StatModelPtr ptr, struct TensorWrapper samples, struct TensorWrapper results, int flags)
{
    TensorPlusFloat retval;

    if (results.isNull()) {
        cv::Mat resultsMat;
        retval.val = ptr->predict(samples.toMat(), resultsMat, flags);
        new (&retval.tensor) TensorWrapper(resultsMat);
    } else {
        retval.val = ptr->predict(samples.toMat(), results.toMat(), flags);
        retval.tensor = results;
    }

    return retval;
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

    return retval;
}
