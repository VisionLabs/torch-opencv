#include <ml.hpp>

using namespace cv;

struct TensorWrapper TrainData_getSubVector(
        struct TensorWrapper vec, struct TensorWrapper idx)
{
    return TensorWrapper(ml::TrainData::getSubVector(vec.toMat(), idx.toMat()));
}

// ParamGrid

struct ParamGridPtr {
    void *ptr;

    inline ml::ParamGrid * operator->() { return static_cast<ml::ParamGrid *>(ptr); }
    inline ParamGridPtr(ml::ParamGrid *ptr) { this->ptr = ptr; }
};

struct ParamGridPtr ParamGrid_ctor(double _minVal, double _maxVal, double _logStep)
{
    return new ml::ParamGrid(_minVal, _maxVal, _logStep);
}

struct ParamGridPtr ParamGrid_ctor_default()
{
    return new ml::ParamGrid();
}

// TrainData

struct TrainDataPtr {
    void *ptr;

    inline ml::TrainData * operator->() { return static_cast<ml::TrainData *>(ptr); }
    inline TrainDataPtr(ml::TrainData *ptr) { this->ptr = ptr; }
};

struct TrainDataPtr TrainData_ctor(
        struct TensorWrapper samples, int layout, struct TensorWrapper responses,
        struct TensorWrapper varIdx, struct TensorWrapper sampleIdx,
        struct TensorWrapper sampleWeights, struct TensorWrapper varType)
{
    return rescueObjectFromPtr(ml::TrainData::create(
            samples.toMat(), layout, responses.toMat(), varIdx.toMat(),
            sampleIdx.toMat(), sampleWeights.toMat(), varType.toMat()));
}

int TrainData_getLayout(struct TrainDataPtr ptr)
{
    return ptr->getLayout();
}

int TrainData_getNTrainSamples(struct TrainDataPtr ptr)
{
    return ptr->getNTrainSamples();
}

int TrainData_getNTestSamples(struct TrainDataPtr ptr)
{
    return ptr->getNTestSamples();
}

int TrainData_getNSamples(struct TrainDataPtr ptr)
{
    return ptr->getNSamples();
}

int TrainData_getNVars(struct TrainDataPtr ptr)
{
    return ptr->getNVars();
}

int TrainData_getNAllVars(struct TrainDataPtr ptr)
{
    return ptr->getNAllVars();
}

struct TensorWrapper TrainData_getSamples(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getSamples());
}

struct TensorWrapper TrainData_getMissing(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getMissing());
}

struct TensorWrapper TrainData_getTrainResponses(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getTrainResponses());
}

struct TensorWrapper TrainData_getTrainNormCatResponses(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getTrainNormCatResponses());
}

struct TensorWrapper TrainData_getTestResponses(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getTestResponses());
}

struct TensorWrapper TrainData_getTestNormCatResponses(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getTestNormCatResponses());
}

struct TensorWrapper TrainData_getResponses(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getResponses());
}

struct TensorWrapper TrainData_getNormCatResponses(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getNormCatResponses());
}

struct TensorWrapper TrainData_getSampleWeights(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getSampleWeights());
}

struct TensorWrapper TrainData_getTrainSampleWeights(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getTrainSampleWeights());
}

struct TensorWrapper TrainData_getTestSampleWeights(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getTestSampleWeights());
}

struct TensorWrapper TrainData_getVarIdx(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getVarIdx());
}

struct TensorWrapper TrainData_getVarType(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getVarType());
}

int TrainData_getResponseType(struct TrainDataPtr ptr)
{
    return ptr->getResponseType();
}

struct TensorWrapper TrainData_getTrainSampleIdx(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getTrainSampleIdx());
}

struct TensorWrapper TrainData_getTestSampleIdx(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getTestSampleIdx());
}

struct TensorWrapper TrainData_getDefaultSubstValues(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getDefaultSubstValues());
}

struct TensorWrapper TrainData_getClassLabels(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getClassLabels());
}

struct TensorWrapper TrainData_getCatOfs(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getCatOfs());
}

struct TensorWrapper TrainData_getCatMap(struct TrainDataPtr ptr)
{
    return TensorWrapper(ptr->getCatMap());
}

void TrainData_shuffleTrainTest(struct TrainDataPtr ptr)
{
    ptr->shuffleTrainTest();
}

struct TensorWrapper TrainData_getSample(
        struct TrainDataPtr ptr, struct TensorWrapper varIdx, int sidx)
{
    cv::Mat varIdxMat = varIdx;
    std::vector<float> output(varIdxMat.rows * varIdxMat.cols);
    ptr->getSample(varIdxMat, sidx, output.data());
    return TensorWrapper(cv::Mat(output));
}

struct TensorWrapper TrainData_getTrainSamples(
        struct TrainDataPtr ptr, int layout, bool compressSamples, bool compressVars)
{
    return TensorWrapper(ptr->getTrainSamples(layout, compressSamples, compressVars));
}

struct TensorWrapper TrainData_getValues(
        struct TrainDataPtr ptr, int vi, struct TensorWrapper sidx)
{
    cv::Mat sidxMat = sidx;
    std::vector<float> output(sidxMat.rows * sidxMat.cols);
    ptr->getValues(vi, sidxMat, output.data());
    return TensorWrapper(cv::Mat(output));
}

struct TensorWrapper TrainData_getNormCatValues(
        struct TrainDataPtr ptr, int vi, struct TensorWrapper sidx)
{
    cv::Mat sidxMat = sidx;
    std::vector<int> output(sidxMat.rows * sidxMat.cols);
    ptr->getNormCatValues(vi, sidxMat, output.data());
    return TensorWrapper(cv::Mat(output));
}

void TrainData_setTrainTestSplit(struct TrainDataPtr ptr, int count, bool shuffle)
{
    ptr->setTrainTestSplit(count, shuffle);
}

void TrainData_setTrainTestSplitRatio(struct TrainDataPtr ptr, double ratio, bool shuffle)
{
    ptr->setTrainTestSplitRatio(ratio, shuffle);
}

