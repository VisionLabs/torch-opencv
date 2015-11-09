#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/ml.hpp>

using namespace cv;

extern "C"
struct TensorWrapper TrainData_getSubVector(struct TensorWrapper vec, struct TensorWrapper idx);

struct ParamGridPtr {
    void *ptr;

    inline ml::ParamGrid * operator->() { return static_cast<ml::ParamGrid *>(ptr); }
    inline ParamGridPtr(ml::ParamGrid *ptr) { this->ptr = ptr; }
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
struct TensorPlusFloat StatModel_predict(
        struct StatModelPtr ptr, struct TensorWrapper samples, struct TensorWrapper results, int flags);

