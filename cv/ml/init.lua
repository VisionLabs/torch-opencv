local cv = require 'cv._env'

local ffi = require 'ffi'

local ml = {}

ml.ANN_MLP_BACKPROP = 0
ml.ANN_MLP_GAUSSIAN = 2
ml.ANN_MLP_IDENTITY = 0
ml.ANN_MLP_NO_INPUT_SCALE = 2
ml.ANN_MLP_NO_OUTPUT_SCALE = 4
ml.ANN_MLP_RPROP = 1
ml.ANN_MLP_SIGMOID_SYM = 1
ml.ANN_MLP_UPDATE_WEIGHTS = 1
ml.BOOST_DISCRETE = 0
ml.BOOST_GENTLE = 3
ml.BOOST_LOGIT = 2
ml.BOOST_REAL = 1
ml.Boost_DISCRETE = 0
ml.Boost_GENTLE = 3
ml.Boost_LOGIT = 2
ml.Boost_REAL = 1
ml.COL_SAMPLE = 1
ml.DTREES_PREDICT_AUTO = 0
ml.DTREES_PREDICT_MASK = 768
ml.DTREES_PREDICT_MAX_VOTE = 512
ml.DTREES_PREDICT_SUM = 256
ml.DTrees_PREDICT_AUTO = 0
ml.DTrees_PREDICT_MASK = 768
ml.DTrees_PREDICT_MAX_VOTE = 512
ml.DTrees_PREDICT_SUM = 256
ml.EM_COV_MAT_DEFAULT = 1
ml.EM_COV_MAT_DIAGONAL = 1
ml.EM_COV_MAT_GENERIC = 2
ml.EM_COV_MAT_SPHERICAL = 0
ml.EM_DEFAULT_MAX_ITERS = 100
ml.EM_DEFAULT_NCLUSTERS = 5
ml.EM_START_AUTO_STEP = 0
ml.EM_START_E_STEP = 1
ml.EM_START_M_STEP = 2
ml.KNEAREST_BRUTE_FORCE = 1
ml.KNEAREST_KDTREE = 2
ml.KNearest_BRUTE_FORCE = 1
ml.KNearest_KDTREE = 2
ml.LOGISTIC_REGRESSION_BATCH = 0
ml.LOGISTIC_REGRESSION_MINI_BATCH = 1
ml.LOGISTIC_REGRESSION_REG_DISABLE = -1
ml.LOGISTIC_REGRESSION_REG_L1 = 0
ml.LOGISTIC_REGRESSION_REG_L2 = 1
ml.LogisticRegression_BATCH = 0
ml.LogisticRegression_MINI_BATCH = 1
ml.LogisticRegression_REG_DISABLE = -1
ml.LogisticRegression_REG_L1 = 0
ml.LogisticRegression_REG_L2 = 1
ml.ROW_SAMPLE = 0
ml.STAT_MODEL_COMPRESSED_INPUT = 2
ml.STAT_MODEL_PREPROCESSED_INPUT = 4
ml.STAT_MODEL_RAW_OUTPUT = 1
ml.STAT_MODEL_UPDATE_MODEL = 1
ml.SVM_C = 0
ml.SVM_CHI2 = 4
ml.SVM_COEF = 4
ml.SVM_CUSTOM = -1
ml.SVM_C_SVC = 100
ml.SVM_DEGREE = 5
ml.SVM_EPS_SVR = 103
ml.SVM_GAMMA = 1
ml.SVM_INTER = 5
ml.SVM_LINEAR = 0
ml.SVM_NU = 3
ml.SVM_NU_SVC = 101
ml.SVM_NU_SVR = 104
ml.SVM_ONE_CLASS = 102
ml.SVM_P = 2
ml.SVM_POLY = 1
ml.SVM_RBF = 2
ml.SVM_SIGMOID = 3
ml.StatModel_COMPRESSED_INPUT = 2
ml.StatModel_PREPROCESSED_INPUT = 4
ml.StatModel_RAW_OUTPUT = 1
ml.StatModel_UPDATE_MODEL = 1
ml.TEST_ERROR = 0
ml.TRAIN_ERROR = 1
ml.VAR_CATEGORICAL = 1
ml.VAR_NUMERICAL = 0
ml.VAR_ORDERED = 0

ffi.cdef[[
struct TensorWrapper randMVNormal(
        struct TensorWrapper mean, struct TensorWrapper cov, int nsamples, struct TensorWrapper samples);

struct TensorArray createConcentricSpheresTestSet(
        int nsamples, int nfeatures, int nclasses, struct TensorWrapper samples, struct TensorWrapper responses);
]]

local C = ffi.load(cv.libPath('ml'))

function ml.randMVNormal(t)
    local argRules = {
        {"mean", required = true},
        {"cov", required = true},
        {"nsamples", required = true},
        {"samples", default = nil}
    }
    local mean, cov, nsamples, samples = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.randMVNormal(
        cv.wrap_tensor(mean), cv.wrap_tensor(cov), nsamples, cv.wrap_tensor(samples)))
end

function ml.createConcentricSpheresTestSet(t)
    local argRules = {
        {"nsamples", required = true},
        {"nfeatures", required = true},
        {"nclasses", required = true},
        {"samples", default = nil},
        {"responses", default = nil}
    }
    local nsamples, nfeatures, nclasses, samples, responses = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.createConcentricSpheresTestSet(
        nsamples, nfeatures, nclasses, cv.wrap_tensor(samples), cv.wrap_tensor(responses)))
end

--- ***************** Classes *****************
require 'cv.Classes'

local Classes = ffi.load(cv.libPath('Classes'))

ffi.cdef[[
struct PtrWrapper ParamGrid_ctor(double _minVal, double _maxVal, double _logStep);

struct PtrWrapper ParamGrid_ctor_default();

void ParamGrid_dtor(struct PtrWrapper ptr);

struct PtrWrapper TrainData_ctor(
        struct TensorWrapper samples, int layout, struct TensorWrapper responses,
        struct TensorWrapper varIdx, struct TensorWrapper sampleIdx,
        struct TensorWrapper sampleWeights, struct TensorWrapper varType);

void TrainData_dtor(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getSubVector(struct TensorWrapper vec, struct TensorWrapper idx);

int TrainData_getLayout(struct PtrWrapper ptr);

int TrainData_getNTrainSamples(struct PtrWrapper ptr);

int TrainData_getNTestSamples(struct PtrWrapper ptr);

int TrainData_getNSamples(struct PtrWrapper ptr);

int TrainData_getNVars(struct PtrWrapper ptr);

int TrainData_getNAllVars(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getSamples(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getMissing(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getTrainResponses(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getTrainNormCatResponses(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getTestResponses(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getTestNormCatResponses(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getResponses(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getNormCatResponses(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getSampleWeights(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getTrainSampleWeights(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getTestSampleWeights(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getVarIdx(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getVarType(struct PtrWrapper ptr);

int TrainData_getResponseType(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getTrainSampleIdx(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getTestSampleIdx(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getDefaultSubstValues(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getClassLabels(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getCatOfs(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getCatMap(struct PtrWrapper ptr);

void TrainData_shuffleTrainTest(struct PtrWrapper ptr);

struct TensorWrapper TrainData_getSample(
        struct PtrWrapper ptr, struct TensorWrapper varIdx, int sidx);

struct TensorWrapper TrainData_getTrainSamples(
        struct PtrWrapper ptr, int layout, bool compressSamples, bool compressVars);

struct TensorWrapper TrainData_getValues(
        struct PtrWrapper ptr, int vi, struct TensorWrapper sidx);

struct TensorWrapper TrainData_getNormCatValues(
        struct PtrWrapper ptr, int vi, struct TensorWrapper sidx);

void TrainData_setTrainTestSplit(struct PtrWrapper ptr, int count, bool shuffle);

void TrainData_setTrainTestSplitRatio(struct PtrWrapper ptr, double ratio, bool shuffle);

int StatModel_getVarCount(struct PtrWrapper ptr);

bool StatModel_empty(struct PtrWrapper ptr);

bool StatModel_isTrained(struct PtrWrapper ptr);

bool StatModel_isClassifier(struct PtrWrapper ptr);

bool StatModel_train(struct PtrWrapper ptr, struct PtrWrapper trainData, int flags);

bool StatModel_train_Mat(
        struct PtrWrapper ptr, struct TensorWrapper samples, int layout, struct TensorWrapper responses);

struct TensorPlusFloat StatModel_calcError(
        struct PtrWrapper ptr, struct PtrWrapper data, bool test, struct TensorWrapper resp);

float StatModel_predict(
        struct PtrWrapper ptr, struct TensorWrapper samples, struct TensorWrapper results, int flags);

struct PtrWrapper NormalBayesClassifier_ctor();

struct TensorArrayPlusFloat NormalBayesClassifier_predictProb(
        struct PtrWrapper ptr, struct TensorWrapper inputs,
        struct TensorWrapper outputs, struct TensorWrapper outputProbs, int flags);

struct PtrWrapper KNearest_ctor();

void KNearest_setDefaultK(struct PtrWrapper ptr, int val);

int KNearest_getDefaultK(struct PtrWrapper ptr);

void KNearest_setIsClassifier(struct PtrWrapper ptr, bool val);

bool KNearest_getIsClassifier(struct PtrWrapper ptr);

void KNearest_setEmax(struct PtrWrapper ptr, int val);

int KNearest_getEmax(struct PtrWrapper ptr);

void KNearest_setAlgorithmType(struct PtrWrapper ptr, int val);

int KNearest_getAlgorithmType(struct PtrWrapper ptr);

float KNearest_findNearest(
        struct PtrWrapper ptr, struct TensorWrapper samples, int k,
        struct TensorWrapper results, struct TensorWrapper neighborResponses,
        struct TensorWrapper dist);

struct PtrWrapper SVM_ctor();

void SVM_setType(struct PtrWrapper ptr, int val);

int SVM_getType(struct PtrWrapper ptr);

void SVM_setGamma(struct PtrWrapper ptr, double val);

double SVM_getGamma(struct PtrWrapper ptr);

void SVM_setCoef0(struct PtrWrapper ptr, double val);

double SVM_getCoef0(struct PtrWrapper ptr);

void SVM_setDegree(struct PtrWrapper ptr, double val);

double SVM_getDegree(struct PtrWrapper ptr);

void SVM_setC(struct PtrWrapper ptr, double val);

double SVM_getC(struct PtrWrapper ptr);

void SVM_setNu(struct PtrWrapper ptr, double val);

double SVM_getNu(struct PtrWrapper ptr);

void SVM_setP(struct PtrWrapper ptr, double val);

double SVM_getP(struct PtrWrapper ptr);

void SVM_setClassWeights(struct PtrWrapper ptr, struct TensorWrapper val);

struct TensorWrapper SVM_getClassWeights(struct PtrWrapper ptr);

void SVM_setTermCriteria(struct PtrWrapper ptr, struct TermCriteriaWrapper val);

struct TermCriteriaWrapper SVM_getTermCriteria(struct PtrWrapper ptr);

int SVM_getKernelType(struct PtrWrapper ptr);

void SVM_setKernel(struct PtrWrapper ptr, int val);

//void SVM_setCustomKernel(struct PtrWrapper ptr, struct PtrWrapper val);

bool SVM_trainAuto(
        struct PtrWrapper ptr, struct PtrWrapper data, int kFold, struct PtrWrapper Cgrid,
        struct PtrWrapper gammaGrid, struct PtrWrapper pGrid, struct PtrWrapper nuGrid,
        struct PtrWrapper coeffGrid, struct PtrWrapper degreeGrid, bool balanced);

struct TensorWrapper SVM_getSupportVectors(struct PtrWrapper ptr);

struct TensorArrayPlusDouble SVM_getDecisionFunction(
        struct PtrWrapper ptr, int i, struct TensorWrapper alpha, struct TensorWrapper svidx);

struct PtrWrapper SVM_getDefaultGrid(int param_id);

struct PtrWrapper EM_ctor();

void EM_setClustersNumber(struct PtrWrapper ptr, int val);

int EM_getClustersNumber(struct PtrWrapper ptr);

void EM_setCovarianceMatrixType(struct PtrWrapper ptr, int val);

int EM_getCovarianceMatrixType(struct PtrWrapper ptr);

void EM_setTermCriteria(struct PtrWrapper ptr, struct TermCriteriaWrapper val);

struct TermCriteriaWrapper EM_getTermCriteria(struct PtrWrapper ptr);

struct TensorWrapper EM_getWeights(struct PtrWrapper ptr);

struct TensorWrapper EM_getMeans(struct PtrWrapper ptr);

struct TensorArray EM_getCovs(struct PtrWrapper ptr);

struct Vec2dWrapper EM_predict2(
        struct PtrWrapper ptr, struct TensorWrapper sample, struct TensorWrapper probs);

bool EM_trainEM(
        struct PtrWrapper ptr, struct TensorWrapper samples,
        struct TensorWrapper logLikelihoods,
        struct TensorWrapper labels, struct TensorWrapper probs);

bool EM_trainE(
        struct PtrWrapper ptr, struct TensorWrapper samples, struct TensorWrapper means0,
        struct TensorWrapper covs0, struct TensorWrapper weights0,
        struct TensorWrapper logLikelihoods, struct TensorWrapper labels,
        struct TensorWrapper probs);

bool EM_trainM(
        struct PtrWrapper ptr, struct TensorWrapper samples, struct TensorWrapper probs0,
        struct TensorWrapper logLikelihoods, struct TensorWrapper labels,
        struct TensorWrapper probs);

struct Node {
    double value;
    int classIdx;
    int parent;
    int left;
    int right;
    int defaultDir;
    int split;
};

struct Split {
    int varIdx;
    bool inversed;
    float quality;
    int next;
    float c;
    int subsetOfs;
};

struct ConstNodeArray {
    struct Node *ptr;
    int size;
};

struct ConstSplitArray {
    struct Split *ptr;
    int size;
};

struct PtrWrapper DTrees_ctor();

void DTrees_setMaxCategories(struct PtrWrapper ptr, int val);

int DTrees_getMaxCategories(struct PtrWrapper ptr);

void DTrees_setMaxDepth(struct PtrWrapper ptr, int val);

int DTrees_getMaxDepth(struct PtrWrapper ptr);

void DTrees_setMinSampleCount(struct PtrWrapper ptr, int val);

int DTrees_getMinSampleCount(struct PtrWrapper ptr);

void DTrees_setCVFolds(struct PtrWrapper ptr, int val);

int DTrees_getCVFolds(struct PtrWrapper ptr);

void DTrees_setUseSurrogates(struct PtrWrapper ptr, bool val);

bool DTrees_getUseSurrogates(struct PtrWrapper ptr);

void DTrees_setUse1SERule(struct PtrWrapper ptr, bool val);

bool DTrees_getUse1SERule(struct PtrWrapper ptr);

void DTrees_setTruncatePrunedTree(struct PtrWrapper ptr, bool val);

bool DTrees_getTruncatePrunedTree(struct PtrWrapper ptr);

void DTrees_setRegressionAccuracy(struct PtrWrapper ptr, float val);

float DTrees_getRegressionAccuracy(struct PtrWrapper ptr);

void DTrees_setPriors(struct PtrWrapper ptr, struct TensorWrapper val);

struct TensorWrapper DTrees_getPriors(struct PtrWrapper ptr);

struct TensorWrapper DTrees_getRoots(struct PtrWrapper ptr);

struct ConstNodeArray DTrees_getNodes(struct PtrWrapper ptr);

struct ConstSplitArray DTrees_getSplits(struct PtrWrapper ptr);

struct TensorWrapper DTrees_getSubsets(struct PtrWrapper ptr);

struct PtrWrapper RTrees_ctor();

void RTrees_setCalculateVarImportance(struct PtrWrapper ptr, bool val);

bool RTrees_getCalculateVarImportance(struct PtrWrapper ptr);

void RTrees_setActiveVarCount(struct PtrWrapper ptr, int val);

int RTrees_getActiveVarCount(struct PtrWrapper ptr);

void RTrees_setTermCriteria(struct PtrWrapper ptr, struct TermCriteriaWrapper val);

struct PtrWrapper RTrees_load(const char *filename, const char *objname);

struct PtrWrapper Boost_ctor();

void Boost_setBoostType(struct PtrWrapper ptr, int val);

int Boost_getBoostType(struct PtrWrapper ptr);

void Boost_setWeakCount(struct PtrWrapper ptr, int val);

int Boost_getWeakCount(struct PtrWrapper ptr);

void Boost_setWeightTrimRate(struct PtrWrapper ptr, double val);

double Boost_getWeightTrimRate(struct PtrWrapper ptr);

struct PtrWrapper ANN_MLP_ctor();

void ANN_MLP_setTrainMethod(struct PtrWrapper ptr, int method, double param1, double param2);

int ANN_MLP_getTrainMethod(struct PtrWrapper ptr);

void ANN_MLP_setActivationFunction(struct PtrWrapper ptr, int type, double param1, double param2);

void ANN_MLP_setLayerSizes(struct PtrWrapper ptr, struct TensorWrapper val);

struct TensorWrapper ANN_MLP_getLayerSizes(struct PtrWrapper ptr);

void ANN_MLP_setTermCriteria(struct PtrWrapper ptr, struct TermCriteriaWrapper val);

struct TermCriteriaWrapper ANN_MLP_getTermCriteria(struct PtrWrapper ptr);

void ANN_MLP_setBackpropWeightScale(struct PtrWrapper ptr, double val);

double ANN_MLP_getBackpropWeightScale(struct PtrWrapper ptr);

void ANN_MLP_setBackpropMomentumScale(struct PtrWrapper ptr, double val);

double ANN_MLP_getBackpropMomentumScale(struct PtrWrapper ptr);

void ANN_MLP_setRpropDW0(struct PtrWrapper ptr, double val);

double ANN_MLP_getRpropDW0(struct PtrWrapper ptr);

void ANN_MLP_setRpropDWPlus(struct PtrWrapper ptr, double val);

double ANN_MLP_getRpropDWPlus(struct PtrWrapper ptr);

void ANN_MLP_setRpropDWMinus(struct PtrWrapper ptr, double val);

double ANN_MLP_getRpropDWMinus(struct PtrWrapper ptr);

void ANN_MLP_setRpropDWMin(struct PtrWrapper ptr, double val);

double ANN_MLP_getRpropDWMin(struct PtrWrapper ptr);

void ANN_MLP_setRpropDWMax(struct PtrWrapper ptr, double val);

double ANN_MLP_getRpropDWMax(struct PtrWrapper ptr);

struct TensorWrapper ANN_MLP_getWeights(struct PtrWrapper ptr, int layerIdx);

struct PtrWrapper LogisticRegression_ctor();

void LogisticRegression_setLearningRate(struct PtrWrapper ptr, double val);

double LogisticRegression_getLearningRate(struct PtrWrapper ptr);

void LogisticRegression_setIterations(struct PtrWrapper ptr, int val);

int LogisticRegression_getIterations(struct PtrWrapper ptr);

void LogisticRegression_setRegularization(struct PtrWrapper ptr, int val);

int LogisticRegression_getRegularization(struct PtrWrapper ptr);

void LogisticRegression_setTrainMethod(struct PtrWrapper ptr, int val);

int LogisticRegression_getTrainMethod(struct PtrWrapper ptr);

void LogisticRegression_setMiniBatchSize(struct PtrWrapper ptr, int val);

int LogisticRegression_getMiniBatchSize(struct PtrWrapper ptr);

void LogisticRegression_setTermCriteria(struct PtrWrapper ptr, struct TermCriteriaWrapper val);

struct TermCriteriaWrapper LogisticRegression_getTermCriteria(struct PtrWrapper ptr);

struct TensorWrapper LogisticRegression_get_learnt_thetas(struct PtrWrapper ptr);

struct PtrWrapper ANN_MLP_load(const char *filename, const char *objname);

struct PtrWrapper LogisticRegression_load(const char *filename, const char *objname);

struct PtrWrapper Boost_load(const char *filename, const char *objname);

struct PtrWrapper RTrees_load(const char *filename, const char *objname);

struct PtrWrapper DTrees_load(const char *filename, const char *objname);

struct PtrWrapper EM_load(const char *filename, const char *objname);

struct PtrWrapper SVM_load(const char *filename, const char *objname);

struct PtrWrapper KNearest_load(const char *filename, const char *objname);

struct PtrWrapper NormalBayesClassifier_load(const char *filename, const char *objname);  
]]

-- ParamGrid

do
    local ParamGrid = torch.class('ml.ParamGrid', ml)

    function ParamGrid:__init(t)
    	if type(t) ~= 'table' then
    		self.ptr = t
            return
    	end

        if t[1] or t._minVal then
            local argRules = {
                {"_minVal", required = true},
                {"_maxVal", required = true},
                {"_logStep", required = true}
            }
            local _minVal, _maxVal, _logStep = cv.argcheck(t, argRules)

            self.ptr = ffi.gc(C.ParamGrid_ctor(_minVal, _maxVal, _logStep), C.ParamGrid_dtor)
        else
            self.ptr = ffi.gc(C.ParamGrid_ctor_default(), C.ParamGrid_dtor)
        end
    end
end

-- TrainData

do
    local TrainData = torch.class('ml.TrainData', ml)

    function TrainData:__init(t)
        local argRules = {
            {"samples", required = true},
            {"layout", required = true},
            {"responses", required = true},
            {"varIdx", default = nil},
            {"sampleIdx", default = nil},
            {"sampleWeights", default = nil},
            {"varType", default = nil}
        }
        local samples, layout, responses, varIdx, sampleIdx, sampleWeights, varType
            = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(
            C.TrainData_ctor(
                cv.wrap_tensor(samples), layout, cv.wrap_tensor(responses), cv.wrap_tensor(varIdx),
                cv.wrap_tensor(sampleIdx), cv.wrap_tensor(sampleWeights), cv.wrap_tensor(varType)),
            C.TrainData_dtor)
    end

    function TrainData:loadFromCSV(t)
        local argRules = {
            {"filename", required = true},
            {"headerLineCount", required = true},
            {"responseStartIdx", default = -1},
            {"responseEndIdx", default = -1},
            {"varTypeSpec", default = ''},
            {"delimiter", default = ','},
            {"missch", default = '?'}
        }
        local filename, headerLineCount, responseStartIdx, responseEndIdx, 
            varTypeSpec, delimiter, missch = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(
            C.TrainData_loadFromCSV(filename, headerLineCount, 
                responseStartIdx, responseEndIdx, varTypeSpec, delimiter, missch),
            C.TrainData_dtor)
    end

    function TrainData:getSubVector(t)
        local argRules = {
            {"vec", required = true},
            {"idx", required = true}
        }
        local vec, idx = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(
            C.TrainData_getSubVector(cv.wrap_tensor(vec), cv.wrap_tensor(idx)))
    end

    function TrainData:getLayout()
        return C.TrainData_getLayout(self.ptr)
    end

    function TrainData:getNTrainSamples()
        return C.TrainData_getNTrainSamples(self.ptr)
    end

    function TrainData:getNTestSamples()
        return C.TrainData_getNTestSamples(self.ptr)
    end

    function TrainData:getNSamples()
        return C.TrainData_getNSamples(self.ptr)
    end

    function TrainData:getNVars()
        return C.TrainData_getNVars(self.ptr)
    end

    function TrainData:getNAllVars()
        return C.TrainData_getNAllVars(self.ptr)
    end

    function TrainData:getSamples()
        return cv.unwrap_tensors(C.TrainData_getSamples(self.ptr))
    end

    function TrainData:getMissing()
        return cv.unwrap_tensors(C.TrainData_getMissing(self.ptr))
    end

    function TrainData:getTrainResponses()
        return cv.unwrap_tensors(C.TrainData_getTrainResponses(self.ptr))
    end

    function TrainData:getTrainNormCatResponses()
        return cv.unwrap_tensors(C.TrainData_getTrainNormCatResponses(self.ptr))
    end

    function TrainData:getTestResponses()
        return cv.unwrap_tensors(C.TrainData_getTestResponses(self.ptr))
    end

    function TrainData:getTestNormCatResponses()
        return cv.unwrap_tensors(C.TrainData_getTestNormCatResponses(self.ptr))
    end

    function TrainData:getResponses()
        return cv.unwrap_tensors(C.TrainData_getResponses(self.ptr))
    end

    function TrainData:getNormCatResponses()
        return cv.unwrap_tensors(C.TrainData_getNormCatResponses(self.ptr))
    end

    function TrainData:getSampleWeights()
        return cv.unwrap_tensors(C.TrainData_getSampleWeights(self.ptr))
    end

    function TrainData:getTrainSampleWeights()
        return cv.unwrap_tensors(C.TrainData_getTrainSampleWeights(self.ptr))
    end

    function TrainData:getTestSampleWeights()
        return cv.unwrap_tensors(C.TrainData_getTestSampleWeights(self.ptr))
    end

    function TrainData:getVarIdx()
        return cv.unwrap_tensors(C.TrainData_getVarIdx(self.ptr))
    end

    function TrainData:getVarType()
        return cv.unwrap_tensors(C.TrainData_getVarType(self.ptr))
    end

    function TrainData:getResponseType()
        return C.TrainData_getResponseType(self.ptr)
    end

    function TrainData:getTrainSampleIdx()
        return cv.unwrap_tensors(C.TrainData_getTrainSampleIdx(self.ptr))
    end

    function TrainData:getTestSampleIdx()
        return cv.unwrap_tensors(C.TrainData_getTestSampleIdx(self.ptr))
    end

    function TrainData:getDefaultSubstValues()
        return cv.unwrap_tensors(C.TrainData_getDefaultSubstValues(self.ptr))
    end

    function TrainData:getClassLabels()
        return cv.unwrap_tensors(C.TrainData_getClassLabels(self.ptr))
    end

    function TrainData:getCatOfs()
        return cv.unwrap_tensors(C.TrainData_getCatOfs(self.ptr))
    end

    function TrainData:getCatMap()
        return cv.unwrap_tensors(C.TrainData_getCatMap(self.ptr))
    end

    function TrainData:shuffleTrainTest()
        C.TrainData_shuffleTrainTest(self.ptr)
    end

    function TrainData:getCatCount(t)
        local argRules = {
            {"vi", required = true}
        }
        local vi = cv.argcheck(t, argRules)

        return C.TrainData_getCatCount(self.ptr, vi)
    end

    function TrainData:getSample(t)
        local argRules = {
            {"varIdx", required = true},
            {"sidx", required = true}
        }
        local varIdx, sidx = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.TrainData_getSample(
            self.ptr, cv.wrap_tensor(varIdx), sidx))
    end

    function TrainData:getTrainSamples(t)
        local argRules = {
            {"layout", default = cv.ROW_SAMPLE},
            {"compressSamples", default = true},
            {"compressVars", default = true}
        }
        local layout, compressSamples, compressVars = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.TrainData_getTrainSamples(
            self.ptr, layout, compressSamples, compressVars))
    end

    function TrainData:getValues(t)
        local argRules = {
            {"vi", required = true},
            {"sidx", required = true}
        }
        local vi, sidx = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.TrainData_getValues(
            self.ptr, vi, cv.wrap_tensor(sidx)))
    end

    function TrainData:getNormCatValues(t)
        local argRules = {
            {"vi", required = true},
            {"sidx", required = true}
        }
        local vi, sidx = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.TrainData_getNormCatValues(
            self.ptr, vi, cv.wrap_tensor(sidx)))
    end

    function TrainData:setTrainTestSplit(t)
        local argRules = {
            {"count", required = true},
            {"shuffle", default = true}
        }
        local count, shuffle = cv.argcheck(t, argRules)

        C.TrainData_setTrainTestSplit(self.ptr, count, shuffle)
    end

    function TrainData:setTrainTestSplitRatio(t)
        local argRules = {
            {"ratio", required = true},
            {"shuffle", default = true}
        }
        local ratio, shuffle = cv.argcheck(t, argRules)

        C.TrainData_setTrainTestSplitRatio(self.ptr, ratio, shuffle)
    end
end

-- StatModel

do
    local StatModel = torch.class('ml.StatModel', 'cv.Algorithm', ml)

    function StatModel:getVarCount()
        return C.StatModel_getVarCount(self.ptr)
    end

    function StatModel:empty()
        return C.StatModel_empty(self.ptr)
    end

    function StatModel:isTrained()
        return C.StatModel_isTrained(self.ptr)
    end

    function StatModel:isClassifier()
        return C.StatModel_isClassifier(self.ptr)
    end

    function StatModel:train(t)
        if torch.isTensor(t[1] or t.samples) then
            local argRules = {
                {"samples", required = true},
                {"layout", required = true},
                {"responses", required = true}
            }
            local samples, layout, responses = cv.argcheck(t, argRules)

            return C.StatModel_train_Mat(self.ptr, cv.wrap_tensor(samples), layout, cv.wrap_tensor(responses))
        else
            local argRules = {
                {"trainData", required = true},
                {"flags", default = 0}
            }
            local trainData, flags = cv.argcheck(t, argRules)

            return C.StatModel_train(self.ptr, trainData.ptr, flags)
        end
    end

    function StatModel:calcError(t)
        local argRules = {
            {"data", required = true},
            {"test", required = true},
            {"resp", default = nil}
        }
        local data, test, resp = cv.argcheck(t, argRules)

        local result = C.StatModel_calcError(self.ptr, data.ptr, test, cv.wrap_tensor(resp))
        return result.val, cv.unwrap_tensors(result.tensor)
    end

    function StatModel:predict(t)
        local argRules = {
            {"samples", required = true},
            {"results", default = nil},
            {"flags", default = 0}
        }
        local samples, results, flags = cv.argcheck(t, argRules)

        return C.StatModel_predict(self.ptr, cv.wrap_tensor(samples), cv.wrap_tensor(results), flags)
    end
end

-- NormalBayesClassifier

do
	local NormalBayesClassifier = torch.class('ml.NormalBayesClassifier', 'ml.StatModel', ml)

    function NormalBayesClassifier:load(t)
        local argRules = {
            {"filename", required = true},
            {"objname", default = ''}
        }
        local filename, objname = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.NormalBayesClassifier_load(filename, objname), Classes.Algorithm_dtor)
    end

	function NormalBayesClassifier:__init()
		self.ptr = ffi.gc(C.NormalBayesClassifier_ctor(), Classes.Algorithm_dtor)
	end

	function NormalBayesClassifier:predictProb(t)
		local argRules = {
			{"inputs", required = true},
			{"outputs", required = true},
			{"outputProbs", required = true},
			{"flags", default = 0}
		}
		local inputs, outputs, outputProbs, flags = cv.argcheck(t, argRules)

		local result = C.NormalBayesClassifier_predictProb(
			self.ptr, cv.wrap_tensor(inputs), cv.wrap_tensor(outputs),
			cv.wrap_tensor(outputProbs), flags)
		return result.val, cv.unwrap_tensors(result.tensors)
	end
end

-- KNearest

do
    local KNearest = torch.class('ml.KNearest', 'ml.StatModel', ml)

    function KNearest:load(t)
        local argRules = {
            {"filename", required = true},
            {"objname", default = ''}
        }
        local filename, objname = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.KNearest_load(filename, objname), Classes.Algorithm_dtor)
    end

    function KNearest:__init()
        self.ptr = ffi.gc(C.KNearest_ctor(), Classes.Algorithm_dtor)
    end

    function KNearest:setDefaultK(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.KNearest_setDefaultK(self.ptr, val)
    end

    function KNearest:getDefaultK()
        return C.KNearest_getDefaultK(self.ptr)
    end

    function KNearest:setIsClassifier(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.KNearest_setIsClassifier(self.ptr, val)
    end

    function KNearest:getIsClassifier()
        return C.KNearest_getIsClassifier(self.ptr)
    end

    function KNearest:setEmax(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.KNearest_setEmax(self.ptr, val)
    end

    function KNearest:getEmax()
        return C.KNearest_getEmax(self.ptr)
    end

    function KNearest:setAlgorithmType(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.KNearest_setAlgorithmType(self.ptr, val)
    end

    function KNearest:getAlgorithmType()
        return C.KNearest_getAlgorithmType(self.ptr)
    end

    function KNearest:findNearest(t)
        local argRules = {
            {"samples", required = true},
            {"k", required = true},
            {"results", default = nil},
            {"neighborResponses", default = nil},
            {"dist", default = nil}
        }
        local samples, k, results, neighborResponses, dist = cv.argcheck(t, argRules)

        return C.KNearest_findNearest(self.ptr, cv.wrap_tensor(samples), k,
            cv.wrap_tensor(results), cv.wrap_tensor(neighborResponses), cv.wrap_tensor(dist))
    end
end

-- SVM

do
	local SVM = torch.class('ml.SVM', 'ml.StatModel', ml)

    function SVM:load(t)
        local argRules = {
            {"filename", required = true},
            {"objname", default = ''}
        }
        local filename, objname = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.SVM_load(filename, objname), Classes.Algorithm_dtor)
    end

	function SVM:__init()
		self.ptr = ffi.gc(C.SVM_ctor(), Classes.Algorithm_dtor)
	end

	function SVM:setType(t)
	    local argRules = {
	        {"val", required = true}
	    }
	    local val = cv.argcheck(t, argRules)

	    C.SVM_setType(self.ptr, val)
	end

	function SVM:getType()
	    return C.SVM_getType(self.ptr)
	end

	function SVM:setGamma(t)
	    local argRules = {
	        {"val", required = true}
	    }
	    local val = cv.argcheck(t, argRules)

	    C.SVM_setGamma(self.ptr, val)
	end

	function SVM:getGamma()
	    return C.SVM_getGamma(self.ptr)
	end

	function SVM:setCoef0(t)
	    local argRules = {
	        {"val", required = true}
	    }
	    local val = cv.argcheck(t, argRules)

	    C.SVM_setCoef0(self.ptr, val)
	end

	function SVM:getCoef0()
	    return C.SVM_getCoef0(self.ptr)
	end

	function SVM:setDegree(t)
	    local argRules = {
	        {"val", required = true}
	    }
	    local val = cv.argcheck(t, argRules)

	    C.SVM_setDegree(self.ptr, val)
	end

	function SVM:getDegree()
	    return C.SVM_getDegree(self.ptr)
	end

	function SVM:setC(t)
	    local argRules = {
	        {"val", required = true}
	    }
	    local val = cv.argcheck(t, argRules)

	    C.SVM_setC(self.ptr, val)
	end

	function SVM:getC()
	    return C.SVM_getC(self.ptr)
	end

	function SVM:setNu(t)
	    local argRules = {
	        {"val", required = true}
	    }
	    local val = cv.argcheck(t, argRules)

	    C.SVM_setNu(self.ptr, val)
	end

	function SVM:getNu()
	    return C.SVM_getNu(self.ptr)
	end

	function SVM:setP(t)
	    local argRules = {
	        {"val", required = true}
	    }
	    local val = cv.argcheck(t, argRules)

	    C.SVM_setP(self.ptr, val)
	end

	function SVM:getP()
	    return C.SVM_getP(self.ptr)
	end

	function SVM:setClassWeights(t)
	    local argRules = {
	        {"val", required = true}
	    }
	    local val = cv.argcheck(t, argRules)

	    C.SVM_setClassWeights(self.ptr, cv.wrap_tensor(val))
	end

	function SVM:getClassWeights()
	    return cv.unwrap_tensors(C.SVM_getClassWeights(self.ptr))
	end

	function SVM:setTermCriteria(t)
	    local argRules = {
	        {"val", required = true, operator = cv.TermCriteria}
	    }
	    local val = cv.argcheck(t, argRules)

	    C.SVM_setTermCriteria(self.ptr, val)
	end

	function SVM:getTermCriteria()
	    return C.SVM_getTermCriteria(self.ptr)
	end

	function SVM:getKernelType()
	    return C.SVM_getKernelType(self.ptr)
	end

	function SVM:setKernel(t)
	    local argRules = {
	        {"val", required = true}
	    }
	    local val = cv.argcheck(t, argRules)

	    C.SVM_setKernel(self.ptr, val)
	end

	-- TODO this
	-- function SVM:setCustomKernel(t)
	--     local argRules = {
	--         {"val", required = true}
	--     }
	--     local val = cv.argcheck(t, argRules)

	--     C.SVM_setCustomKernel(self.ptr, val)
	-- end

	function SVM:trainAuto(t)
		local argRules = {
			{"data", required = true},
			{"kFold", default = 10},
			{"Cgrid", default = SVM.getDefaultGrid{ml.SVM_C}},
			{"gammaGrid", default = SVM.getDefaultGrid{ml.SVM_GAMMA}},
			{"pGrid", default = SVM.getDefaultGrid{ml.SVM_P}},
			{"nuGrid", default = SVM.getDefaultGrid{ml.SVM_NU}},
			{"coeffGrid", default = SVM.getDefaultGrid{ml.SVM_COEF}},
			{"degreeGrid", default = SVM.getDefaultGrid{ml.SVM_DEGREE}},
			{"balanced", default = false}
		}
		local data, kFold, Cgrid, gammaGrid, pGrid, nuGrid, coeffGrid, degreeGrid, balanced
			= cv.argcheck(t, argRules)

		return C.SVM_trainAuto(self.ptr,
			data.ptr, kFold, Cgrid.ptr, gammaGrid.ptr, pGrid.ptr,
			nuGrid.ptr, coeffGrid.ptr, degreeGrid.ptr, balanced)
	end

	function SVM:getSupportVectors()
	    return cv.unwrap_tensors(C.SVM_getSupportVectors(self.ptr))
	end

	function SVM:getDecisionFunction(t)
		local argRules = {
	        {"i", required = true},
	        {"alpha", default = nil},
	        {"svidx", default = nil}
	    }
	    local i, alpha, svidx = cv.argcheck(t, argRules)

	    local result = C.SVM_getDecisionFunction(self.ptr, i, cv.wrap_tensor(alpha), cv.wrap_tensor(svidx))
	    return result.val, cv.unwrap_tensors(result.tensors)
	end

	function SVM.getDefaultGrid(t)
		local argRules = {
	        {"param_id", required = true}
	    }
	    local param_id = cv.argcheck(t, argRules)

	    return ml.ParamGrid(C.SVM_getDefaultGrid(param_id))
	end
end

-- EM

do
    local EM = torch.class('ml.EM', 'ml.StatModel', ml)

    function EM:load(t)
        local argRules = {
            {"filename", required = true},
            {"objname", default = ''}
        }
        local filename, objname = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.EM_load(filename, objname), Classes.Algorithm_dtor)
    end

    function EM:__init()
        self.ptr = ffi.gc(C.EM_ctor(), Classes.Algorithm_dtor)
    end

    function EM:setClustersNumber(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.EM_setClustersNumber(self.ptr, val)
    end

    function EM:getClustersNumber()
        return C.EM_getClustersNumber(self.ptr)
    end

    function EM:setCovarianceMatrixType(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.EM_setCovarianceMatrixType(self.ptr, val)
    end

    function EM:getCovarianceMatrixType()
        return C.EM_getCovarianceMatrixType(self.ptr)
    end

    function EM:setTermCriteria(t)
        local argRules = {
            {"val", required = true, operator = cv.TermCriteria}
        }
        local val = cv.argcheck(t, argRules)

        C.EM_setTermCriteria(self.ptr, val)
    end

    function EM:getTermCriteria()
        return C.EM_getTermCriteria(self.ptr)
    end

    function EM:getWeights()
        return C.EM_getWeights(self.ptr)
    end

    function EM:getMeans()
        return C.EM_getMeans(self.ptr)
    end

    function EM:getCovs()
        return cv.unwrap_tensors(C.EM_getCovs(self.ptr), true)
    end

    function EM:predict2(t)
        local argRules = {
            {"sample", required = true},
            {"probs", default = nil}
        }
        local sample, probs = cv.argcheck(t, argRules)

        local result = C.EM_predict2(self.ptr, cv.wrap_tensor(sample), cv.wrap_tensor(probs))
        return {result.v0, result.v1}
    end

    function EM:trainEM(t)
        local argRules = {
            {"samples", required = true},
            {"logLikelihoods", default = nil},
            {"labels", default = nil},
            {"probs", default = nil}
        }
        local samples, logLikelihoods, labels, probs = cv.argcheck(t, argRules)

        return C.EM_trainEM(self.ptr, cv.wrap_tensor(samples), cv.wrap_tensor(logLikelihoods),
            cv.wrap_tensor(labels), cv.wrap_tensor(probs))
    end

    function EM:trainE(t)
        local argRules = {
            {"samples", required = true},
            {"means0", required = true},
            {"covs0", default = nil},
            {"weights0", default = nil},
            {"logLikelihoods", default = nil},
            {"labels", default = nil},
            {"probs", default = nil}
        }
        local samples, means0, covs0, weights0, logLikelihoods, labels, probs
            = cv.argcheck(t, argRules)

        return C.EM_trainE(
            self.ptr, cv.wrap_tensor(samples), cv.wrap_tensor(means0),
            cv.wrap_tensor(covs0), cv.wrap_tensor(weights0),
            cv.wrap_tensor(logLikelihoods), cv.wrap_tensor(labels), cv.wrap_tensor(probs))
    end

    function EM:trainM(t)
        local argRules = {
            {"samples", required = true},
            {"probs0", required = true},
            {"logLikelihoods", default = nil},
            {"labels", default = nil},
            {"probs", default = nil}
        }
        local samples, probs0, logLikelihoods, labels, probs = cv.argcheck(t, argRules)

        return C.EM_trainM(
            self.ptr, cv.wrap_tensor(samples), cv.wrap_tensor(probs0),
            cv.wrap_tensor(logLikelihoods), cv.wrap_tensor(labels), cv.wrap_tensor(probs))
    end
end

-- DTrees

do
    local DTrees = torch.class('ml.DTrees', 'ml.StatModel', ml)

    function DTrees:load(t)
        local argRules = {
            {"filename", required = true},
            {"objname", default = ''}
        }
        local filename, objname = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.DTrees_load(filename, objname), Classes.Algorithm_dtor)
    end

    function DTrees:__init()
        self.ptr = ffi.gc(C.DTrees_ctor(), Classes.Algorithm_dtor)
    end

    function DTrees:setMaxCategories(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DTrees_setMaxCategories(self.ptr, val)
    end

    function DTrees:getMaxCategories()
        return C.DTrees_getMaxCategories(self.ptr)
    end

    function DTrees:setMaxDepth(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DTrees_setMaxDepth(self.ptr, val)
    end

    function DTrees:getMaxDepth()
        return C.DTrees_getMaxDepth(self.ptr)
    end

    function DTrees:setMinSampleCount(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DTrees_setMinSampleCount(self.ptr, val)
    end

    function DTrees:getMinSampleCount()
        return C.DTrees_getMinSampleCount(self.ptr)
    end

    function DTrees:setCVFolds(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DTrees_setCVFolds(self.ptr, val)
    end

    function DTrees:getCVFolds()
        return C.DTrees_getCVFolds(self.ptr)
    end

    function DTrees:setUseSurrogates(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DTrees_setUseSurrogates(self.ptr, val)
    end

    function DTrees:getUseSurrogates()
        return C.DTrees_getUseSurrogates(self.ptr)
    end

    function DTrees:setUse1SERule(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DTrees_setUse1SERule(self.ptr, val)
    end

    function DTrees:getUse1SERule()
        return C.DTrees_getUse1SERule(self.ptr)
    end

    function DTrees:setTruncatePrunedTree(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DTrees_setTruncatePrunedTree(self.ptr, val)
    end

    function DTrees:getTruncatePrunedTree()
        return C.DTrees_getTruncatePrunedTree(self.ptr)
    end

    function DTrees:setRegressionAccuracy(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DTrees_setRegressionAccuracy(self.ptr, val)
    end

    function DTrees:getRegressionAccuracy()
        return C.DTrees_getRegressionAccuracy(self.ptr)
    end

    function DTrees:setPriors(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DTrees_setPriors(self.ptr, cv.wrap_tensor(val))
    end

    function DTrees:getPriors()
        return cv.unwrap_tensors(C.DTrees_getPriors(self.ptr))
    end

    function DTrees:getRoots()
        return cv.unwrap_tensors(C.DTrees_getRoots(self.ptr))
    end

    function DTrees:getNodes()
        local result = C.DTrees_getNodes(self.ptr)
        return result.ptr, result.size
    end

    function DTrees:getSplits()
        local result = C.DTrees_getSplits(self.ptr)
        return result.ptr, result.size
    end

    function DTrees:getSubsets()
        return cv.unwrap_tensors(C.DTrees_getSubsets(self.ptr))
    end
end

-- RTrees

do
    local RTrees = torch.class('ml.RTrees', 'ml.DTrees', ml)

    function RTrees:load(t)
        local argRules = {
            {"filename", required = true},
            {"objname", default = ''}
        }
        local filename, objname = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.RTrees_load(filename, objname), Classes.Algorithm_dtor)
    end

    function RTrees:load(t)
        local argRules = {
            {"filename", required = true},
            {"objname", default = ''}
        }
        local filename, objname = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.RTrees_load(filename, objname), Classes.Algorithm_dtor)
    end

    function RTrees:__init()
        self.ptr = ffi.gc(C.RTrees_ctor(), Classes.Algorithm_dtor)
    end

    function RTrees:setCalculateVarImportance(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.RTrees_setCalculateVarImportance(self.ptr, val)
    end

    function RTrees:getCalculateVarImportance()
        return C.RTrees_getCalculateVarImportance(self.ptr)
    end

    function RTrees:setActiveVarCount(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.RTrees_setActiveVarCount(self.ptr, val)
    end

    function RTrees:getActiveVarCount()
        return C.RTrees_getActiveVarCount(self.ptr)
    end

    function RTrees:setTermCriteria(t)
        local argRules = {
            {"val", required = true, operator = cv.TermCriteria}
        }
        local val = cv.argcheck(t, argRules)

        C.RTrees_setTermCriteria(self.ptr, val)
    end

    function RTrees:getTermCriteria()
        return C.RTrees_getTermCriteria(self.ptr)
    end

    function RTrees:getVarImportance()
        return cv.unwrap_tensors(C.RTrees_getVarImportance(self.ptr))
    end
end

-- Boost

do
    local Boost = torch.class('ml.Boost', 'ml.DTrees', ml)

    function Boost:load(t)
        local argRules = {
            {"filename", required = true},
            {"objname", default = ''}
        }
        local filename, objname = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.Boost_load(filename, objname), Classes.Algorithm_dtor)
    end

    function Boost:__init()
        self.ptr = ffi.gc(C.Boost_ctor(), Classes.Algorithm_dtor)
    end

    function Boost:setBoostType(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.Boost_setBoostType(self.ptr, val)
    end

    function Boost:getBoostType()
        return C.Boost_getBoostType(self.ptr)
    end

    function Boost:setWeakCount(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.Boost_setWeakCount(self.ptr, val)
    end

    function Boost:getWeakCount()
        return C.Boost_getWeakCount(self.ptr)
    end

    function Boost:setWeightTrimRate(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.Boost_setWeightTrimRate(self.ptr, val)
    end

    function Boost:getWeightTrimRate()
        return C.Boost_getWeightTrimRate(self.ptr)
    end
end

-- ANN_MLP

do
    local ANN_MLP = torch.class('ml.ANN_MLP', 'ml.StatModel', ml)

    function ANN_MLP:load(t)
        local argRules = {
            {"filename", required = true},
            {"objname", default = ''}
        }
        local filename, objname = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.ANN_MLP_load(filename, objname), Classes.Algorithm_dtor)
    end

    function ANN_MLP:__init()
        self.ptr = ffi.gc(C.ANN_MLP_ctor(), Classes.Algorithm_dtor)
    end

    function ANN_MLP:setTrainMethod(t)
        local argRules = {
            {"method", required = true},
            {"param1", default = 0},
            {"param2", default = 0},
        }
        local method, param1, param2 = cv.argcheck(t, argRules)

        C.ANN_MLP_setTrainMethod(self.ptr, method, param1, param2)
    end

    function ANN_MLP:getTrainMethod()
        return C.ANN_MLP_getTrainMethod(self.ptr)
    end

    function ANN_MLP:setActivationFunction(t)
        local argRules = {
            {"type", required = true},
            {"param1", default = 0},
            {"param2", default = 0},
        }
        local type, param1, param2 = cv.argcheck(t, argRules)

        C.ANN_MLP_setActivationFunction(self.ptr, type, param1, param2)
    end

    function ANN_MLP:getActivationFunction()
        return C.ANN_MLP_getActivationFunction(self.ptr)
    end

    function ANN_MLP:setLayerSizes(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.ANN_MLP_setLayerSizes(self.ptr, cv.wrap_tensor(val))
    end

    function ANN_MLP:getLayerSizes()
        return cv.unwrap_tensors(C.ANN_MLP_getLayerSizes(self.ptr))
    end

    function ANN_MLP:setTermCriteria(t)
        local argRules = {
            {"val", required = true, operator = cv.TermCriteria}
        }
        local val = cv.argcheck(t, argRules)

        C.ANN_MLP_setTermCriteria(self.ptr, val)
    end

    function ANN_MLP:getTermCriteria()
        return C.ANN_MLP_getTermCriteria(self.ptr)
    end

    function ANN_MLP:setBackpropWeightScale(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.ANN_MLP_setBackpropWeightScale(self.ptr, val)
    end

    function ANN_MLP:getBackpropWeightScale()
        return C.ANN_MLP_getBackpropWeightScale(self.ptr)
    end

    function ANN_MLP:setBackpropMomentumScale(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.ANN_MLP_setBackpropMomentumScale(self.ptr, val)
    end

    function ANN_MLP:getBackpropMomentumScale()
        return C.ANN_MLP_getBackpropMomentumScale(self.ptr)
    end

    function ANN_MLP:setRpropDW0(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.ANN_MLP_setRpropDW0(self.ptr, val)
    end

    function ANN_MLP:getRpropDW0()
        return C.ANN_MLP_getRpropDW0(self.ptr)
    end

    function ANN_MLP:setRpropDWPlus(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.ANN_MLP_setRpropDWPlus(self.ptr, val)
    end

    function ANN_MLP:getRpropDWPlus()
        return C.ANN_MLP_getRpropDWPlus(self.ptr)
    end

    function ANN_MLP:setRpropDWMinus(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.ANN_MLP_setRpropDWMinus(self.ptr, val)
    end

    function ANN_MLP:getRpropDWMinus()
        return C.ANN_MLP_getRpropDWMinus(self.ptr)
    end

    function ANN_MLP:setRpropDWMin(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.ANN_MLP_setRpropDWMin(self.ptr, val)
    end

    function ANN_MLP:getRpropDWMin()
        return C.ANN_MLP_getRpropDWMin(self.ptr)
    end

    function ANN_MLP:setRpropDWMax(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.ANN_MLP_setRpropDWMax(self.ptr, val)
    end

    function ANN_MLP:getRpropDWMax()
        return C.ANN_MLP_getRpropDWMax(self.ptr)
    end

    function ANN_MLP:setWeights(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.ANN_MLP_setWeights(self.ptr, val)
    end

    function ANN_MLP:getWeights()
        return cv.unwrap_tensors(C.ANN_MLP_getWeights(self.ptr))
    end
end

-- LogisticRegression

do
    local LogisticRegression = torch.class('ml.LogisticRegression', 'ml.StatModel', ml)

    function LogisticRegression:load(t)
        local argRules = {
            {"filename", required = true},
            {"objname", default = ''}
        }
        local filename, objname = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.LogisticRegression_load(filename, objname), Classes.Algorithm_dtor)
    end

    function LogisticRegression:__init()
        self.ptr = ffi.gc(C.LogisticRegression_ctor(), Classes.Algorithm_dtor)
    end

    function LogisticRegression:setLearningRate(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.LogisticRegression_setLearningRate(self.ptr, val)
    end

    function LogisticRegression:getLearningRate()
        return C.LogisticRegression_getLearningRate(self.ptr)
    end

    function LogisticRegression:setIterations(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.LogisticRegression_setIterations(self.ptr, val)
    end

    function LogisticRegression:getIterations()
        return C.LogisticRegression_getIterations(self.ptr)
    end

    function LogisticRegression:setRegularization(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.LogisticRegression_setRegularization(self.ptr, val)
    end

    function LogisticRegression:getRegularization()
        return C.LogisticRegression_getRegularization(self.ptr)
    end

    function LogisticRegression:setTrainMethod(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.LogisticRegression_setTrainMethod(self.ptr, val)
    end

    function LogisticRegression:getTrainMethod()
        return C.LogisticRegression_getTrainMethod(self.ptr)
    end

    function LogisticRegression:setMiniBatchSize(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.LogisticRegression_setMiniBatchSize(self.ptr, val)
    end

    function LogisticRegression:getMiniBatchSize()
        return C.LogisticRegression_getMiniBatchSize(self.ptr)
    end

    function LogisticRegression:setTermCriteria(t)
        local argRules = {
            {"val", required = true, operator = cv.TermCriteria}
        }
        local val = cv.argcheck(t, argRules)

        C.LogisticRegression_setTermCriteria(self.ptr, val)
    end

    function LogisticRegression:getTermCriteria()
        return C.LogisticRegression_getTermCriteria(self.ptr)
    end

    function LogisticRegression:get_learnt_thetas()
        return cv.unwrap_tensors(C.LogisticRegression_get_learnt_thetas(self.ptr))
    end
end

return ml
