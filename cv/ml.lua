require 'cv'

local ffi = require 'ffi'

cv.ml = {}

cv.ml.ANN_MLP_BACKPROP = 0
cv.ml.ANN_MLP_GAUSSIAN = 2
cv.ml.ANN_MLP_IDENTITY = 0
cv.ml.ANN_MLP_NO_INPUT_SCALE = 2
cv.ml.ANN_MLP_NO_OUTPUT_SCALE = 4
cv.ml.ANN_MLP_RPROP = 1
cv.ml.ANN_MLP_SIGMOID_SYM = 1
cv.ml.ANN_MLP_UPDATE_WEIGHTS = 1
cv.ml.BOOST_DISCRETE = 0
cv.ml.BOOST_GENTLE = 3
cv.ml.BOOST_LOGIT = 2
cv.ml.BOOST_REAL = 1
cv.ml.Boost_DISCRETE = 0
cv.ml.Boost_GENTLE = 3
cv.ml.Boost_LOGIT = 2
cv.ml.Boost_REAL = 1
cv.ml.COL_SAMPLE = 1
cv.ml.DTREES_PREDICT_AUTO = 0
cv.ml.DTREES_PREDICT_MASK = 768
cv.ml.DTREES_PREDICT_MAX_VOTE = 512
cv.ml.DTREES_PREDICT_SUM = 256
cv.ml.DTrees_PREDICT_AUTO = 0
cv.ml.DTrees_PREDICT_MASK = 768
cv.ml.DTrees_PREDICT_MAX_VOTE = 512
cv.ml.DTrees_PREDICT_SUM = 256
cv.ml.EM_COV_MAT_DEFAULT = 1
cv.ml.EM_COV_MAT_DIAGONAL = 1
cv.ml.EM_COV_MAT_GENERIC = 2
cv.ml.EM_COV_MAT_SPHERICAL = 0
cv.ml.EM_DEFAULT_MAX_ITERS = 100
cv.ml.EM_DEFAULT_NCLUSTERS = 5
cv.ml.EM_START_AUTO_STEP = 0
cv.ml.EM_START_E_STEP = 1
cv.ml.EM_START_M_STEP = 2
cv.ml.KNEAREST_BRUTE_FORCE = 1
cv.ml.KNEAREST_KDTREE = 2
cv.ml.KNearest_BRUTE_FORCE = 1
cv.ml.KNearest_KDTREE = 2
cv.ml.LOGISTIC_REGRESSION_BATCH = 0
cv.ml.LOGISTIC_REGRESSION_MINI_BATCH = 1
cv.ml.LOGISTIC_REGRESSION_REG_DISABLE = -1
cv.ml.LOGISTIC_REGRESSION_REG_L1 = 0
cv.ml.LOGISTIC_REGRESSION_REG_L2 = 1
cv.ml.LogisticRegression_BATCH = 0
cv.ml.LogisticRegression_MINI_BATCH = 1
cv.ml.LogisticRegression_REG_DISABLE = -1
cv.ml.LogisticRegression_REG_L1 = 0
cv.ml.LogisticRegression_REG_L2 = 1
cv.ml.ROW_SAMPLE = 0
cv.ml.STAT_MODEL_COMPRESSED_INPUT = 2
cv.ml.STAT_MODEL_PREPROCESSED_INPUT = 4
cv.ml.STAT_MODEL_RAW_OUTPUT = 1
cv.ml.STAT_MODEL_UPDATE_MODEL = 1
cv.ml.SVM_C = 0
cv.ml.SVM_CHI2 = 4
cv.ml.SVM_COEF = 4
cv.ml.SVM_CUSTOM = -1
cv.ml.SVM_C_SVC = 100
cv.ml.SVM_DEGREE = 5
cv.ml.SVM_EPS_SVR = 103
cv.ml.SVM_GAMMA = 1
cv.ml.SVM_INTER = 5
cv.ml.SVM_LINEAR = 0
cv.ml.SVM_NU = 3
cv.ml.SVM_NU_SVC = 101
cv.ml.SVM_NU_SVR = 104
cv.ml.SVM_ONE_CLASS = 102
cv.ml.SVM_P = 2
cv.ml.SVM_POLY = 1
cv.ml.SVM_RBF = 2
cv.ml.SVM_SIGMOID = 3
cv.ml.StatModel_COMPRESSED_INPUT = 2
cv.ml.StatModel_PREPROCESSED_INPUT = 4
cv.ml.StatModel_RAW_OUTPUT = 1
cv.ml.StatModel_UPDATE_MODEL = 1
cv.ml.TEST_ERROR = 0
cv.ml.TRAIN_ERROR = 1
cv.ml.VAR_CATEGORICAL = 1
cv.ml.VAR_NUMERICAL = 0
cv.ml.VAR_ORDERED = 0

ffi.cdef[[
struct TensorWrapper TrainData_getSubVector(struct TensorWrapper vec, struct TensorWrapper idx);
]]

local C = ffi.load(cv.libPath('ml'))

function TrainData_getSubVector(t)
    local argRules = {
        {"vec", required = true},
        {"idx", required = true}
    }
    local vec, idx = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.TrainData_getSubVector(cv.wrap_tensor(vec), cv.wrap_tensor(idx)))
end

--- ***************** Classes *****************
require 'cv.Classes'

local Classes = ffi.load(cv.libPath('Classes'))

ffi.cdef[[
struct PtrWrapper ParamGrid_ctor(double _minVal, double _maxVal, double _logStep);

struct PtrWrapper ParamGrid_ctor_default();

struct PtrWrapper TrainData_ctor(
        struct TensorWrapper samples, int layout, struct TensorWrapper responses,
        struct TensorWrapper varIdx, struct TensorWrapper sampleIdx,
        struct TensorWrapper sampleWeights, struct TensorWrapper varType);

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

//void SVM_setCustomKernel(struct PtrWrapper ptr, struct KernelPtr val);

bool SVM_trainAuto(
        struct PtrWrapper ptr, struct TrainDataPtr data, int kFold, struct ParamGridPtr Cgrid,
        struct ParamGridPtr gammaGrid, struct ParamGridPtr pGrid, struct ParamGridPtr nuGrid,
        struct ParamGridPtr coeffGrid, struct ParamGridPtr degreeGrid, bool balanced);

struct TensorWrapper SVM_getSupportVectors(struct PtrWrapper ptr);

struct TensorArrayPlusDouble SVM_getDecisionFunction(
        struct PtrWrapper ptr, int i, struct TensorWrapper alpha, struct TensorWrapper svidx);

struct ParamGridPtr SVM_getDefaultGrid(struct PtrWrapper ptr, int param_id);

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

]]

-- ParamGrid

do
    local ParamGrid = torch.class('cv.ml.ParamGrid')

    function ParamGrid:__init(t)
    	if type(t) ~= 'table' then
    		self.ptr = t
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
    local TrainData = torch.class('cv.TrainData')

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

        self.ptr = C.TrainData_ctor(
            cv.wrap_tensor(samples), layout, cv.wrap_tensor(responses), cv.wrap_tensor(varIdx), 
            cv.wrap_tensor(sampleIdx), cv.wrap_tensor(sampleWeights), cv.wrap_tensor(varType)
        )
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
    local StatModel = torch.class('cv.StatModel', 'cv.Algorithm')

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
	local NormalBayesClassifier = torch.class('cv.NormalBayesClassifier', 'cv.StatModel')

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
    local KNearest = torch.class('cv.KNearest', 'cv.StatModel')

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
	local SVM = torch.class('cv.SVM', 'cv.StatModel')

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
			{"Cgrid", default = SVM.getDefaultGrid(cv.ml.SVM_C)},
			{"gammaGrid", default = SVM.getDefaultGrid(cv.ml.SVM_GAMMA)},
			{"pGrid", default = SVM.getDefaultGrid(cv.ml.SVM_P)},
			{"nuGrid", default = SVM.getDefaultGrid(cv.ml.SVM_NU)},
			{"coeffGrid", default = SVM.getDefaultGrid(cv.ml.SVM_COEF)},
			{"degreeGrid", default = SVM.getDefaultGrid(cv.ml.SVM_DEGREE)},
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

	function SVM:getDefaultGrid(t)
		local argRules = {
	        {"param_id", required = true}
	    }
	    local param_id = cv.argcheck(t, argRules)

	    return cv.ParamGrid(C.SVM_getDefaultGrid(self.ptr, param_id))
	end
end

-- EM

do
    local EM = torch.class('cv.EM', 'cv.StatModel')

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
    local DTrees = torch.class('cv.DTrees', 'cv.StatModel')

end

-- RTrees

do
    local RTrees = torch.class('cv.RTrees', 'cv.DTrees')

end

-- Boost

do
    local Boost = torch.class('cv.Boost', 'cv.DTrees')

end

-- ANN_MLP

do
    local ANN_MLP = torch.class('cv.ANN_MLP', 'cv.StatModel')

end

-- LogisticRegression

do
    local LogisticRegression = torch.class('cv.LogisticRegression', 'cv.StatModel')

end

