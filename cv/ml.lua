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

]]

local C = ffi.load(cv.libPath('ml'))

function cv.ml.TrainData_getSubVector(t)
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

]]

-- ParamGrid

do
    local ParamGrid = torch.class('cv.ml.ParamGrid')

    function ParamGrid:__init(t)
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
    local TrainData = torch.class('cv.ml.TrainData')

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

    function cv.ml.TrainData:getLayout()
        return C.TrainData_getLayout(self.ptr)
    end

    function cv.ml.TrainData:getNTrainSamples()
        return C.TrainData_getNTrainSamples(self.ptr)
    end

    function cv.ml.TrainData:getNTestSamples()
        return C.TrainData_getNTestSamples(self.ptr)
    end

    function cv.ml.TrainData:getNSamples()
        return C.TrainData_getNSamples(self.ptr)
    end

    function cv.ml.TrainData:getNVars()
        return C.TrainData_getNVars(self.ptr)
    end

    function cv.ml.TrainData:getNAllVars()
        return C.TrainData_getNAllVars(self.ptr)
    end

    function cv.ml.TrainData:getSamples()
        return cv.unwrap_tensors(C.TrainData_getSamples(self.ptr))
    end

    function cv.ml.TrainData:getMissing()
        return cv.unwrap_tensors(C.TrainData_getMissing(self.ptr))
    end

    function cv.ml.TrainData:getTrainResponses()
        return cv.unwrap_tensors(C.TrainData_getTrainResponses(self.ptr))
    end

    function cv.ml.TrainData:getTrainNormCatResponses()
        return cv.unwrap_tensors(C.TrainData_getTrainNormCatResponses(self.ptr))
    end

    function cv.ml.TrainData:getTestResponses()
        return cv.unwrap_tensors(C.TrainData_getTestResponses(self.ptr))
    end

    function cv.ml.TrainData:getTestNormCatResponses()
        return cv.unwrap_tensors(C.TrainData_getTestNormCatResponses(self.ptr))
    end

    function cv.ml.TrainData:getResponses()
        return cv.unwrap_tensors(C.TrainData_getResponses(self.ptr))
    end

    function cv.ml.TrainData:getNormCatResponses()
        return cv.unwrap_tensors(C.TrainData_getNormCatResponses(self.ptr))
    end

    function cv.ml.TrainData:getSampleWeights()
        return cv.unwrap_tensors(C.TrainData_getSampleWeights(self.ptr))
    end

    function cv.ml.TrainData:getTrainSampleWeights()
        return cv.unwrap_tensors(C.TrainData_getTrainSampleWeights(self.ptr))
    end

    function cv.ml.TrainData:getTestSampleWeights()
        return cv.unwrap_tensors(C.TrainData_getTestSampleWeights(self.ptr))
    end

    function cv.ml.TrainData:getVarIdx()
        return cv.unwrap_tensors(C.TrainData_getVarIdx(self.ptr))
    end

    function cv.ml.TrainData:getVarType()
        return cv.unwrap_tensors(C.TrainData_getVarType(self.ptr))
    end

    function cv.ml.TrainData:getResponseType()
        return C.TrainData_getResponseType(self.ptr)
    end

    function cv.ml.TrainData:getTrainSampleIdx()
        return cv.unwrap_tensors(C.TrainData_getTrainSampleIdx(self.ptr))
    end

    function cv.ml.TrainData:getTestSampleIdx()
        return cv.unwrap_tensors(C.TrainData_getTestSampleIdx(self.ptr))
    end

    function cv.ml.TrainData:getDefaultSubstValues()
        return cv.unwrap_tensors(C.TrainData_getDefaultSubstValues(self.ptr))
    end

    function cv.ml.TrainData:getClassLabels()
        return cv.unwrap_tensors(C.TrainData_getClassLabels(self.ptr))
    end

    function cv.ml.TrainData:getCatOfs()
        return cv.unwrap_tensors(C.TrainData_getCatOfs(self.ptr))
    end

    function cv.ml.TrainData:getCatMap()
        return cv.unwrap_tensors(C.TrainData_getCatMap(self.ptr))
    end

    function cv.ml.TrainData:shuffleTrainTest()
        C.TrainData_shuffleTrainTest(self.ptr)
    end

    function cv.ml.TrainData:getCatCount(t)
        local argRules = {
            {"vi", required = true}
        }
        local vi = cv.argcheck(t, argRules)

        return C.TrainData_getCatCount(self.ptr, vi)
    end

    function cv.ml.TrainData:getSample(t)
        local argRules = {
            {"varIdx", required = true},
            {"sidx", required = true}
        }
        local varIdx, sidx = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.TrainData_getSample(
            self.ptr, cv.wrap_tensor(varIdx), sidx))
    end

    function cv.ml.TrainData:getTrainSamples(t)
        local argRules = {
            {"layout", default = cv.ml.ROW_SAMPLE},
            {"compressSamples", default = true},
            {"compressVars", default = true}
        }
        local layout, compressSamples, compressVars = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.TrainData_getTrainSamples(
            self.ptr, layout, compressSamples, compressVars))
    end

    function cv.ml.TrainData:getValues(t)
        local argRules = {
            {"vi", required = true},
            {"sidx", required = true}
        }
        local vi, sidx = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.TrainData_getValues(
            self.ptr, vi, cv.wrap_tensor(sidx)))
    end

    function cv.ml.TrainData:getNormCatValues(t)
        local argRules = {
            {"vi", required = true},
            {"sidx", required = true}
        }
        local vi, sidx = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.TrainData_getNormCatValues(
            self.ptr, vi, cv.wrap_tensor(sidx)))
    end

    function cv.ml.TrainData:setTrainTestSplit(t)
        local argRules {
            {"count", required = true},
            {"shuffle", default = true}
        }
        local count, shuffle = cv.argcheck(t, argRules)

        C.TrainData_setTrainTestSplit(count, shuffle)
    end

    function cv.ml.TrainData:setTrainTestSplitRatio(t)
        local argRules {
            {"ratio", required = true},
            {"shuffle", default = true}
        }
        local ratio, shuffle = cv.argcheck(t, argRules)

        C.TrainData_setTrainTestSplitRatio(ratio, shuffle)
    end
end