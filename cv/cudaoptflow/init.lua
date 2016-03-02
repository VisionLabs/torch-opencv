local cv = require 'cv._env'
require 'cutorch'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or require 'cv._env_cuda'

local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper DenseOpticalFlow_calcCuda(struct cutorchInfo info,
    struct PtrWrapper ptr, struct TensorWrapper I0, struct TensorWrapper I1,
    struct TensorWrapper flow);

struct PtrWrapper BroxOpticalFlow_ctorCuda(
        double alpha, double gamma, double scale_factor, int inner_iterations,
        int outer_iterations, int solver_iterations);

void BroxOpticalFlow_setFlowSmoothnessCuda(struct PtrWrapper ptr, double val);

double BroxOpticalFlow_getFlowSmoothnessCuda(struct PtrWrapper ptr);

void BroxOpticalFlow_setGradientConstancyImportanceCuda(struct PtrWrapper ptr, double val);

double BroxOpticalFlow_getGradientConstancyImportanceCuda(struct PtrWrapper ptr);

void BroxOpticalFlow_setPyramidScaleFactorCuda(struct PtrWrapper ptr, double val);

double BroxOpticalFlow_getPyramidScaleFactorCuda(struct PtrWrapper ptr);

void BroxOpticalFlow_setInnerIterationsCuda(struct PtrWrapper ptr, int val);

int BroxOpticalFlow_getInnerIterationsCuda(struct PtrWrapper ptr);

void BroxOpticalFlow_setOuterIterationsCuda(struct PtrWrapper ptr, int val);

int BroxOpticalFlow_getOuterIterationsCuda(struct PtrWrapper ptr);

void BroxOpticalFlow_setSolverIterationsCuda(struct PtrWrapper ptr, int val);

int BroxOpticalFlow_getSolverIterationsCuda(struct PtrWrapper ptr);

struct PtrWrapper SparsePyrLKOpticalFlow_ctorCuda(
        struct SizeWrapper winSize, int maxLevel, int iters, bool useInitialFlow);

void SparsePyrLKOpticalFlow_setWinSizeCuda(struct PtrWrapper ptr, struct SizeWrapper val);

struct SizeWrapper SparsePyrLKOpticalFlow_getWinSizeCuda(struct PtrWrapper ptr);

void SparsePyrLKOpticalFlow_setMaxLevelCuda(struct PtrWrapper ptr, int val);

int SparsePyrLKOpticalFlow_getMaxLevelCuda(struct PtrWrapper ptr);

void SparsePyrLKOpticalFlow_setNumItersCuda(struct PtrWrapper ptr, int val);

int SparsePyrLKOpticalFlow_getNumItersCuda(struct PtrWrapper ptr);

void SparsePyrLKOpticalFlow_setUseInitialFlowCuda(struct PtrWrapper ptr, bool val);

bool SparsePyrLKOpticalFlow_getUseInitialFlowCuda(struct PtrWrapper ptr);

struct PtrWrapper DensePyrLKOpticalFlow_ctorCuda(
        struct SizeWrapper winSize, int maxLevel, int iters, bool useInitialFlow);

void DensePyrLKOpticalFlow_setWinSizeCuda(struct PtrWrapper ptr, struct SizeWrapper val);

struct SizeWrapper DensePyrLKOpticalFlow_getWinSizeCuda(struct PtrWrapper ptr);

void DensePyrLKOpticalFlow_setMaxLevelCuda(struct PtrWrapper ptr, int val);

int DensePyrLKOpticalFlow_getMaxLevelCuda(struct PtrWrapper ptr);

void DensePyrLKOpticalFlow_setNumItersCuda(struct PtrWrapper ptr, int val);

int DensePyrLKOpticalFlow_getNumItersCuda(struct PtrWrapper ptr);

void DensePyrLKOpticalFlow_setUseInitialFlowCuda(struct PtrWrapper ptr, bool val);

bool DensePyrLKOpticalFlow_getUseInitialFlowCuda(struct PtrWrapper ptr);

struct PtrWrapper FarnebackOpticalFlow_ctorCuda(
        int NumLevels, double PyrScale, bool FastPyramids, int WinSize,
        int NumIters, int PolyN, double PolySigma, int Flags);

void FarnebackOpticalFlow_setNumLevelsCuda(struct PtrWrapper ptr, int val);

int FarnebackOpticalFlow_getNumLevelsCuda(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setPyrScaleCuda(struct PtrWrapper ptr, double val);

double FarnebackOpticalFlow_getPyrScaleCuda(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setFastPyramidsCuda(struct PtrWrapper ptr, bool val);

bool FarnebackOpticalFlow_getFastPyramidsCuda(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setWinSizeCuda(struct PtrWrapper ptr, int val);

int FarnebackOpticalFlow_getWinSizeCuda(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setNumItersCuda(struct PtrWrapper ptr, int val);

int FarnebackOpticalFlow_getNumItersCuda(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setPolyNCuda(struct PtrWrapper ptr, int val);

int FarnebackOpticalFlow_getPolyNCuda(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setPolySigmaCuda(struct PtrWrapper ptr, double val);

double FarnebackOpticalFlow_getPolySigmaCuda(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setFlagsCuda(struct PtrWrapper ptr, int val);

int FarnebackOpticalFlow_getFlagsCuda(struct PtrWrapper ptr);

struct PtrWrapper OpticalFlowDual_TVL1_ctorCuda(
        double tau, double lambda, double theta, int nscales, int warps, double epsilon, 
        int iterations, double scaleStep, double gamma, bool useInitialFlow);

void OpticalFlowDual_TVL1_setTauCuda(struct PtrWrapper ptr, double val);

double OpticalFlowDual_TVL1_getTauCuda(struct PtrWrapper ptr);

void OpticalFlowDual_TVL1_setLambdaCuda(struct PtrWrapper ptr, double val);

double OpticalFlowDual_TVL1_getLambdaCuda(struct PtrWrapper ptr);

void OpticalFlowDual_TVL1_setGammaCuda(struct PtrWrapper ptr, double val);

double OpticalFlowDual_TVL1_getGammaCuda(struct PtrWrapper ptr);

void OpticalFlowDual_TVL1_setThetaCuda(struct PtrWrapper ptr, double val);

double OpticalFlowDual_TVL1_getThetaCuda(struct PtrWrapper ptr);

void OpticalFlowDual_TVL1_setNumScalesCuda(struct PtrWrapper ptr, int val);

int OpticalFlowDual_TVL1_getNumScalesCuda(struct PtrWrapper ptr);

void OpticalFlowDual_TVL1_setNumWarpsCuda(struct PtrWrapper ptr, int val);

int OpticalFlowDual_TVL1_getNumWarpsCuda(struct PtrWrapper ptr);

void OpticalFlowDual_TVL1_setEpsilonCuda(struct PtrWrapper ptr, double val);

double OpticalFlowDual_TVL1_getEpsilonCuda(struct PtrWrapper ptr);

void OpticalFlowDual_TVL1_setNumIterationsCuda(struct PtrWrapper ptr, int val);

int OpticalFlowDual_TVL1_getNumIterationsCuda(struct PtrWrapper ptr);

void OpticalFlowDual_TVL1_setScaleStepCuda(struct PtrWrapper ptr, double val);

double OpticalFlowDual_TVL1_getScaleStepCuda(struct PtrWrapper ptr);

void OpticalFlowDual_TVL1_setUseInitialFlowCuda(struct PtrWrapper ptr, bool val);

bool OpticalFlowDual_TVL1_getUseInitialFlowCuda(struct PtrWrapper ptr);
]]

local C = ffi.load(cv.libPath('cudaoptflow'))

require 'cv.Classes'
local Classes = ffi.load(cv.libPath('Classes'))

do
    local DenseOpticalFlow = torch.class('cuda.DenseOpticalFlow', 'cv.Algorithm', cv.cuda)

    function DenseOpticalFlow:calc(t)
        local argRules = {
            {"I0", required = true, operator = cv.wrap_tensor},
            {"I1", required = true, operator = cv.wrap_tensor},
            {"flow", default = nil, operator = cv.wrap_tensor}
        }

        return cv.unwrap_tensors(C.DenseOpticalFlow_calcCuda(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end
end

do
    local SparseOpticalFlow = torch.class('cuda.SparseOpticalFlow', 'cv.Algorithm', cv.cuda)

    function SparseOpticalFlow:calc(t)
        local argRules = {
            {"prevImg", required = true, operator = cv.wrap_tensor},
            {"nextImg", required = true, operator = cv.wrap_tensor},
            {"prevPts", default = nil, operator = cv.wrap_tensor},
            {"status", default = nil, operator = cv.wrap_tensor},
            {"err", default = nil, operator = cv.wrap_tensor}
        }

        return cv.unwrap_tensors(C.SparseOpticalFlow_calcCuda(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end
end

do
    local BroxOpticalFlow = torch.class('cuda.BroxOpticalFlow', 'cuda.DenseOpticalFlow', cv.cuda)

    function BroxOpticalFlow:__init(t)
        local argRules = {
            {"alpha", default = 0.197},
            {"gamma", default = 50.0},
            {"scale_factor", default = 0.8},
            {"inner_iterations", default = 5},
            {"outer_iterations", default = 150},
            {"solver_iterations", default = 10}
        }
        
        self.ptr = ffi.gc(C.BroxOpticalFlow_ctorCuda(cv.argcheck(t, argRules)), Classes.Algorithm_dtor)
    end

    function BroxOpticalFlow:setFlowSmoothness(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.BroxOpticalFlow_setFlowSmoothnessCuda(self.ptr, val)
    end

    function BroxOpticalFlow:getFlowSmoothness()
        return C.BroxOpticalFlow_getFlowSmoothnessCuda(self.ptr)
    end

    function BroxOpticalFlow:setGradientConstancyImportance(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.BroxOpticalFlow_setGradientConstancyImportanceCuda(self.ptr, val)
    end

    function BroxOpticalFlow:getGradientConstancyImportance()
        return C.BroxOpticalFlow_getGradientConstancyImportanceCuda(self.ptr)
    end

    function BroxOpticalFlow:setPyramidScaleFactor(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.BroxOpticalFlow_setPyramidScaleFactorCuda(self.ptr, val)
    end

    function BroxOpticalFlow:getPyramidScaleFactor()
        return C.BroxOpticalFlow_getPyramidScaleFactorCuda(self.ptr)
    end

    function BroxOpticalFlow:setInnerIterations(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.BroxOpticalFlow_setInnerIterationsCuda(self.ptr, val)
    end

    function BroxOpticalFlow:getInnerIterations()
        return C.BroxOpticalFlow_getInnerIterationsCuda(self.ptr)
    end

    function BroxOpticalFlow:setOuterIterations(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.BroxOpticalFlow_setOuterIterationsCuda(self.ptr, val)
    end

    function BroxOpticalFlow:getOuterIterations()
        return C.BroxOpticalFlow_getOuterIterationsCuda(self.ptr)
    end

    function BroxOpticalFlow:setSolverIterations(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.BroxOpticalFlow_setSolverIterationsCuda(self.ptr, val)
    end

    function BroxOpticalFlow:getSolverIterations()
        return C.BroxOpticalFlow_getSolverIterationsCuda(self.ptr)
    end
end

do
    local SparsePyrLKOpticalFlow = torch.class(
        'cuda.SparsePyrLKOpticalFlow', 'cuda.SparseOpticalFlow', cv.cuda)

    function SparsePyrLKOpticalFlow:__init(t)
        local argRules = {
            {"winSize", default = {21, 21}, operator = cv.Size},
            {"maxLevel", default = 3},
            {"iters", default = 30},
            {"useInitialFlow", default = false}
        }

        self.ptr = ffi.gc(
            C.SparsePyrLKOpticalFlow_ctorCuda(cv.argcheck(t, argRules)),
            Classes.Algorithm_dtor)
    end

    function SparsePyrLKOpticalFlow:setWinSize(t)
        local argRules = {
            {"val", required = true, operator = cv.Size}
        }
        local val = cv.argcheck(t, argRules)
        
        C.SparsePyrLKOpticalFlow_setWinSizeCuda(self.ptr, val)
    end

    function SparsePyrLKOpticalFlow:getWinSize()
        return C.SparsePyrLKOpticalFlow_getWinSizeCuda(self.ptr)
    end

    function SparsePyrLKOpticalFlow:setMaxLevel(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.SparsePyrLKOpticalFlow_setMaxLevelCuda(self.ptr, val)
    end

    function SparsePyrLKOpticalFlow:getMaxLevel()
        return C.SparsePyrLKOpticalFlow_getMaxLevelCuda(self.ptr)
    end

    function SparsePyrLKOpticalFlow:setNumIters(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.SparsePyrLKOpticalFlow_setNumItersCuda(self.ptr, val)
    end

    function SparsePyrLKOpticalFlow:getNumIters()
        return C.SparsePyrLKOpticalFlow_getNumItersCuda(self.ptr)
    end

    function SparsePyrLKOpticalFlow:setUseInitialFlow(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.SparsePyrLKOpticalFlow_setUseInitialFlowCuda(self.ptr, val)
    end

    function SparsePyrLKOpticalFlow:getUseInitialFlow()
        return C.SparsePyrLKOpticalFlow_getUseInitialFlowCuda(self.ptr)
    end
end

do
    local DensePyrLKOpticalFlow = torch.class(
        'cuda.DensePyrLKOpticalFlow', 'cuda.DenseOpticalFlow', cv.cuda)

    function DensePyrLKOpticalFlow:__init(t)
        local argRules = {
            {"winSize", default = {13, 13}, operator = cv.Size},
            {"maxLevel", default = 3},
            {"iters", default = 30},
            {"useInitialFlow", default = false}
        }

        self.ptr = ffi.gc(
            C.DensePyrLKOpticalFlow_ctorCuda(cv.argcheck(t, argRules)),
            Classes.Algorithm_dtor)
    end

    function DensePyrLKOpticalFlow:setWinSize(t)
        local argRules = {
            {"val", required = true, operator = cv.Size}
        }
        local val = cv.argcheck(t, argRules)
        
        C.DensePyrLKOpticalFlow_setWinSizeCuda(self.ptr, val)
    end

    function DensePyrLKOpticalFlow:getWinSize()
        return C.DensePyrLKOpticalFlow_getWinSizeCuda(self.ptr)
    end

    function DensePyrLKOpticalFlow:setMaxLevel(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.DensePyrLKOpticalFlow_setMaxLevelCuda(self.ptr, val)
    end

    function DensePyrLKOpticalFlow:getMaxLevel()
        return C.DensePyrLKOpticalFlow_getMaxLevelCuda(self.ptr)
    end

    function DensePyrLKOpticalFlow:setNumIters(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.DensePyrLKOpticalFlow_setNumItersCuda(self.ptr, val)
    end

    function DensePyrLKOpticalFlow:getNumIters()
        return C.DensePyrLKOpticalFlow_getNumItersCuda(self.ptr)
    end

    function DensePyrLKOpticalFlow:setUseInitialFlow(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.DensePyrLKOpticalFlow_setUseInitialFlowCuda(self.ptr, val)
    end

    function DensePyrLKOpticalFlow:getUseInitialFlow()
        return C.DensePyrLKOpticalFlow_getUseInitialFlowCuda(self.ptr)
    end
end

do
    local FarnebackOpticalFlow = torch.class('cuda.FarnebackOpticalFlow', 'cuda.DenseOpticalFlow', cv.cuda)

    function FarnebackOpticalFlow:__init(t)
        local argRules = {
            {"numLevels", default = 5},
            {"pyrScale", default = 0.5},
            {"fastPyramids", default = false},
            {"winSize", default = 13},
            {"numIters", default = 10},
            {"polyN", default = 5},
            {"polySigma", default = 1.1},
            {"flags", default = 0},
        }

        self.ptr = ffi.gc(C.FarnebackOpticalFlow_ctorCuda(cv.argcheck(t, argRules)), Classes.Algorithm_dtor)
    end

    function FarnebackOpticalFlow:setNumLevels(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FarnebackOpticalFlow_setNumLevelsCuda(self.ptr, val)
    end

    function FarnebackOpticalFlow:getNumLevels()
        return C.FarnebackOpticalFlow_getNumLevelsCuda(self.ptr)
    end

    function FarnebackOpticalFlow:setPyrScale(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FarnebackOpticalFlow_setPyrScaleCuda(self.ptr, val)
    end

    function FarnebackOpticalFlow:getPyrScale()
        return C.FarnebackOpticalFlow_getPyrScaleCuda(self.ptr)
    end

    function FarnebackOpticalFlow:setFastPyramids(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FarnebackOpticalFlow_setFastPyramidsCuda(self.ptr, val)
    end

    function FarnebackOpticalFlow:getFastPyramids()
        return C.FarnebackOpticalFlow_getFastPyramidsCuda(self.ptr)
    end

    function FarnebackOpticalFlow:setWinSize(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FarnebackOpticalFlow_setWinSizeCuda(self.ptr, val)
    end

    function FarnebackOpticalFlow:getWinSize()
        return C.FarnebackOpticalFlow_getWinSizeCuda(self.ptr)
    end

    function FarnebackOpticalFlow:setNumIters(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FarnebackOpticalFlow_setNumItersCuda(self.ptr, val)
    end

    function FarnebackOpticalFlow:getNumIters()
        return C.FarnebackOpticalFlow_getNumItersCuda(self.ptr)
    end

    function FarnebackOpticalFlow:setPolyN(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FarnebackOpticalFlow_setPolyNCuda(self.ptr, val)
    end

    function FarnebackOpticalFlow:getPolyN()
        return C.FarnebackOpticalFlow_getPolyNCuda(self.ptr)
    end

    function FarnebackOpticalFlow:setPolySigma(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FarnebackOpticalFlow_setPolySigmaCuda(self.ptr, val)
    end

    function FarnebackOpticalFlow:getPolySigma()
        return C.FarnebackOpticalFlow_getPolySigmaCuda(self.ptr)
    end

    function FarnebackOpticalFlow:setFlags(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FarnebackOpticalFlow_setFlagsCuda(self.ptr, val)
    end

    function FarnebackOpticalFlow:getFlags()
        return C.FarnebackOpticalFlow_getFlagsCuda(self.ptr)
    end
end

do
    local OpticalFlowDual_TVL1 = torch.class('cuda.OpticalFlowDual_TVL1', 'cuda.DenseOpticalFlow', cv.cuda)

    function OpticalFlowDual_TVL1:__init(t)
        local argRules = {
            {"tau", default = 0.25},
            {"lambda", default = 0.15},
            {"theta", default = 0.3},
            {"nscales", default = 5},
            {"warps", default = 5},
            {"epsilon", default = 0.01},
            {"iterations", default = 300},
            {"scaleStep", default = 0.8},
            {"gamma", default = 0.0},
            {"useInitialFlow", default = false}
        }

        self.ptr = ffi.gc(C.OpticalFlowDual_TVL1_ctorCuda(cv.argcheck(t, argRules)), Classes.Algorithm_dtor)
    end

    function OpticalFlowDual_TVL1:setTau(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setTauCuda(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getTau()
        return C.OpticalFlowDual_TVL1_getTauCuda(self.ptr)
    end

    function OpticalFlowDual_TVL1:setLambda(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setLambdaCuda(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getLambda()
        return C.OpticalFlowDual_TVL1_getLambdaCuda(self.ptr)
    end

    function OpticalFlowDual_TVL1:setGamma(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setGammaCuda(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getGamma()
        return C.OpticalFlowDual_TVL1_getGammaCuda(self.ptr)
    end

    function OpticalFlowDual_TVL1:setTheta(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setThetaCuda(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getTheta()
        return C.OpticalFlowDual_TVL1_getThetaCuda(self.ptr)
    end

    function OpticalFlowDual_TVL1:setNumScales(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setNumScalesCuda(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getNumScales()
        return C.OpticalFlowDual_TVL1_getNumScalesCuda(self.ptr)
    end

    function OpticalFlowDual_TVL1:setNumWarps(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setNumWarpsCuda(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getNumWarps()
        return C.OpticalFlowDual_TVL1_getNumWarpsCuda(self.ptr)
    end

    function OpticalFlowDual_TVL1:setEpsilon(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setEpsilonCuda(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getEpsilon()
        return C.OpticalFlowDual_TVL1_getEpsilonCuda(self.ptr)
    end

    function OpticalFlowDual_TVL1:setNumIterations(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setNumIterationsCuda(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getNumIterations()
        return C.OpticalFlowDual_TVL1_getNumIterationsCuda(self.ptr)
    end

    function OpticalFlowDual_TVL1:setScaleStep(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setScaleStepCuda(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getScaleStep()
        return C.OpticalFlowDual_TVL1_getScaleStepCuda(self.ptr)
    end

    function OpticalFlowDual_TVL1:setUseInitialFlow(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setUseInitialFlowCuda(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getUseInitialFlow()
        return C.OpticalFlowDual_TVL1_getUseInitialFlowCuda(self.ptr)
    end
end

return cv.cuda
