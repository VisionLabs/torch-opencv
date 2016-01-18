local cv = require 'cv._env'
require 'cutorch'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or {}

local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper DenseOpticalFlow_calc(struct THCState *state,
    struct PtrWrapper ptr, struct TensorWrapper I0, struct TensorWrapper I1,
    struct TensorWrapper flow);

struct PtrWrapper BroxOpticalFlow_ctor(
        double alpha, double gamma, double scale_factor, int inner_iterations,
        int outer_iterations, int solver_iterations);

void BroxOpticalFlow_setFlowSmoothness(struct PtrWrapper ptr, double val);

double BroxOpticalFlow_getFlowSmoothness(struct PtrWrapper ptr);

void BroxOpticalFlow_setGradientConstancyImportance(struct PtrWrapper ptr, double val);

double BroxOpticalFlow_getGradientConstancyImportance(struct PtrWrapper ptr);

void BroxOpticalFlow_setPyramidScaleFactor(struct PtrWrapper ptr, double val);

double BroxOpticalFlow_getPyramidScaleFactor(struct PtrWrapper ptr);

void BroxOpticalFlow_setInnerIterations(struct PtrWrapper ptr, int val);

int BroxOpticalFlow_getInnerIterations(struct PtrWrapper ptr);

void BroxOpticalFlow_setOuterIterations(struct PtrWrapper ptr, int val);

int BroxOpticalFlow_getOuterIterations(struct PtrWrapper ptr);

void BroxOpticalFlow_setSolverIterations(struct PtrWrapper ptr, int val);

int BroxOpticalFlow_getSolverIterations(struct PtrWrapper ptr);

struct PtrWrapper SparsePyrLKOpticalFlow_ctor(
        struct SizeWrapper winSize, int maxLevel, int iters, bool useInitialFlow);

void SparsePyrLKOpticalFlow_setWinSize(struct PtrWrapper ptr, struct SizeWrapper val);

struct SizeWrapper SparsePyrLKOpticalFlow_getWinSize(struct PtrWrapper ptr);

void SparsePyrLKOpticalFlow_setMaxLevel(struct PtrWrapper ptr, int val);

int SparsePyrLKOpticalFlow_getMaxLevel(struct PtrWrapper ptr);

void SparsePyrLKOpticalFlow_setNumIters(struct PtrWrapper ptr, int val);

int SparsePyrLKOpticalFlow_getNumIters(struct PtrWrapper ptr);

void SparsePyrLKOpticalFlow_setUseInitialFlow(struct PtrWrapper ptr, bool val);

bool SparsePyrLKOpticalFlow_getUseInitialFlow(struct PtrWrapper ptr);

struct PtrWrapper DensePyrLKOpticalFlow_ctor(
        struct SizeWrapper winSize, int maxLevel, int iters, bool useInitialFlow);

void DensePyrLKOpticalFlow_setWinSize(struct PtrWrapper ptr, struct SizeWrapper val);

struct SizeWrapper DensePyrLKOpticalFlow_getWinSize(struct PtrWrapper ptr);

void DensePyrLKOpticalFlow_setMaxLevel(struct PtrWrapper ptr, int val);

int DensePyrLKOpticalFlow_getMaxLevel(struct PtrWrapper ptr);

void DensePyrLKOpticalFlow_setNumIters(struct PtrWrapper ptr, int val);

int DensePyrLKOpticalFlow_getNumIters(struct PtrWrapper ptr);

void DensePyrLKOpticalFlow_setUseInitialFlow(struct PtrWrapper ptr, bool val);

bool DensePyrLKOpticalFlow_getUseInitialFlow(struct PtrWrapper ptr);

struct PtrWrapper FarnebackOpticalFlow_ctor(
        int NumLevels, double PyrScale, bool FastPyramids, int WinSize,
        int NumIters, int PolyN, double PolySigma, int Flags);

void FarnebackOpticalFlow_setNumLevels(struct PtrWrapper ptr, int val);

int FarnebackOpticalFlow_getNumLevels(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setPyrScale(struct PtrWrapper ptr, double val);

double FarnebackOpticalFlow_getPyrScale(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setFastPyramids(struct PtrWrapper ptr, bool val);

bool FarnebackOpticalFlow_getFastPyramids(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setWinSize(struct PtrWrapper ptr, int val);

int FarnebackOpticalFlow_getWinSize(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setNumIters(struct PtrWrapper ptr, int val);

int FarnebackOpticalFlow_getNumIters(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setPolyN(struct PtrWrapper ptr, int val);

int FarnebackOpticalFlow_getPolyN(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setPolySigma(struct PtrWrapper ptr, double val);

double FarnebackOpticalFlow_getPolySigma(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setFlags(struct PtrWrapper ptr, int val);

int FarnebackOpticalFlow_getFlags(struct PtrWrapper ptr);

struct PtrWrapper OpticalFlowDual_TVL1_ctor(
        double tau, double lambda, double theta, int nscales, int warps, double epsilon, 
        int iterations, double scaleStep, double gamma, bool useInitialFlow);

void OpticalFlowDual_TVL1_setTau(struct PtrWrapper ptr, double val);

double OpticalFlowDual_TVL1_getTau(struct PtrWrapper ptr);

void OpticalFlowDual_TVL1_setLambda(struct PtrWrapper ptr, double val);

double OpticalFlowDual_TVL1_getLambda(struct PtrWrapper ptr);

void OpticalFlowDual_TVL1_setGamma(struct PtrWrapper ptr, double val);

double OpticalFlowDual_TVL1_getGamma(struct PtrWrapper ptr);

void OpticalFlowDual_TVL1_setTheta(struct PtrWrapper ptr, double val);

double OpticalFlowDual_TVL1_getTheta(struct PtrWrapper ptr);

void OpticalFlowDual_TVL1_setNumScales(struct PtrWrapper ptr, int val);

int OpticalFlowDual_TVL1_getNumScales(struct PtrWrapper ptr);

void OpticalFlowDual_TVL1_setNumWarps(struct PtrWrapper ptr, int val);

int OpticalFlowDual_TVL1_getNumWarps(struct PtrWrapper ptr);

void OpticalFlowDual_TVL1_setEpsilon(struct PtrWrapper ptr, double val);

double OpticalFlowDual_TVL1_getEpsilon(struct PtrWrapper ptr);

void OpticalFlowDual_TVL1_setNumIterations(struct PtrWrapper ptr, int val);

int OpticalFlowDual_TVL1_getNumIterations(struct PtrWrapper ptr);

void OpticalFlowDual_TVL1_setScaleStep(struct PtrWrapper ptr, double val);

double OpticalFlowDual_TVL1_getScaleStep(struct PtrWrapper ptr);

void OpticalFlowDual_TVL1_setUseInitialFlow(struct PtrWrapper ptr, bool val);

bool OpticalFlowDual_TVL1_getUseInitialFlow(struct PtrWrapper ptr);
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

        return cv.unwrap_tensors(C.DenseOpticalFlow_calc(
            cutorch._state, self.ptr, cv.argcheck(t, argRules)))
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

        return cv.unwrap_tensors(C.SparseOpticalFlow_calc(
            cutorch._state, self.ptr, cv.argcheck(t, argRules)))
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
        
        self.ptr = ffi.gc(C.BroxOpticalFlow_ctor(cv.argcheck(t, argRules)), Classes.Algorithm_dtor)
    end

    function BroxOpticalFlow:setFlowSmoothness(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.BroxOpticalFlow_setFlowSmoothness(self.ptr, val)
    end

    function BroxOpticalFlow:getFlowSmoothness()
        return C.BroxOpticalFlow_getFlowSmoothness(self.ptr)
    end

    function BroxOpticalFlow:setGradientConstancyImportance(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.BroxOpticalFlow_setGradientConstancyImportance(self.ptr, val)
    end

    function BroxOpticalFlow:getGradientConstancyImportance()
        return C.BroxOpticalFlow_getGradientConstancyImportance(self.ptr)
    end

    function BroxOpticalFlow:setPyramidScaleFactor(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.BroxOpticalFlow_setPyramidScaleFactor(self.ptr, val)
    end

    function BroxOpticalFlow:getPyramidScaleFactor()
        return C.BroxOpticalFlow_getPyramidScaleFactor(self.ptr)
    end

    function BroxOpticalFlow:setInnerIterations(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.BroxOpticalFlow_setInnerIterations(self.ptr, val)
    end

    function BroxOpticalFlow:getInnerIterations()
        return C.BroxOpticalFlow_getInnerIterations(self.ptr)
    end

    function BroxOpticalFlow:setOuterIterations(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.BroxOpticalFlow_setOuterIterations(self.ptr, val)
    end

    function BroxOpticalFlow:getOuterIterations()
        return C.BroxOpticalFlow_getOuterIterations(self.ptr)
    end

    function BroxOpticalFlow:setSolverIterations(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.BroxOpticalFlow_setSolverIterations(self.ptr, val)
    end

    function BroxOpticalFlow:getSolverIterations()
        return C.BroxOpticalFlow_getSolverIterations(self.ptr)
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
            C.SparsePyrLKOpticalFlow_ctor(cv.argcheck(t, argRules)), 
            Classes.Algorithm_dtor)
    end

    function SparsePyrLKOpticalFlow:setWinSize(t)
        local argRules = {
            {"val", required = true, operator = cv.Size}
        }
        local val = cv.argcheck(t, argRules)
        
        C.SparsePyrLKOpticalFlow_setWinSize(self.ptr, val)
    end

    function SparsePyrLKOpticalFlow:getWinSize()
        return C.SparsePyrLKOpticalFlow_getWinSize(self.ptr)
    end

    function SparsePyrLKOpticalFlow:setMaxLevel(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.SparsePyrLKOpticalFlow_setMaxLevel(self.ptr, val)
    end

    function SparsePyrLKOpticalFlow:getMaxLevel()
        return C.SparsePyrLKOpticalFlow_getMaxLevel(self.ptr)
    end

    function SparsePyrLKOpticalFlow:setNumIters(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.SparsePyrLKOpticalFlow_setNumIters(self.ptr, val)
    end

    function SparsePyrLKOpticalFlow:getNumIters()
        return C.SparsePyrLKOpticalFlow_getNumIters(self.ptr)
    end

    function SparsePyrLKOpticalFlow:setUseInitialFlow(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.SparsePyrLKOpticalFlow_setUseInitialFlow(self.ptr, val)
    end

    function SparsePyrLKOpticalFlow:getUseInitialFlow()
        return C.SparsePyrLKOpticalFlow_getUseInitialFlow(self.ptr)
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
            C.DensePyrLKOpticalFlow_ctor(cv.argcheck(t, argRules)), 
            Classes.Algorithm_dtor)
    end

    function DensePyrLKOpticalFlow:setWinSize(t)
        local argRules = {
            {"val", required = true, operator = cv.Size}
        }
        local val = cv.argcheck(t, argRules)
        
        C.DensePyrLKOpticalFlow_setWinSize(self.ptr, val)
    end

    function DensePyrLKOpticalFlow:getWinSize()
        return C.DensePyrLKOpticalFlow_getWinSize(self.ptr)
    end

    function DensePyrLKOpticalFlow:setMaxLevel(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.DensePyrLKOpticalFlow_setMaxLevel(self.ptr, val)
    end

    function DensePyrLKOpticalFlow:getMaxLevel()
        return C.DensePyrLKOpticalFlow_getMaxLevel(self.ptr)
    end

    function DensePyrLKOpticalFlow:setNumIters(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.DensePyrLKOpticalFlow_setNumIters(self.ptr, val)
    end

    function DensePyrLKOpticalFlow:getNumIters()
        return C.DensePyrLKOpticalFlow_getNumIters(self.ptr)
    end

    function DensePyrLKOpticalFlow:setUseInitialFlow(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.DensePyrLKOpticalFlow_setUseInitialFlow(self.ptr, val)
    end

    function DensePyrLKOpticalFlow:getUseInitialFlow()
        return C.DensePyrLKOpticalFlow_getUseInitialFlow(self.ptr)
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

        self.ptr = ffi.gc(C.FarnebackOpticalFlow_ctor(cv.argcheck(t, argRules)), Classes.Algorithm_dtor)
    end

    function FarnebackOpticalFlow:setNumLevels(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FarnebackOpticalFlow_setNumLevels(self.ptr, val)
    end

    function FarnebackOpticalFlow:getNumLevels()
        return C.FarnebackOpticalFlow_getNumLevels(self.ptr)
    end

    function FarnebackOpticalFlow:setPyrScale(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FarnebackOpticalFlow_setPyrScale(self.ptr, val)
    end

    function FarnebackOpticalFlow:getPyrScale()
        return C.FarnebackOpticalFlow_getPyrScale(self.ptr)
    end

    function FarnebackOpticalFlow:setFastPyramids(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FarnebackOpticalFlow_setFastPyramids(self.ptr, val)
    end

    function FarnebackOpticalFlow:getFastPyramids()
        return C.FarnebackOpticalFlow_getFastPyramids(self.ptr)
    end

    function FarnebackOpticalFlow:setWinSize(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FarnebackOpticalFlow_setWinSize(self.ptr, val)
    end

    function FarnebackOpticalFlow:getWinSize()
        return C.FarnebackOpticalFlow_getWinSize(self.ptr)
    end

    function FarnebackOpticalFlow:setNumIters(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FarnebackOpticalFlow_setNumIters(self.ptr, val)
    end

    function FarnebackOpticalFlow:getNumIters()
        return C.FarnebackOpticalFlow_getNumIters(self.ptr)
    end

    function FarnebackOpticalFlow:setPolyN(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FarnebackOpticalFlow_setPolyN(self.ptr, val)
    end

    function FarnebackOpticalFlow:getPolyN()
        return C.FarnebackOpticalFlow_getPolyN(self.ptr)
    end

    function FarnebackOpticalFlow:setPolySigma(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FarnebackOpticalFlow_setPolySigma(self.ptr, val)
    end

    function FarnebackOpticalFlow:getPolySigma()
        return C.FarnebackOpticalFlow_getPolySigma(self.ptr)
    end

    function FarnebackOpticalFlow:setFlags(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FarnebackOpticalFlow_setFlags(self.ptr, val)
    end

    function FarnebackOpticalFlow:getFlags()
        return C.FarnebackOpticalFlow_getFlags(self.ptr)
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

        self.ptr = ffi.gc(C.OpticalFlowDual_TVL1_ctor(cv.argcheck(t, argRules)), Classes.Algorithm_dtor)
    end

    function OpticalFlowDual_TVL1:setTau(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setTau(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getTau()
        return C.OpticalFlowDual_TVL1_getTau(self.ptr)
    end

    function OpticalFlowDual_TVL1:setLambda(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setLambda(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getLambda()
        return C.OpticalFlowDual_TVL1_getLambda(self.ptr)
    end

    function OpticalFlowDual_TVL1:setGamma(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setGamma(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getGamma()
        return C.OpticalFlowDual_TVL1_getGamma(self.ptr)
    end

    function OpticalFlowDual_TVL1:setTheta(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setTheta(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getTheta()
        return C.OpticalFlowDual_TVL1_getTheta(self.ptr)
    end

    function OpticalFlowDual_TVL1:setNumScales(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setNumScales(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getNumScales()
        return C.OpticalFlowDual_TVL1_getNumScales(self.ptr)
    end

    function OpticalFlowDual_TVL1:setNumWarps(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setNumWarps(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getNumWarps()
        return C.OpticalFlowDual_TVL1_getNumWarps(self.ptr)
    end

    function OpticalFlowDual_TVL1:setEpsilon(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setEpsilon(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getEpsilon()
        return C.OpticalFlowDual_TVL1_getEpsilon(self.ptr)
    end

    function OpticalFlowDual_TVL1:setNumIterations(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setNumIterations(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getNumIterations()
        return C.OpticalFlowDual_TVL1_getNumIterations(self.ptr)
    end

    function OpticalFlowDual_TVL1:setScaleStep(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setScaleStep(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getScaleStep()
        return C.OpticalFlowDual_TVL1_getScaleStep(self.ptr)
    end

    function OpticalFlowDual_TVL1:setUseInitialFlow(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.OpticalFlowDual_TVL1_setUseInitialFlow(self.ptr, val)
    end

    function OpticalFlowDual_TVL1:getUseInitialFlow()
        return C.OpticalFlowDual_TVL1_getUseInitialFlow(self.ptr)
    end
end

return cv.cuda
