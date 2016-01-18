local cv = require 'cv._env'

local ffi = require 'ffi'

local superres = {}

ffi.cdef[[
struct PtrWrapper createFrameSource();

struct PtrWrapper createFrameSource_Video(const char *fileName);

struct PtrWrapper createFrameSource_Video_CUDA(const char *fileName);

struct PtrWrapper createFrameSource_Camera(int deviceId);

void FrameSource_dtor(struct PtrWrapper ptr);

struct TensorWrapper FrameSource_nextFrame(struct PtrWrapper ptr, struct TensorWrapper frame);

void FrameSource_reset(struct PtrWrapper ptr);

struct PtrWrapper createSuperResolution_BTVL1();

struct PtrWrapper createSuperResolution_BTVL1_CUDA();

struct TensorWrapper SuperResolution_nextFrame(struct PtrWrapper ptr, struct TensorWrapper frame);

void SuperResolution_reset(struct PtrWrapper ptr);

void SuperResolution_setInput(struct PtrWrapper ptr, struct PtrWrapper frameSource);

void SuperResolution_collectGarbage(struct PtrWrapper ptr);

void SuperResolution_setScale(struct PtrWrapper ptr, int val);

int SuperResolution_getScale(struct PtrWrapper ptr);

void SuperResolution_setIterations(struct PtrWrapper ptr, int val);

int SuperResolution_getIterations(struct PtrWrapper ptr);

void SuperResolution_setTau(struct PtrWrapper ptr, double val);

double SuperResolution_getTau(struct PtrWrapper ptr);

void SuperResolution_setLabmda(struct PtrWrapper ptr, double val);

double SuperResolution_getLabmda(struct PtrWrapper ptr);

void SuperResolution_setAlpha(struct PtrWrapper ptr, double val);

double SuperResolution_getAlpha(struct PtrWrapper ptr);

void SuperResolution_setKernelSize(struct PtrWrapper ptr, int val);

int SuperResolution_getKernelSize(struct PtrWrapper ptr);

void SuperResolution_setBlurKernelSize(struct PtrWrapper ptr, int val);

int SuperResolution_getBlurKernelSize(struct PtrWrapper ptr);

void SuperResolution_setBlurSigma(struct PtrWrapper ptr, double val);

double SuperResolution_getBlurSigma(struct PtrWrapper ptr);

void SuperResolution_setTemporalAreaRadius(struct PtrWrapper ptr, int val);

int SuperResolution_getTemporalAreaRadius(struct PtrWrapper ptr);

void SuperResolution_setOpticalFlow(struct PtrWrapper ptr, struct PtrWrapper val);

struct PtrWrapper SuperResolution_getOpticalFlow(struct PtrWrapper ptr);

struct TensorArray DenseOpticalFlowExt_calc(
        struct PtrWrapper ptr, struct TensorWrapper frame0, struct TensorWrapper frame1,
        struct TensorWrapper flow1, struct TensorWrapper flow2);

void DenseOpticalFlowExt_collectGarbage(struct PtrWrapper ptr);

// FarnebackOpticalFlow

struct PtrWrapper createOptFlow_Farneback();

struct PtrWrapper createOptFlow_Farneback_CUDA();

void FarnebackOpticalFlow_setPyrScale(struct PtrWrapper ptr, double val);

double FarnebackOpticalFlow_getPyrScale(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setLevelsNumber(struct PtrWrapper ptr, int val);

int FarnebackOpticalFlow_getLevelsNumber(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setWindowSize(struct PtrWrapper ptr, int val);

int FarnebackOpticalFlow_getWindowSize(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setIterations(struct PtrWrapper ptr, int val);

int FarnebackOpticalFlow_getIterations(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setPolyN(struct PtrWrapper ptr, int val);

int FarnebackOpticalFlow_getPolyN(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setPolySigma(struct PtrWrapper ptr, int val);

double FarnebackOpticalFlow_getPolySigma(struct PtrWrapper ptr);

void FarnebackOpticalFlow_setFlags(struct PtrWrapper ptr, int val);

int FarnebackOpticalFlow_getFlags(struct PtrWrapper ptr);

struct PtrWrapper createOptFlow_DualTVL1();

struct PtrWrapper createOptFlow_DualTVL1_CUDA();

void DualTVL1OpticalFlow_setTau(struct PtrWrapper ptr, double val);

double DualTVL1OpticalFlow_getTau(struct PtrWrapper ptr);

void DualTVL1OpticalFlow_setLambda(struct PtrWrapper ptr, double val);

double DualTVL1OpticalFlow_getLambda(struct PtrWrapper ptr);

void DualTVL1OpticalFlow_setTheta(struct PtrWrapper ptr, double val);

double DualTVL1OpticalFlow_getTheta(struct PtrWrapper ptr);

void DualTVL1OpticalFlow_setScalesNumber(struct PtrWrapper ptr, int val);

int DualTVL1OpticalFlow_getScalesNumber(struct PtrWrapper ptr);

void DualTVL1OpticalFlow_setWarpingsNumber(struct PtrWrapper ptr, int val);

int DualTVL1OpticalFlow_getWarpingsNumber(struct PtrWrapper ptr);

void DualTVL1OpticalFlow_setEpsilon(struct PtrWrapper ptr, double val);

double DualTVL1OpticalFlow_getEpsilon(struct PtrWrapper ptr);

void DualTVL1OpticalFlow_setIterations(struct PtrWrapper ptr, int val);

int DualTVL1OpticalFlow_getIterations(struct PtrWrapper ptr);

void DualTVL1OpticalFlow_setUseInitialFlow(struct PtrWrapper ptr, bool val);

bool DualTVL1OpticalFlow_getUseInitialFlow(struct PtrWrapper ptr);

struct PtrWrapper createOptFlow_Brox_CUDA();

void BroxOpticalFlow_setAlpha(struct PtrWrapper ptr, double val);

double BroxOpticalFlow_getAlpha(struct PtrWrapper ptr);

void BroxOpticalFlow_setGamma(struct PtrWrapper ptr, double val);

double BroxOpticalFlow_getGamma(struct PtrWrapper ptr);

void BroxOpticalFlow_setScaleFactor(struct PtrWrapper ptr, double val);

double BroxOpticalFlow_getScaleFactor(struct PtrWrapper ptr);

void BroxOpticalFlow_setInnerIterations(struct PtrWrapper ptr, int val);

int BroxOpticalFlow_getInnerIterations(struct PtrWrapper ptr);

void BroxOpticalFlow_setOuterIterations(struct PtrWrapper ptr, int val);

int BroxOpticalFlow_getOuterIterations(struct PtrWrapper ptr);

void BroxOpticalFlow_setSolverIterations(struct PtrWrapper ptr, int val);

int BroxOpticalFlow_getSolverIterations(struct PtrWrapper ptr);

struct PtrWrapper createOptFlow_PyrLK_CUDA();

void PyrLKOpticalFlow_setWindowSize(struct PtrWrapper ptr, int val);

int PyrLKOpticalFlow_getWindowSize(struct PtrWrapper ptr);

void PyrLKOpticalFlow_setMaxLevel(struct PtrWrapper ptr, int val);

int PyrLKOpticalFlow_getMaxLevel(struct PtrWrapper ptr);

void PyrLKOpticalFlow_setIterations(struct PtrWrapper ptr, int val);

int PyrLKOpticalFlow_getIterations(struct PtrWrapper ptr);
]]

local C = ffi.load(cv.libPath('superres'))

-- FrameSource
require 'cv.Classes'

local Classes = ffi.load(cv.libPath('Classes'))

do
    local FrameSource = torch.class('cv.FrameSource', superres)

    function FrameSource:nextFrame(t)
        local argRules = {
            {"frame", default = nil}
        }
        local frame = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.FrameSource_nextFrame(self.ptr, cv.wrap_tensor(frame)))
    end

    function FrameSource:reset()
        C.FrameSource_reset(self.ptr)
    end
end

function superres.createFrameSource_Empty()
    local retval = torch.factory('cv.FrameSource')()
    retval.ptr = ffi.gc(C.createFrameSource_Empty(), C.FrameSource_dtor)
    return retval
end

function superres.createFrameSource_Video(t)
    local argRules = {
        {"fileName", required = true}
    }
    local fileName = cv.argcheck(t, argRules)

    local retval = torch.factory('cv.FrameSource')()
    retval.ptr = ffi.gc(C.createFrameSource_Video(fileName), C.FrameSource_dtor)
    return retval
end

function superres.createFrameSource_Video_CUDA(t)
    local argRules = {
        {"fileName", required = true}
    }
    local fileName = cv.argcheck(t, argRules)

    local retval = torch.factory('cv.FrameSource')()
    retval.ptr = ffi.gc(C.createFrameSource_Video_CUDA(fileName), C.FrameSource_dtor)
    return retval
end

function superres.createFrameSource_Camera(t)
    local argRules = {
        {"deviceId", default = 0}
    }
    local deviceId = cv.argcheck(t, argRules)

    local retval = torch.factory('cv.FrameSource')()
    retval.ptr = ffi.gc(C.createFrameSource_Camera(deviceId), C.FrameSource_dtor)
    return retval
end

-- DenseOpticalFlowExt

do
    local DenseOpticalFlowExt = torch.class('cv.DenseOpticalFlowExt', 'cv.Algorithm', superres)

    function DenseOpticalFlowExt:calc(t)
        local argRules = {
            {"frame0", required = true},
            {"frame1", required = true},
            {"flow1", required = true},
            {"flow2", default = nil}
        }
        local frame0, frame1, flow1, flow2 = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.DenseOpticalFlowExt_calc(
            self.ptr, cv.wrap_tensor(frame0), cv.wrap_tensor(frame1),
            cv.wrap_tensor(flow1), cv.wrap_tensor(flow2)))
    end

    function DenseOpticalFlowExt:collectGarbage()
        C.DenseOpticalFlowExt_collectGarbage(self.ptr)
    end
end

-- SuperResolution

do
    local SuperResolution = torch.class('cv.SuperResolution', 'cv.Algorithm', superres)

    function SuperResolution:nextFrame(t)
        local argRules = {
            {"frame", default = nil}
        }
        local frame = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.SuperResolution_nextFrame(self.ptr, cv.wrap_tensor(frame)))
    end

    function SuperResolution:reset()
        C.SuperResolution_reset(self.ptr)
    end

    function SuperResolution:setInput(t)
        local argRules = {
            {"frameSource", required = true}
        }
        local frameSource = cv.argcheck(t, argRules)

        self.frameSource = frameSource
        C.SuperResolution_setInput(self.ptr, frameSource.ptr)
    end

    function SuperResolution:collectGarbage()
        C.SuperResolution_collectGarbage(self.ptr)
    end

    function SuperResolution:setScale(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.SuperResolution_setScale(self.ptr, val)
    end

    function SuperResolution:getScale()
        return C.SuperResolution_getScale(self.ptr)
    end

    function SuperResolution:setIterations(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.SuperResolution_setIterations(self.ptr, val)
    end

    function SuperResolution:getIterations()
        return C.SuperResolution_getIterations(self.ptr)
    end

    function SuperResolution:setTau(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.SuperResolution_setTau(self.ptr, val)
    end

    function SuperResolution:getTau()
        return C.SuperResolution_getTau(self.ptr)
    end

    function SuperResolution:setLabmda(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.SuperResolution_setLabmda(self.ptr, val)
    end

    function SuperResolution:getLabmda()
        return C.SuperResolution_getLabmda(self.ptr)
    end

    function SuperResolution:setAlpha(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.SuperResolution_setAlpha(self.ptr, val)
    end

    function SuperResolution:getAlpha()
        return C.SuperResolution_getAlpha(self.ptr)
    end

    function SuperResolution:setKernelSize(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.SuperResolution_setKernelSize(self.ptr, val)
    end

    function SuperResolution:getKernelSize()
        return C.SuperResolution_getKernelSize(self.ptr)
    end

    function SuperResolution:setBlurKernelSize(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.SuperResolution_setBlurKernelSize(self.ptr, val)
    end

    function SuperResolution:getBlurKernelSize()
        return C.SuperResolution_getBlurKernelSize(self.ptr)
    end

    function SuperResolution:setBlurSigma(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.SuperResolution_setBlurSigma(self.ptr, val)
    end

    function SuperResolution:getBlurSigma()
        return C.SuperResolution_getBlurSigma(self.ptr)
    end

    function SuperResolution:setTemporalAreaRadius(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.SuperResolution_setTemporalAreaRadius(self.ptr, val)
    end

    function SuperResolution:getTemporalAreaRadius()
        return C.SuperResolution_getTemporalAreaRadius(self.ptr)
    end

    function SuperResolution:setOpticalFlow(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        self.optFlow = val
        C.SuperResolution_setOpticalFlow(self.ptr, val.ptr)
    end

    -- I hope you never use it
    function SuperResolution:getOpticalFlow()
        local retval = torch.factory('cv.DenseOpticalFlowExt')()
        -- don't destroy it
        retval.ptr = C.SuperResolution_getOpticalFlow(self.ptr)
        return retval
    end
end

function superres.createSuperResolution_BTVL1()
    local retval = torch.factory('cv.SuperResolution')()
    retval.ptr = ffi.gc(C.createSuperResolution_BTVL1(), Classes.Algorithm_dtor)
    return retval
end

function superres.createSuperResolution_BTVL1_CUDA()
    local retval = torch.factory('cv.SuperResolution')()
    retval.ptr = ffi.gc(C.createSuperResolution_BTVL1_CUDA(), Classes.Algorithm_dtor)
    return retval
end

-- FarnebackOpticalFlow

do
    local FarnebackOpticalFlow = torch.class('cv.FarnebackOpticalFlow', 'cv.DenseOpticalFlowExt', superres)

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

    function FarnebackOpticalFlow:setLevelsNumber(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.FarnebackOpticalFlow_setLevelsNumber(self.ptr, val)
    end

    function FarnebackOpticalFlow:getLevelsNumber()
        return C.FarnebackOpticalFlow_getLevelsNumber(self.ptr)
    end

    function FarnebackOpticalFlow:setWindowSize(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.FarnebackOpticalFlow_setWindowSize(self.ptr, val)
    end

    function FarnebackOpticalFlow:getWindowSize()
        return C.FarnebackOpticalFlow_getWindowSize(self.ptr)
    end

    function FarnebackOpticalFlow:setIterations(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.FarnebackOpticalFlow_setIterations(self.ptr, val)
    end

    function FarnebackOpticalFlow:getIterations()
        return C.FarnebackOpticalFlow_getIterations(self.ptr)
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

function superres.createOptFlow_Farneback()
    local retval = torch.factory('cv.FarnebackOpticalFlow')()
    retval.ptr = ffi.gc(C.createOptFlow_Farneback(), Classes.Algorithm_dtor)
    return retval
end

function superres.createOptFlow_Farneback_CUDA()
    local retval = torch.factory('cv.FarnebackOpticalFlow')()
    retval.ptr = ffi.gc(C.createOptFlow_Farneback_CUDA(), Classes.Algorithm_dtor)
    return retval
end

-- DualTVL1OpticalFlow

do
    local DualTVL1OpticalFlow = torch.class('cv.DualTVL1OpticalFlow', 'cv.DenseOpticalFlowExt', superres)

    function DualTVL1OpticalFlow:setTau(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setTau(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getTau()
        return C.DualTVL1OpticalFlow_getTau(self.ptr)
    end

    function DualTVL1OpticalFlow:setLambda(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setLambda(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getLambda()
        return C.DualTVL1OpticalFlow_getLambda(self.ptr)
    end

    function DualTVL1OpticalFlow:setTheta(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setTheta(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getTheta()
        return C.DualTVL1OpticalFlow_getTheta(self.ptr)
    end

    function DualTVL1OpticalFlow:setScalesNumber(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setScalesNumber(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getScalesNumber()
        return C.DualTVL1OpticalFlow_getScalesNumber(self.ptr)
    end

    function DualTVL1OpticalFlow:setWarpingsNumber(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setWarpingsNumber(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getWarpingsNumber()
        return C.DualTVL1OpticalFlow_getWarpingsNumber(self.ptr)
    end

    function DualTVL1OpticalFlow:setEpsilon(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setEpsilon(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getEpsilon()
        return C.DualTVL1OpticalFlow_getEpsilon(self.ptr)
    end

    function DualTVL1OpticalFlow:setIterations(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setIterations(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getIterations()
        return C.DualTVL1OpticalFlow_getIterations(self.ptr)
    end

    function DualTVL1OpticalFlow:setUseInitialFlow(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.DualTVL1OpticalFlow_setUseInitialFlow(self.ptr, val)
    end

    function DualTVL1OpticalFlow:getUseInitialFlow()
        return C.DualTVL1OpticalFlow_getUseInitialFlow(self.ptr)
    end
end

function superres.createOptFlow_DualTVL1()
    local retval = torch.factory('cv.DualTVL1OpticalFlow')()
    retval.ptr = ffi.gc(C.createOptFlow_DualTVL1(), Classes.Algorithm_dtor)
    return retval
end

function superres.createOptFlow_DualTVL1_CUDA()
    local retval = torch.factory('cv.DualTVL1OpticalFlow')()
    retval.ptr = ffi.gc(C.createOptFlow_DualTVL1_CUDA(), Classes.Algorithm_dtor)
    return retval
end

-- BroxOpticalFlow

do
    local BroxOpticalFlow = torch.class('cv.BroxOpticalFlow', 'cv.DenseOpticalFlowExt', superres)

    function BroxOpticalFlow:setAlpha(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.BroxOpticalFlow_setAlpha(self.ptr, val)
    end

    function BroxOpticalFlow:getAlpha()
        return C.BroxOpticalFlow_getAlpha(self.ptr)
    end

    function BroxOpticalFlow:setGamma(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.BroxOpticalFlow_setGamma(self.ptr, val)
    end

    function BroxOpticalFlow:getGamma()
        return C.BroxOpticalFlow_getGamma(self.ptr)
    end

    function BroxOpticalFlow:setScaleFactor(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.BroxOpticalFlow_setScaleFactor(self.ptr, val)
    end

    function BroxOpticalFlow:getScaleFactor()
        return C.BroxOpticalFlow_getScaleFactor(self.ptr)
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

function superres.createOptFlow_Brox_CUDA()
    local retval = torch.factory('cv.BroxOpticalFlow')()
    retval.ptr = ffi.gc(C.createOptFlow_Brox_CUDA(), Classes.Algorithm_dtor)
    return retval
end

-- PyrLKOpticalFlow

do
    local PyrLKOpticalFlow = torch.class('cv.PyrLKOpticalFlow', 'cv.DenseOpticalFlowExt', superres)

    function PyrLKOpticalFlow:setWindowSize(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.PyrLKOpticalFlow_setWindowSize(self.ptr, val)
    end

    function PyrLKOpticalFlow:getWindowSize()
        return C.PyrLKOpticalFlow_getWindowSize(self.ptr)
    end

    function PyrLKOpticalFlow:setMaxLevel(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.PyrLKOpticalFlow_setMaxLevel(self.ptr, val)
    end

    function PyrLKOpticalFlow:getMaxLevel()
        return C.PyrLKOpticalFlow_getMaxLevel(self.ptr)
    end

    function PyrLKOpticalFlow:setIterations(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)

        C.PyrLKOpticalFlow_setIterations(self.ptr, val)
    end

    function PyrLKOpticalFlow:getIterations()
        return C.PyrLKOpticalFlow_getIterations(self.ptr)
    end
end

function superres.createOptFlow_PyrLK_CUDA()
    local retval = torch.factory('cv.PyrLKOpticalFlow')()
    retval.ptr = ffi.gc(C.createOptFlow_PyrLK_CUDA(), Classes.Algorithm_dtor)
    return retval
end

return superres
