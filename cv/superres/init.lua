local cv = require 'cv._env'

local ffi = require 'ffi'

ffi.cdef[[
struct PtrWrapper createFrameSource();

struct PtrWrapper createFrameSource_Video(const char *fileName);

struct PtrWrapper createFrameSource_Video_CUDA(const char *fileName);

struct PtrWrapper createFrameSource_Camera(int deviceId);

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
]]

local C = ffi.load(cv.libPath('superres'))

-- FrameSource
require 'cv.Classes'

local Classes = ffi.load(cv.libPath('Classes'))

do
    local FrameSource = cv.newTorchClass('cv.FrameSource')

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

function cv.createFrameSource_Empty()
    local retval = torch.factory('cv.FrameSource')()
    retval.ptr = ffi.gc(C.createFrameSource_Empty(), C.FrameSource_dtor)
    return retval
end

function cv.createFrameSource_Video(t)
    local argRules = {
        {"fileName", required = true}
    }
    local fileName = cv.argcheck(t, argRules)

    local retval = torch.factory('cv.FrameSource')()
    retval.ptr = ffi.gc(C.createFrameSource_Video(fileName), C.FrameSource_dtor)
    return retval
end

function cv.createFrameSource_Video_CUDA(t)
    local argRules = {
        {"fileName", required = true}
    }
    local fileName = cv.argcheck(t, argRules)

    local retval = torch.factory('cv.FrameSource')()
    retval.ptr = ffi.gc(C.createFrameSource_Video_CUDA(fileName), C.FrameSource_dtor)
    return retval
end

function cv.createFrameSource_Camera(t)
    local argRules = {
        {"deviceId", default = 0}
    }
    local deviceId = cv.argcheck(t, argRules)

    local retval = torch.factory('cv.FrameSource')()
    retval.ptr = ffi.gc(C.createFrameSource_Camera(deviceId), C.FrameSource_dtor)
    return retval
end

-- SuperResolution

do
    local SuperResolution = cv.newTorchClass('cv.SuperResolution', 'cv.Algorithm')

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

    -- TODO this
    --[[
    function SuperResolution:setOpticalFlow(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.SuperResolution_setOpticalFlow(self.ptr, val)
    end

    function SuperResolution:getOpticalFlow()
        return C.SuperResolution_getOpticalFlow(self.ptr)
    end
    --]]
end

function cv.createSuperResolution_BTVL1()
    local retval = torch.factory('cv.SuperResolution')()
    retval.ptr = ffi.gc(C.createSuperResolution_BTVL1(), Classes.Algorithm_dtor)
    return retval
end

function cv.createSuperResolution_BTVL1_CUDA()
    local retval = torch.factory('cv.SuperResolution')()
    retval.ptr = ffi.gc(C.createSuperResolution_BTVL1_CUDA(), Classes.Algorithm_dtor)
    return retval
end

return cv
