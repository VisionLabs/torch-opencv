local cv = require 'cv._env'
require 'cutorch'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or require 'cv._env_cuda'

local ffi = require 'ffi'

ffi.cdef[[

]]

local C = ffi.load(cv.libPath('cudaobjdetect'))

require 'cv.Classes'
local Classes = ffi.load(cv.libPath('Classes'))

do
    local HOG = torch.class('cuda.HOG', 'cv.Algorithm', cv.cuda)

    function HOG:__init()
        local argRules = {
            {"win_size", default = {64, 128}, operator = cv.Size},
            {"block_size", default = {16, 16}, operator = cv.Size},
            {"block_stride", default = {8, 8}, operator = cv.Size},
            {"cell_size", default = {8, 8}, operator = cv.Size},
            {"nbins", default = 9}
        }
        self.ptr = ffi.gc(C.HOG_ctor(cv.argcheck(t, argRules)), Classes.Algorithm_dtor)
    end

    function HOG:setWinSigma(t)
        local argRules = {
            {"val", required = true}
        }
        C.HOG_setWinSigma(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getWinSigma()
        return C.HOG_getWinSigma(self.ptr)
    end

    function HOG:setL2HysThreshold(t)
        local argRules = {
            {"val", required = true}
        }
        C.HOG_setL2HysThreshold(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getL2HysThreshold()
        return C.HOG_getL2HysThreshold(self.ptr)
    end

    function HOG:setGammaCorrection(t)
        local argRules = {
            {"val", required = true}
        }
        C.HOG_setGammaCorrection(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getGammaCorrection()
        return C.HOG_getGammaCorrection(self.ptr)
    end

    function HOG:setNumLevels(t)
        local argRules = {
            {"val", required = true}
        }
        C.HOG_setNumLevels(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getNumLevels()
        return C.HOG_getNumLevels(self.ptr)
    end

    function HOG:setHitThreshold(t)
        local argRules = {
            {"val", required = true}
        }
        C.HOG_setHitThreshold(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getHitThreshold()
        return C.HOG_getHitThreshold(self.ptr)
    end

    function HOG:setWinStride(t)
        local argRules = {
            {"val", required = true, operator = cv.Size}
        }
        C.HOG_setWinStride(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getWinStride()
        return C.HOG_getWinStride(self.ptr)
    end

    function HOG:setScaleFactor(t)
        local argRules = {
            {"val", required = true}
        }
        C.HOG_setScaleFactor(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getScaleFactor()
        return C.HOG_getScaleFactor(self.ptr)
    end

    function HOG:setGroupThreshold(t)
        local argRules = {
            {"val", required = true}
        }
        C.HOG_setGroupThreshold(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getGroupThreshold()
        return C.HOG_getGroupThreshold(self.ptr)
    end

    function HOG:setDescriptorFormat(t)
        local argRules = {
            {"val", required = true}
        }
        C.HOG_setDescriptorFormat(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getDescriptorFormat()
        return C.HOG_getDescriptorFormat(self.ptr)
    end

    function HOG:getDescriptorSize()
        return C.HOG_getDescriptorSize(self.ptr)
    end

    function HOG:getBlockHistogramSize()
        return C.HOG_getBlockHistogramSize(self.ptr)
    end

    function HOG:setSVMDetector(t)
        local argRules = {
            {"val", required = true, operator = cv.wrap_tensor}
        }
        C.HOG_setSVMDetector(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getDefaultPeopleDetector()
        return C.HOG_getDefaultPeopleDetector(self.ptr)
    end

    function HOG:detect(t)
        local argRules = {
            {"img", required = true, operator = cv.wrap_tensor}
        }
        local retval = C.HOG_detect(cv.cuda._info(), self.ptr, cv.argcheck(t, argRules))
        return retval.points, cv.unwrap_tensors(retval.tensor)
    end

    function HOG:detectMultiScale(t)
        local argRules = {
            {"img", required = true, operator = cv.wrap_tensor}
        }
        local retval = C.HOG_detectMultiScale(cv.cuda._info(), self.ptr, cv.argcheck(t, argRules))
        return retval.rects, cv.unwrap_tensors(retval.tensor)
    end

    function HOG:compute(t)
        local argRules = {
            {"img", required = true, operator = cv.wrap_tensor},
            {"descriptors", default = nil, operator = cv.wrap_tensor}
        }
        return cv.unwrap_tensors(C.HOG_compute(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end
end


return cv.cuda
