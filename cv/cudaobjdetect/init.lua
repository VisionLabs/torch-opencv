local cv = require 'cv._env'
require 'cutorch'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or require 'cv._env_cuda'

local ffi = require 'ffi'

ffi.cdef[[
struct PtrWrapper HOG_ctorCuda(
        struct SizeWrapper win_size, struct SizeWrapper block_size,
        struct SizeWrapper block_stride, struct SizeWrapper cell_size, int nbins);

void HOG_setWinSigmaCuda(struct PtrWrapper ptr, double val);

double HOG_getWinSigmaCuda(struct PtrWrapper ptr);

void HOG_setL2HysThresholdCuda(struct PtrWrapper ptr, double val);

double HOG_getL2HysThresholdCuda(struct PtrWrapper ptr);

void HOG_setGammaCorrectionCuda(struct PtrWrapper ptr, bool val);

bool HOG_getGammaCorrectionCuda(struct PtrWrapper ptr);

void HOG_setNumLevelsCuda(struct PtrWrapper ptr, int val);

int HOG_getNumLevelsCuda(struct PtrWrapper ptr);

void HOG_setHitThresholdCuda(struct PtrWrapper ptr, double val);

double HOG_getHitThresholdCuda(struct PtrWrapper ptr);

void HOG_setWinStrideCuda(struct PtrWrapper ptr, struct SizeWrapper val);

struct SizeWrapper HOG_getWinStrideCuda(struct PtrWrapper ptr);

void HOG_setScaleFactorCuda(struct PtrWrapper ptr, double val);

double HOG_getScaleFactorCuda(struct PtrWrapper ptr);

void HOG_setGroupThresholdCuda(struct PtrWrapper ptr, int val);

int HOG_getGroupThresholdCuda(struct PtrWrapper ptr);

void HOG_setDescriptorFormatCuda(struct PtrWrapper ptr, int val);

int HOG_getDescriptorFormatCuda(struct PtrWrapper ptr);

size_t HOG_getDescriptorSizeCuda(struct PtrWrapper ptr);

size_t HOG_getBlockHistogramSizeCuda(struct PtrWrapper ptr);

void HOG_setSVMDetectorCuda(struct PtrWrapper ptr, struct TensorWrapper val);

struct TensorWrapper HOG_getDefaultPeopleDetectorCuda(struct PtrWrapper ptr);

struct TensorPlusPointArray HOG_detectCuda(
        struct cutorchInfo info, struct PtrWrapper ptr, struct TensorWrapper img);

struct TensorPlusRectArray HOG_detectMultiScaleCuda(
        struct cutorchInfo info, struct PtrWrapper ptr, struct TensorWrapper img);

struct TensorWrapper HOG_computeCuda(
        struct cutorchInfo info, struct PtrWrapper ptr, struct TensorWrapper img,
        struct TensorWrapper descriptors);

struct PtrWrapper CascadeClassifier_ctor_filenameCuda(const char *filename);

struct PtrWrapper CascadeClassifier_ctor_fileCuda(struct FileStoragePtr file);

void CascadeClassifier_setMaxObjectSizeCuda(struct PtrWrapper ptr, struct SizeWrapper val);

struct SizeWrapper CascadeClassifier_getMaxObjectSizeCuda(struct PtrWrapper ptr);

void CascadeClassifier_setMinObjectSizeCuda(struct PtrWrapper ptr, struct SizeWrapper val);

struct SizeWrapper CascadeClassifier_getMinObjectSizeCuda(struct PtrWrapper ptr);

void CascadeClassifier_setScaleFactorCuda(struct PtrWrapper ptr, double val);

double CascadeClassifier_getScaleFactorCuda(struct PtrWrapper ptr);

void CascadeClassifier_setMinNeighborsCuda(struct PtrWrapper ptr, int val);

int CascadeClassifier_getMinNeighborsCuda(struct PtrWrapper ptr);

void CascadeClassifier_setFindLargestObjectCuda(struct PtrWrapper ptr, bool val);

bool CascadeClassifier_getFindLargestObjectCuda(struct PtrWrapper ptr);

void CascadeClassifier_setMaxNumObjectsCuda(struct PtrWrapper ptr, int val);

int CascadeClassifier_getMaxNumObjectsCuda(struct PtrWrapper ptr);

struct SizeWrapper CascadeClassifier_getClassifierSizeCuda(struct PtrWrapper ptr);

struct TensorWrapper CascadeClassifier_detectMultiScaleCuda(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper image, struct TensorWrapper objects);

struct RectArray CascadeClassifier_convertCuda(
        struct PtrWrapper ptr, struct TensorWrapper gpu_objects);
]]

local C = ffi.load(cv.libPath('cudaobjdetect'))

require 'cv.Classes'
local Classes = ffi.load(cv.libPath('Classes'))

do
    local HOG = torch.class('cuda.HOG', 'cv.Algorithm', cv.cuda)

    function HOG:__init(t)
        local argRules = {
            {"win_size", default = {64, 128}, operator = cv.Size},
            {"block_size", default = {16, 16}, operator = cv.Size},
            {"block_stride", default = {8, 8}, operator = cv.Size},
            {"cell_size", default = {8, 8}, operator = cv.Size},
            {"nbins", default = 9}
        }
        self.ptr = ffi.gc(C.HOG_ctorCuda(cv.argcheck(t, argRules)), Classes.Algorithm_dtor)
    end

    function HOG:setWinSigma(t)
        local argRules = {
            {"val", required = true}
        }
        C.HOG_setWinSigmaCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getWinSigma()
        return C.HOG_getWinSigmaCuda(self.ptr)
    end

    function HOG:setL2HysThreshold(t)
        local argRules = {
            {"val", required = true}
        }
        C.HOG_setL2HysThresholdCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getL2HysThreshold()
        return C.HOG_getL2HysThresholdCuda(self.ptr)
    end

    function HOG:setGammaCorrection(t)
        local argRules = {
            {"val", required = true}
        }
        C.HOG_setGammaCorrectionCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getGammaCorrection()
        return C.HOG_getGammaCorrectionCuda(self.ptr)
    end

    function HOG:setNumLevels(t)
        local argRules = {
            {"val", required = true}
        }
        C.HOG_setNumLevelsCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getNumLevels()
        return C.HOG_getNumLevelsCuda(self.ptr)
    end

    function HOG:setHitThreshold(t)
        local argRules = {
            {"val", required = true}
        }
        C.HOG_setHitThresholdCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getHitThreshold()
        return C.HOG_getHitThresholdCuda(self.ptr)
    end

    function HOG:setWinStride(t)
        local argRules = {
            {"val", required = true, operator = cv.Size}
        }
        C.HOG_setWinStrideCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getWinStride()
        return C.HOG_getWinStrideCuda(self.ptr)
    end

    function HOG:setScaleFactor(t)
        local argRules = {
            {"val", required = true}
        }
        C.HOG_setScaleFactorCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getScaleFactor()
        return C.HOG_getScaleFactorCuda(self.ptr)
    end

    function HOG:setGroupThreshold(t)
        local argRules = {
            {"val", required = true}
        }
        C.HOG_setGroupThresholdCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getGroupThreshold()
        return C.HOG_getGroupThresholdCuda(self.ptr)
    end

    function HOG:setDescriptorFormat(t)
        local argRules = {
            {"val", required = true}
        }
        C.HOG_setDescriptorFormatCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getDescriptorFormat()
        return C.HOG_getDescriptorFormatCuda(self.ptr)
    end

    function HOG:getDescriptorSize()
        return C.HOG_getDescriptorSizeCuda(self.ptr)
    end

    function HOG:getBlockHistogramSize()
        return C.HOG_getBlockHistogramSizeCuda(self.ptr)
    end

    function HOG:setSVMDetector(t)
        local argRules = {
            {"val", required = true, operator = cv.wrap_tensor}
        }
        C.HOG_setSVMDetectorCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function HOG:getDefaultPeopleDetector()
        return cv.unwrap_tensors(C.HOG_getDefaultPeopleDetectorCuda(self.ptr))
    end

    function HOG:detect(t)
        local argRules = {
            {"img", required = true, operator = cv.wrap_tensor}
        }
        local retval = C.HOG_detectCuda(cv.cuda._info(), self.ptr, cv.argcheck(t, argRules))
        return retval.points, cv.unwrap_tensors(retval.tensor)
    end

    function HOG:detectMultiScale(t)
        local argRules = {
            {"img", required = true, operator = cv.wrap_tensor}
        }
        local retval = C.HOG_detectMultiScaleCuda(cv.cuda._info(), self.ptr, cv.argcheck(t, argRules))
        return retval.rects, cv.unwrap_tensors(retval.tensor)
    end

    function HOG:compute(t)
        local argRules = {
            {"img", required = true, operator = cv.wrap_tensor},
            {"descriptors", default = nil, operator = cv.wrap_tensor}
        }
        return cv.unwrap_tensors(C.HOG_computeCuda(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end
end


do
    local CascadeClassifier = torch.class('cuda.CascadeClassifier', 'cv.Algorithm', cv.cuda)

    function CascadeClassifier:__init(t)
        local argRules = {
            {"file_or_filename", required = true}
        }
        local f = cv.argcheck(t, argRules)
        if type(f) == 'string' then
            self.ptr = ffi.gc(C.CascadeClassifier_ctor_filenameCuda(f), Classes.Algorithm_dtor)
        else
            self.ptr = ffi.gc(C.CascadeClassifier_ctor_fileCuda(f), Classes.Algorithm_dtor)
        end
    end

    function CascadeClassifier:setMaxObjectSize(t)
        local argRules = {
            {"val", required = true, operator = cv.Size}
        }
        C.CascadeClassifier_setMaxObjectSizeCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function CascadeClassifier:getMaxObjectSize()
        return C.CascadeClassifier_getMaxObjectSizeCuda(self.ptr)
    end

    function CascadeClassifier:setMinObjectSize(t)
        local argRules = {
            {"val", required = true, operator = cv.Size}
        }
        C.CascadeClassifier_setMinObjectSizeCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function CascadeClassifier:getMinObjectSize()
        return C.CascadeClassifier_getMinObjectSizeCuda(self.ptr)
    end

    function CascadeClassifier:setScaleFactor(t)
        local argRules = {
            {"val", required = true}
        }
        C.CascadeClassifier_setScaleFactorCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function CascadeClassifier:getScaleFactor()
        return C.CascadeClassifier_getScaleFactorCuda(self.ptr)
    end

    function CascadeClassifier:setMinNeighbors(t)
        local argRules = {
            {"val", required = true}
        }
        C.CascadeClassifier_setMinNeighborsCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function CascadeClassifier:getMinNeighbors()
        return C.CascadeClassifier_getMinNeighborsCuda(self.ptr)
    end

    function CascadeClassifier:setFindLargestObject(t)
        local argRules = {
            {"val", required = true}
        }
        C.CascadeClassifier_setFindLargestObjectCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function CascadeClassifier:getFindLargestObject()
        return C.CascadeClassifier_getFindLargestObjectCuda(self.ptr)
    end

    function CascadeClassifier:setMaxNumObjects(t)
        local argRules = {
            {"val", required = true}
        }
        C.CascadeClassifier_setMaxNumObjectsCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function CascadeClassifier:getMaxNumObjects()
        return C.CascadeClassifier_getMaxNumObjectsCuda(self.ptr)
    end

    function CascadeClassifier:getClassifierSize()
        return C.CascadeClassifier_getClassifierSizeCuda(self.ptr)
    end

    function CascadeClassifier:detectMultiScale(t)
        local argRules = {
            {"image", required = true, operator = cv.wrap_tensor},
            {"objects", default = nil, operator = cv.wrap_tensor}
        }
        return cv.unwrap_tensors(C.CascadeClassifier_detectMultiScaleCuda(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end

    function CascadeClassifier:convert(t)
        local argRules = {
            {"gpu_objects", required = true, operator = cv.wrap_tensor}
        }
        return C.CascadeClassifier_convertCuda(self.ptr, cv.argcheck(t, argRules))
    end
end

return cv.cuda
