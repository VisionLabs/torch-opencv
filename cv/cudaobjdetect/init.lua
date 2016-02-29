local cv = require 'cv._env'
require 'cutorch'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or require 'cv._env_cuda'

local ffi = require 'ffi'

ffi.cdef[[
struct PtrWrapper HOG_ctor(
        struct SizeWrapper win_size, struct SizeWrapper block_size,
        struct SizeWrapper block_stride, struct SizeWrapper cell_size, int nbins);

void HOG_setWinSigma(struct PtrWrapper ptr, double val);

double HOG_getWinSigma(struct PtrWrapper ptr);

void HOG_setL2HysThreshold(struct PtrWrapper ptr, double val);

double HOG_getL2HysThreshold(struct PtrWrapper ptr);

void HOG_setGammaCorrection(struct PtrWrapper ptr, bool val);

bool HOG_getGammaCorrection(struct PtrWrapper ptr);

void HOG_setNumLevels(struct PtrWrapper ptr, int val);

int HOG_getNumLevels(struct PtrWrapper ptr);

void HOG_setHitThreshold(struct PtrWrapper ptr, double val);

double HOG_getHitThreshold(struct PtrWrapper ptr);

void HOG_setWinStride(struct PtrWrapper ptr, struct SizeWrapper val);

struct SizeWrapper HOG_getWinStride(struct PtrWrapper ptr);

void HOG_setScaleFactor(struct PtrWrapper ptr, double val);

double HOG_getScaleFactor(struct PtrWrapper ptr);

void HOG_setGroupThreshold(struct PtrWrapper ptr, int val);

int HOG_getGroupThreshold(struct PtrWrapper ptr);

void HOG_setDescriptorFormat(struct PtrWrapper ptr, int val);

int HOG_getDescriptorFormat(struct PtrWrapper ptr);

size_t HOG_getDescriptorSize(struct PtrWrapper ptr);

size_t HOG_getBlockHistogramSize(struct PtrWrapper ptr);

void HOG_setSVMDetector(struct PtrWrapper ptr, struct TensorWrapper val);

struct TensorWrapper HOG_getDefaultPeopleDetector(struct PtrWrapper ptr);

struct TensorPlusPointArray HOG_detect(
        struct cutorchInfo info, struct PtrWrapper ptr, struct TensorWrapper img);

struct TensorPlusRectArray HOG_detectMultiScale(
        struct cutorchInfo info, struct PtrWrapper ptr, struct TensorWrapper img);

struct TensorWrapper HOG_compute(
        struct cutorchInfo info, struct PtrWrapper ptr, struct TensorWrapper img,
        struct TensorWrapper descriptors);

struct PtrWrapper CascadeClassifier_ctor_filename(const char *filename);

struct PtrWrapper CascadeClassifier_ctor_file(struct FileStoragePtr file);

void CascadeClassifier_setMaxObjectSize(struct PtrWrapper ptr, struct SizeWrapper val);

struct SizeWrapper CascadeClassifier_getMaxObjectSize(struct PtrWrapper ptr);

void CascadeClassifier_setMinObjectSize(struct PtrWrapper ptr, struct SizeWrapper val);

struct SizeWrapper CascadeClassifier_getMinObjectSize(struct PtrWrapper ptr);

void CascadeClassifier_setScaleFactor(struct PtrWrapper ptr, double val);

double CascadeClassifier_getScaleFactor(struct PtrWrapper ptr);

void CascadeClassifier_setMinNeighbors(struct PtrWrapper ptr, int val);

int CascadeClassifier_getMinNeighbors(struct PtrWrapper ptr);

void CascadeClassifier_setFindLargestObject(struct PtrWrapper ptr, bool val);

bool CascadeClassifier_getFindLargestObject(struct PtrWrapper ptr);

void CascadeClassifier_setMaxNumObjects(struct PtrWrapper ptr, int val);

int CascadeClassifier_getMaxNumObjects(struct PtrWrapper ptr);

struct SizeWrapper CascadeClassifier_getClassifierSize(struct PtrWrapper ptr);

struct TensorWrapper CascadeClassifier_detectMultiScale(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper image, struct TensorWrapper objects);

struct RectArray CascadeClassifier_convert(
        struct PtrWrapper ptr, struct TensorWrapper gpu_objects);
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


do
    local CascadeClassifier = torch.class('cuda.CascadeClassifier', 'cv.Algorithm', cv.cuda)

    function CascadeClassifier:__init(t)
        local argRules = {
            {"file_or_filename", required = true}
        }
        local f = cv.argcheck(t, argRules)
        if type(f) == 'string' then
            self.ptr = ffi.gc(C.CascadeClassifier_ctor_filename(f), Classes.Algorithm_dtor)
        else
            self.ptr = ffi.gc(C.CascadeClassifier_ctor_file(f), Classes.Algorithm_dtor)
        end
    end

    function CascadeClassifier:setMaxObjectSize(t)
        local argRules = {
            {"val", required = true, operator = cv.Size}
        }
        C.CascadeClassifier_setMaxObjectSize(self.ptr, cv.argcheck(t, argRules))
    end

    function CascadeClassifier:getMaxObjectSize()
        return C.CascadeClassifier_getMaxObjectSize(self.ptr)
    end

    function CascadeClassifier:setMinObjectSize(t)
        local argRules = {
            {"val", required = true, operator = cv.Size}
        }
        C.CascadeClassifier_setMinObjectSize(self.ptr, cv.argcheck(t, argRules))
    end

    function CascadeClassifier:getMinObjectSize()
        return C.CascadeClassifier_getMinObjectSize(self.ptr)
    end

    function CascadeClassifier:setScaleFactor(t)
        local argRules = {
            {"val", required = true}
        }
        C.CascadeClassifier_setScaleFactor(self.ptr, cv.argcheck(t, argRules))
    end

    function CascadeClassifier:getScaleFactor()
        return C.CascadeClassifier_getScaleFactor(self.ptr)
    end

    function CascadeClassifier:setMinNeighbors(t)
        local argRules = {
            {"val", required = true}
        }
        C.CascadeClassifier_setMinNeighbors(self.ptr, cv.argcheck(t, argRules))
    end

    function CascadeClassifier:getMinNeighbors()
        return C.CascadeClassifier_getMinNeighbors(self.ptr)
    end

    function CascadeClassifier:setFindLargestObject(t)
        local argRules = {
            {"val", required = true}
        }
        C.CascadeClassifier_setFindLargestObject(self.ptr, cv.argcheck(t, argRules))
    end

    function CascadeClassifier:getFindLargestObject()
        return C.CascadeClassifier_getFindLargestObject(self.ptr)
    end

    function CascadeClassifier:setMaxNumObjects(t)
        local argRules = {
            {"val", required = true}
        }
        C.CascadeClassifier_setMaxNumObjects(self.ptr, cv.argcheck(t, argRules))
    end

    function CascadeClassifier:getMaxNumObjects()
        return C.CascadeClassifier_getMaxNumObjects(self.ptr)
    end

    function CascadeClassifier:getClassifierSize()
        return C.CascadeClassifier_getClassifierSize(self.ptr)
    end

    function CascadeClassifier:detectMultiScale(t)
        local argRules = {
            {"image", required = true, operator = cv.wrap_tensor},
            {"objects", default = nil, operator = cv.wrap_tensor}
        }
        return cv.unwrap_tensors(C.CascadeClassifier_detectMultiScale(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end

    function CascadeClassifier:convert(t)
        local argRules = {
            {"gpu_objects", required = true, operator = cv.wrap_tensor}
        }
        return C.CascadeClassifier_convert(self.ptr, cv.argcheck(t, argRules))
    end
end

return cv.cuda
