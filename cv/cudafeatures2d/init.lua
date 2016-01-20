local cv = require 'cv._env'
require 'cutorch'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or require 'cv._env_cuda'

local ffi = require 'ffi'

ffi.cdef[[
struct PtrWrapper createBFMatcher(int normType);

bool DescriptorMatcher_isMaskSupported(struct PtrWrapper ptr);

void DescriptorMatcher_add(
        struct PtrWrapper ptr, struct TensorArray descriptors);

struct TensorArray DescriptorMatcher_getTrainDescriptors(
        struct cutorchInfo info, struct PtrWrapper ptr);

void DescriptorMatcher_clear(struct PtrWrapper ptr);

bool DescriptorMatcher_empty(struct PtrWrapper ptr);

void DescriptorMatcher_train(struct PtrWrapper ptr);

struct TensorWrapper DescriptorMatcher_match(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, struct TensorWrapper mask);

struct TensorWrapper DescriptorMatcher_match_masks(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper matches,
        struct TensorArray masks);

struct DMatchArray DescriptorMatcher_matchConvert(
         struct PtrWrapper ptr, struct TensorWrapper gpu_matches);

struct TensorWrapper DescriptorMatcher_knnMatch(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, int k, struct TensorWrapper mask);

struct TensorWrapper DescriptorMatcher_knnMatch_masks(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, int k, struct TensorArray masks);

struct DMatchArrayOfArrays DescriptorMatcher_knnMatchConvert(
        struct PtrWrapper ptr,
        struct TensorWrapper gpu_matches, bool compactResult);

struct TensorWrapper DescriptorMatcher_radiusMatch(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, float maxDistance, struct TensorWrapper mask);

struct TensorWrapper DescriptorMatcher_radiusMatch_masks(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, float maxDistance, struct TensorArray masks);

struct DMatchArrayOfArrays DescriptorMatcher_radiusMatchConvert(
        struct PtrWrapper ptr,
        struct TensorWrapper gpu_matches, bool compactResult);

void Feature2DAsync_dtor(struct Feature2DAsyncPtr ptr);

struct TensorWrapper Feature2DAsync_detectAsync(
        struct cutorchInfo info, struct Feature2DAsyncPtr ptr, struct TensorWrapper image,
        struct TensorWrapper keypoints, struct TensorWrapper mask);

struct TensorArray Feature2DAsync_computeAsync(
        struct cutorchInfo info, struct Feature2DAsyncPtr ptr, struct TensorWrapper image,
        struct TensorWrapper keypoints, struct TensorWrapper descriptors);

struct TensorArray Feature2DAsync_detectAndComputeAsync(
        struct cutorchInfo info, struct Feature2DAsyncPtr ptr, struct TensorWrapper image,
        struct TensorWrapper mask, struct TensorWrapper keypoints,
        struct TensorWrapper descriptors, bool useProvidedKeypoints);

struct KeyPointArray Feature2DAsync_convert(
        struct Feature2DAsyncPtr ptr, struct TensorWrapper gpu_keypoints);

struct FastFeatureDetectorPtr FasfFeatureDetector_ctor(
        int threshold, bool nonmaxSuppression, int type, int max_npoints);

void FasfFeatureDetector_dtor(struct FastFeatureDetectorPtr ptr);

void FastFeatureDetector_setMaxNumPoints(struct FastFeatureDetectorPtr ptr, int val);

int FastFeatureDetector_getMaxNumPoints(struct FastFeatureDetectorPtr ptr);

struct ORBPtr ORB_ctor(
        int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel, 
        int WTA_K, int scoreType, int patchSize, int fastThreshold, bool blurForDescriptor);

void FasfFeatureDetector_dtor(struct ORBPtr ptr);

void ORB_setBlurForDescriptor(struct ORBPtr ptr, bool val);

bool ORB_getBlurForDescriptor(struct ORBPtr ptr);
]]

local C = ffi.load(cv.libPath('cudafeatures2d'))

require 'cv.Classes'
local Classes = ffi.load(cv.libPath('Classes'))

do
    local DescriptorMatcher = torch.class('cuda.DescriptorMatcher', 'cv.Algorithm', cv.cuda)

    function DescriptorMatcher.createBFMatcher(t)
        local argRules = {
            {"normType", default = cv.NORM_L2}
        }
        local retval = torch.factory('cuda.DescriptorMatcher')()
        retval.ptr = ffi.gc(C.createBFMatcher(cv.argcheck(t, argRules)), Classes.Algorithm_dtor)
        return retval
    end

    function DescriptorMatcher:isMaskSupported()
        return C.DescriptorMatcher_isMaskSupported(self.ptr)
    end

    function DescriptorMatcher:add(t)
        local argRules = {
            {"descriptors", required = true}
        }
        C.DescriptorMatcher_add(self.ptr, cv.wrap_tensors(descriptors))
    end

    function DescriptorMatcher:getTrainDescriptors()
        return cv.unwrap_tensors(C.DescriptorMatcher_getTrainDescriptors(
            cv.cuda._info(), self.ptr) , true)
    end

    function DescriptorMatcher:clear()
        C.DescriptorMatcher_clear(self.ptr)
    end

    function DescriptorMatcher:empty()
        return C.DescriptorMatcher_empty(self.ptr)
    end

    function DescriptorMatcher:train()
        C.DescriptorMatcher_train(self.ptr)
    end

    function DescriptorMatcher:match(t)
        local argRules = {
            {"queryDescriptors", required = true, operator = cv.wrap_tensor},
            {"trainDescriptors", default = nil, operator = cv.wrap_tensor},
            {"matches", default = nil, operator = cv.wrap_tensor},
            {"mask", default = nil, operator = cv.wrap_tensor},
            {"masks", default = nil, operator = cv.wrap_tensors}
        }

        assert(not (t.mask and t.masks) and not (t[4] and t[5]))

        if t.masks or t[5] then
            return cv.unwrap_tensors(C.DescriptorMatcher_match_masks(
                cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
        else
            return cv.unwrap_tensors(C.DescriptorMatcher_match(
                cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
        end
    end

    function DescriptorMatcher:matchConvert(t)
        local argRules = {
            {"gpu_matches", required = true, operator = cv.wrap_tensor}
        }
        return cv.gcarray(C.DescriptorMatcher_matchConvert(self.ptr, cv.argcheck(t, argRules)))
    end

    function DescriptorMatcher:knnMatch(t)
        local argRules = {
            {"queryDescriptors", required = true, operator = cv.wrap_tensor},
            {"trainDescriptors", default = nil, operator = cv.wrap_tensor},
            {"matches", default = nil, operator = cv.wrap_tensor},
            {"k", required = true},
            {"mask", default = nil, operator = cv.wrap_tensor},
            {"masks", default = nil, operator = cv.wrap_tensors}
        }

        assert(not (t.mask and t.masks) and not (t[5] and t[6]))

        if t.masks or t[6] then
            return cv.unwrap_tensors(C.DescriptorMatcher_knnMatch_masks(
                cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
        else
            return cv.unwrap_tensors(C.DescriptorMatcher_knnMatch(
                cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
        end
    end

    function DescriptorMatcher:knnMatchConvert(t)
        local argRules = {
            {"gpu_matches", required = true, operator = cv.wrap_tensor},
            {"compactResult", default = false}
        }

        local result = 
            cv.gcarray(C.DescriptorMatcher_knnMatchConvert(self.ptr, cv.argcheck(t, argRules)))

        local retval = {}
        for i = 0, result.size-1 do
            retval[i+1] = cv.gcarray(result[i])
        end

        return retval
    end

    function DescriptorMatcher:radiusMatch(t)
        local argRules = {
            {"queryDescriptors", required = true, operator = cv.wrap_tensor},
            {"trainDescriptors", default = nil, operator = cv.wrap_tensor},
            {"matches", default = nil, operator = cv.wrap_tensor},
            {"maxDistance", required = true},
            {"mask", default = nil, operator = cv.wrap_tensor},
            {"masks", default = nil, operator = cv.wrap_tensors}
        }

        assert(not (t.mask and t.masks) and not (t[5] and t[6]))

        if t.masks or t[6] then
            return cv.unwrap_tensors(C.DescriptorMatcher_radiusMatch_masks(
                cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
        else
            return cv.unwrap_tensors(C.DescriptorMatcher_radiusMatch(
                cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
        end
    end

    function DescriptorMatcher:radiusMatchConvert(t)
        local argRules = {
            {"gpu_matches", required = true, operator = cv.wrap_tensor},
            {"compactResult", default = false}
        }

        local result = 
            cv.gcarray(C.DescriptorMatcher_radiusMatchConvert(self.ptr, cv.argcheck(t, argRules)))

        local retval = {}
        for i = 0, result.size-1 do
            retval[i+1] = cv.gcarray(result[i])
        end

        return retval
    end
end

return cv.cuda
