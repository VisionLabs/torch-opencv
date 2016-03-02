local cv = require 'cv._env'
require 'cutorch'
require 'cv.features2d'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or require 'cv._env_cuda'

local ffi = require 'ffi'

ffi.cdef[[
struct PtrWrapper createBFMatcherCuda(int normType);

bool DescriptorMatcher_isMaskSupportedCuda(struct PtrWrapper ptr);

void DescriptorMatcher_addCuda(
        struct PtrWrapper ptr, struct TensorArray descriptors);

struct TensorArray DescriptorMatcher_getTrainDescriptorsCuda(
        struct cutorchInfo info, struct PtrWrapper ptr);

void DescriptorMatcher_clearCuda(struct PtrWrapper ptr);

bool DescriptorMatcher_emptyCuda(struct PtrWrapper ptr);

void DescriptorMatcher_trainCuda(struct PtrWrapper ptr);

struct TensorWrapper DescriptorMatcher_matchCuda(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, struct TensorWrapper mask);

struct TensorWrapper DescriptorMatcher_match_masksCuda(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper matches,
        struct TensorArray masks);

struct DMatchArray DescriptorMatcher_matchConvertCuda(
         struct PtrWrapper ptr, struct TensorWrapper gpu_matches);

struct TensorWrapper DescriptorMatcher_knnMatchCuda(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, int k, struct TensorWrapper mask);

struct TensorWrapper DescriptorMatcher_knnMatch_masksCuda(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, int k, struct TensorArray masks);

struct DMatchArrayOfArrays DescriptorMatcher_knnMatchConvertCuda(
        struct PtrWrapper ptr,
        struct TensorWrapper gpu_matches, bool compactResult);

struct TensorWrapper DescriptorMatcher_radiusMatchCuda(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, float maxDistance, struct TensorWrapper mask);

struct TensorWrapper DescriptorMatcher_radiusMatch_masksCuda(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, float maxDistance, struct TensorArray masks);

struct DMatchArrayOfArrays DescriptorMatcher_radiusMatchConvertCuda(
        struct PtrWrapper ptr,
        struct TensorWrapper gpu_matches, bool compactResult);

void Feature2DAsync_dtorCuda(struct PtrWrapper ptr);

struct TensorWrapper Feature2DAsync_detectAsyncCuda(
        struct cutorchInfo info, struct PtrWrapper ptr, struct TensorWrapper image,
        struct TensorWrapper keypoints, struct TensorWrapper mask);

struct TensorArray Feature2DAsync_computeAsyncCuda(
        struct cutorchInfo info, struct PtrWrapper ptr, struct TensorWrapper image,
        struct TensorWrapper keypoints, struct TensorWrapper descriptors);

struct TensorArray Feature2DAsync_detectAndComputeAsyncCuda(
        struct cutorchInfo info, struct PtrWrapper ptr, struct TensorWrapper image,
        struct TensorWrapper mask, struct TensorWrapper keypoints,
        struct TensorWrapper descriptors, bool useProvidedKeypoints);

struct KeyPointArray Feature2DAsync_convertCuda(
        struct PtrWrapper ptr, struct TensorWrapper gpu_keypoints);

struct PtrWrapper FastFeatureDetector_ctorCuda(
        int threshold, bool nonmaxSuppression, int type, int max_npoints);

void FastFeatureDetector_dtorCuda(struct PtrWrapper ptr);

void FastFeatureDetector_setMaxNumPointsCuda(struct PtrWrapper ptr, int val);

int FastFeatureDetector_getMaxNumPointsCuda(struct PtrWrapper ptr);

struct PtrWrapper ORB_ctorCuda(
        int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel, 
        int WTA_K, int scoreType, int patchSize, int fastThreshold, bool blurForDescriptor);

void ORB_setBlurForDescriptorCuda(struct PtrWrapper ptr, bool val);

bool ORB_getBlurForDescriptorCuda(struct PtrWrapper ptr);
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
        retval.ptr = ffi.gc(C.createBFMatcherCuda(cv.argcheck(t, argRules)), Classes.Algorithm_dtor)
        return retval
    end

    function DescriptorMatcher:isMaskSupported()
        return C.DescriptorMatcher_isMaskSupportedCuda(self.ptr)
    end

    function DescriptorMatcher:add(t)
        local argRules = {
            {"descriptors", required = true}
        }
        C.DescriptorMatcher_addCuda(self.ptr, cv.wrap_tensors(descriptors))
    end

    function DescriptorMatcher:getTrainDescriptors()
        return cv.unwrap_tensors(C.DescriptorMatcher_getTrainDescriptorsCuda(
            cv.cuda._info(), self.ptr) , true)
    end

    function DescriptorMatcher:clear()
        C.DescriptorMatcher_clearCuda(self.ptr)
    end

    function DescriptorMatcher:empty()
        return C.DescriptorMatcher_emptyCuda(self.ptr)
    end

    function DescriptorMatcher:train()
        C.DescriptorMatcher_trainCuda(self.ptr)
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
            return cv.unwrap_tensors(C.DescriptorMatcher_match_masksCuda(
                cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
        else
            return cv.unwrap_tensors(C.DescriptorMatcher_matchCuda(
                cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
        end
    end

    function DescriptorMatcher:matchConvert(t)
        local argRules = {
            {"gpu_matches", required = true, operator = cv.wrap_tensor}
        }
        return cv.gcarray(C.DescriptorMatcher_matchConvertCuda(self.ptr, cv.argcheck(t, argRules)))
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
            return cv.unwrap_tensors(C.DescriptorMatcher_knnMatch_masksCuda(
                cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
        else
            return cv.unwrap_tensors(C.DescriptorMatcher_knnMatchCuda(
                cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
        end
    end

    function DescriptorMatcher:knnMatchConvert(t)
        local argRules = {
            {"gpu_matches", required = true, operator = cv.wrap_tensor},
            {"compactResult", default = false}
        }

        local result = 
            cv.gcarray(C.DescriptorMatcher_knnMatchConvertCuda(self.ptr, cv.argcheck(t, argRules)))

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
            return cv.unwrap_tensors(C.DescriptorMatcher_radiusMatch_masksCuda(
                cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
        else
            return cv.unwrap_tensors(C.DescriptorMatcher_radiusMatchCuda(
                cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
        end
    end

    function DescriptorMatcher:radiusMatchConvert(t)
        local argRules = {
            {"gpu_matches", required = true, operator = cv.wrap_tensor},
            {"compactResult", default = false}
        }

        local result = 
            cv.gcarray(C.DescriptorMatcher_radiusMatchConvertCuda(self.ptr, cv.argcheck(t, argRules)))

        local retval = {}
        for i = 0, result.size-1 do
            retval[i+1] = cv.gcarray(result[i])
        end

        return retval
    end
end

do
    local Feature2DAsync = torch.class('cuda.Feature2DAsync', 'cv.Feature2D', cv.cuda)

    function Feature2DAsync:detect(t)
        local argRules = {
            {"image", required = true, operator = cv.wrap_tensor},
            {"keypoints", default = nil, operator = cv.wrap_tensor},
            {"mask", default = nil, operator = cv.wrap_tensor}
        }
        return cv.unwrap_tensors(C.Feature2DAsync_detectAsyncCuda(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end

    function Feature2DAsync:compute(t)
        local argRules = {
            {"image", required = true, operator = cv.wrap_tensor},
            {"keypoints", default = nil, operator = cv.wrap_tensor},
            {"descriptors", default = nil, operator = cv.wrap_tensor}
        }
        return cv.unwrap_tensors(C.Feature2DAsync_computeAsyncCuda(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end

    function Feature2DAsync:detectAndCompute(t)
        local argRules = {
            {"image", required = true, operator = cv.wrap_tensor},
            {"mask", default = nil, operator = cv.wrap_tensor},
            {"keypoints", default = nil, operator = cv.wrap_tensor},
            {"descriptors", default = nil, operator = cv.wrap_tensor},
            {"useProvidedKeypoints", default = false}
        }
        return cv.unwrap_tensors(C.Feature2DAsync_detectAndComputeAsyncCuda(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end

    function Feature2DAsync:convert(t)
        local argRules = {
            {"gpu_keypoints", required = true, operator = cv.wrap_tensor}
        }
        return cv.gcarray(C.Feature2DAsync_convertCuda(self.ptr, cv.argcheck(t, argRules)))
    end
end

do
    local FastFeatureDetector = torch.class(
        'cuda.FastFeatureDetector', 'cuda.Feature2DAsync', cv.cuda)

    function FastFeatureDetector:__init(t)
        local argRules = {
            {"threshold", default = 10},
            {"nonmaxSuppression", default = true},
            {"type", default = cv.FastFeatureDetector_TYPE_9_16},
            {"max_npoints", default = 5000}
        }
        self.ptr = ffi.gc(C.FastFeatureDetector_ctorCuda(
            cv.argcheck(t, argRules), C.Feature2DAsync_dtorCuda))
    end

    function FastFeatureDetector:setMaxNumPoints(t)
        local argRules = {
            {"val", required = true}
        }
        C.FastFeatureDetector_setMaxNumPointsCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function FastFeatureDetector:getMaxNumPoints()
        return C.FastFeatureDetector_getMaxNumPointsCuda(self.ptr)
    end
end

do
    local ORB = torch.class('cuda.ORB', 'cuda.Feature2DAsync', cv.cuda)

    function ORB:__init(t)
        local argRules = {
            {"nfeatures", default = 500},
            {"scaleFactor", default = 1.2},
            {"nlevels", default = 8},
            {"edgeThreshold", default = 31},
            {"firstLevel", default = 0},
            {"WTA_K", default = 2},
            {"scoreType", default = cv.ORB_HARRIS_SCORE},
            {"patchSize", default = 31},
            {"fastThreshold", default = 20},
            {"blurForDescriptor", default = false}
        }
        self.ptr = ffi.gc(C.ORB_ctorCuda(
            cv.argcheck(t, argRules), C.Feature2DAsync_dtorCuda))
    end

    function ORB:setsetBlurForDescriptorMaxNumPoints(t)
        local argRules = {
            {"val", required = true}
        }
        C.ORB_setBlurForDescriptorCuda(self.ptr, cv.argcheck(t, argRules))
    end

    function ORB:getBlurForDescriptor()
        return C.ORB_getBlurForDescriptorCuda(self.ptr)
    end
end

return cv.cuda
