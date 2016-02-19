local cv = require 'cv._env'
require 'cutorch'
require 'cv.features2d'

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

void Feature2DAsync_dtor(struct PtrWrapper ptr);

struct TensorWrapper Feature2DAsync_detectAsync(
        struct cutorchInfo info, struct PtrWrapper ptr, struct TensorWrapper image,
        struct TensorWrapper keypoints, struct TensorWrapper mask);

struct TensorArray Feature2DAsync_computeAsync(
        struct cutorchInfo info, struct PtrWrapper ptr, struct TensorWrapper image,
        struct TensorWrapper keypoints, struct TensorWrapper descriptors);

struct TensorArray Feature2DAsync_detectAndComputeAsync(
        struct cutorchInfo info, struct PtrWrapper ptr, struct TensorWrapper image,
        struct TensorWrapper mask, struct TensorWrapper keypoints,
        struct TensorWrapper descriptors, bool useProvidedKeypoints);

struct KeyPointArray Feature2DAsync_convert(
        struct PtrWrapper ptr, struct TensorWrapper gpu_keypoints);

struct PtrWrapper FastFeatureDetector_ctor(
        int threshold, bool nonmaxSuppression, int type, int max_npoints);

void FastFeatureDetector_dtor(struct PtrWrapper ptr);

void FastFeatureDetector_setMaxNumPoints(struct PtrWrapper ptr, int val);

int FastFeatureDetector_getMaxNumPoints(struct PtrWrapper ptr);

struct PtrWrapper ORB_ctor(
        int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel, 
        int WTA_K, int scoreType, int patchSize, int fastThreshold, bool blurForDescriptor);

void ORB_setBlurForDescriptor(struct PtrWrapper ptr, bool val);

bool ORB_getBlurForDescriptor(struct PtrWrapper ptr);
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

do
    local Feature2DAsync = torch.class('cuda.Feature2DAsync', 'cv.Feature2D', cv.cuda)

    function Feature2DAsync:detect(t)
        local argRules = {
            {"image", required = true, operator = cv.wrap_tensor},
            {"keypoints", default = nil, operator = cv.wrap_tensor},
            {"mask", default = nil, operator = cv.wrap_tensor}
        }
        return cv.unwrap_tensors(C.Feature2DAsync_detectAsync(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end

    function Feature2DAsync:compute(t)
        local argRules = {
            {"image", required = true, operator = cv.wrap_tensor},
            {"keypoints", default = nil, operator = cv.wrap_tensor},
            {"descriptors", default = nil, operator = cv.wrap_tensor}
        }
        return cv.unwrap_tensors(C.Feature2DAsync_computeAsync(
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
        return cv.unwrap_tensors(C.Feature2DAsync_detectAndComputeAsync(
            cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
    end

    function Feature2DAsync:convert(t)
        local argRules = {
            {"gpu_keypoints", required = true, operator = cv.wrap_tensor}
        }
        return cv.gcarray(C.Feature2DAsync_convert(self.ptr, cv.argcheck(t, argRules)))
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
        self.ptr = ffi.gc(C.FastFeatureDetector_ctor(
            cv.argcheck(t, argRules), C.Feature2DAsync_dtor))
    end

    function FastFeatureDetector:setMaxNumPoints(t)
        local argRules = {
            {"val", required = true}
        }
        C.FastFeatureDetector_setMaxNumPoints(self.ptr, cv.argcheck(t, argRules))
    end

    function FastFeatureDetector:getMaxNumPoints()
        return C.FastFeatureDetector_getMaxNumPoints(self.ptr)
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
        self.ptr = ffi.gc(C.ORB_ctor(
            cv.argcheck(t, argRules), C.Feature2DAsync_dtor))
    end

    function ORB:setsetBlurForDescriptorMaxNumPoints(t)
        local argRules = {
            {"val", required = true}
        }
        C.ORB_setBlurForDescriptor(self.ptr, cv.argcheck(t, argRules))
    end

    function ORB:getBlurForDescriptor()
        return C.ORB_getBlurForDescriptor(self.ptr)
    end
end

return cv.cuda
