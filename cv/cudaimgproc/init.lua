local cv = require 'cv._env'
require 'cutorch'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or require 'cv._env_cuda'

local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper cvtColor(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst, int code, int dstCn);

struct TensorWrapper demosaicing(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst, int code, int dcn);

void swapChannels(
        struct cutorchInfo info, struct TensorWrapper image,
        struct TensorWrapper dstOrder);

struct TensorWrapper gammaCorrection(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst, bool forward);

struct TensorWrapper alphaComp(struct cutorchInfo info,
        struct TensorWrapper img1, struct TensorWrapper img2,
        struct TensorWrapper dst, int alpha_op);

struct TensorWrapper calcHist(
        struct cutorchInfo info, struct TensorWrapper src, struct TensorWrapper hist);

struct TensorWrapper equalizeHist(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst);

struct TensorWrapper evenLevels(struct cutorchInfo info,
        struct TensorWrapper levels, int nLevels, int lowerLevel, int upperLevel);

struct TensorWrapper histEven(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper hist,
        int histSize, int lowerLevel, int upperLevel);

struct TensorArray histEven_4(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorArray hist, struct TensorWrapper histSize,
        struct TensorWrapper lowerLevel, struct TensorWrapper upperLevel);

struct TensorWrapper histRange(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper hist,
        struct TensorWrapper levels);

struct TensorArray histRange_4(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorArray hist, struct TensorWrapper levels);

struct PtrWrapper createHarrisCorner(
        int srcType, int blockSize, int ksize, double k, int borderType);

struct PtrWrapper createMinEigenValCorner(
        int srcType, int blockSize, int ksize, int borderType);

struct TensorWrapper CornernessCriteria_compute(
        struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper src, struct TensorWrapper dst);

struct PtrWrapper createGoodFeaturesToTrackDetector(
        int srcType, int maxCorners, double qualityLevel, double minDistance,
        int blockSize, bool useHarrisDetector, double harrisK);

struct TensorWrapper CornersDetector_detect(
        struct cutorchInfo info, struct PtrWrapper ptr, struct TensorWrapper image,
        struct TensorWrapper corners, struct TensorWrapper mask);

struct PtrWrapper createTemplateMatching(
        int srcType, int method, struct SizeWrapper user_block_size);

struct TensorWrapper TemplateMatching_match(
        struct cutorchInfo info, struct PtrWrapper ptr, struct TensorWrapper image,
        struct TensorWrapper templ, struct TensorWrapper result);

struct TensorWrapper bilateralFilter(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst, int kernel_size,
        float sigma_color, float sigma_spatial, int borderMode);

struct TensorWrapper blendLinear(struct cutorchInfo info,
        struct TensorWrapper img1, struct TensorWrapper img2, struct TensorWrapper weights1, 
        struct TensorWrapper weights2, struct TensorWrapper result);
]]

local C = ffi.load(cv.libPath('cudaimgproc'))

require 'cv.Classes'
local Classes = ffi.load(cv.libPath('Classes'))

function cv.cuda.cvtColor(t)
    local argRules = {
        {"src", required = true, operator = cv.wrap_tensor},
        {"dst", default = nil, operator = cv.wrap_tensor},
        {"code", required = true},
        {"dcn", default = 0},
    }
    return cv.unwrap_tensors(C.cvtColor(cv.cuda._info(), cv.argcheck(t, argRules)))
end

function cv.cuda.demosaicing(t)
    local argRules = {
        {"src", required = true, operator = cv.wrap_tensor},
        {"dst", default = nil, operator = cv.wrap_tensor},
        {"code", required = true},
        {"dcn", default = -1},
    }
    return cv.unwrap_tensors(C.demosaicing(cv.cuda._info(), cv.argcheck(t, argRules)))
end

function cv.cuda.swapChannels(t)
    local argRules = {
        {"image", required = true, operator = cv.wrap_tensor},
        {"dstOrder", operator = cv.Vec4i},
    }
    C.swapChannels(cv.cuda._info(), cv.argcheck(t, argRules))
end

function cv.cuda.gammaCorrection(t)
    local argRules = {
        {"src", required = true, operator = cv.wrap_tensor},
        {"dst", default = nil, operator = cv.wrap_tensor},
        {"forward", default = true}
    }
    return cv.unwrap_tensors(C.gammaCorrection(cv.cuda._info(), cv.argcheck(t, argRules)))
end

function cv.cuda.alphaComp(t)
    local argRules = {
        {"img1", required = true, operator = cv.wrap_tensor},
        {"img2", required = true, operator = cv.wrap_tensor},
        {"dst", default = nil, operator = cv.wrap_tensor},
        {"alpha_op", required = true}
    }
    return cv.unwrap_tensors(C.alphaComp(cv.cuda._info(), cv.argcheck(t, argRules)))
end

function cv.cuda.calcHist(t)
    local argRules = {
        {"src", required = true, operator = cv.wrap_tensor},
        {"hist", default = nil, operator = cv.wrap_tensor}
    }
    return cv.unwrap_tensors(C.calcHist(cv.cuda._info(), cv.argcheck(t, argRules)))
end

function cv.cuda.equalizeHist(t)
    local argRules = {
        {"src", required = true, operator = cv.wrap_tensor},
        {"dst", default = nil, operator = cv.wrap_tensor}
    }
    return cv.unwrap_tensors(C.equalizeHist(cv.cuda._info(), cv.argcheck(t, argRules)))
end

function cv.cuda.evenLevels(t)
    local argRules = {
        {"levels", default = nil},
        {"nLevels", required = true},
        {"lowerLevel", required = true},
        {"upperLevel", required = true}
    }
    return cv.unwrap_tensors(C.evenLevels(cv.cuda._info(), cv.argcheck(t, argRules)))
end

do
    local CornernessCriteria = torch.class('cuda.CornernessCriteria', 'cv.Algorithm', cv.cuda)

    function CornernessCriteria:compute(t)
        local argRules = {
            {"src", required = true, operator = cv.wrap_tensor},
            {"dst", default = nil, operator = cv.wrap_tensor}
        }
        return cv.unwrap_tensors(C.CornernessCriteria_compute(cv.cuda._info(), 
            self.ptr, cv.argcheck(t,argRules)))
    end
end

function cv.cuda.createHarrisCorner(t)
    local argRules = {
        {"srcType", required = true},
        {"blockSize", required = true},
        {"ksize", required = true},
        {"k", required = true},
        {"borderType", default = cv.BORDER_REFLECT101},
    }
    local retval = torch.factory('cuda.CornernessCriteria')()
    retval.ptr = ffi.gc(
        C.createHarrisCorner(cv.argcheck(t, argRules)),
        Classes.Algorithm_dtor)
    return retval
end

function cv.cuda.createMinEigenValCorner(t)
    local argRules = {
        {"srcType", required = true},
        {"blockSize", required = true},
        {"ksize", required = true},
        {"borderType", default = cv.BORDER_REFLECT101},
    }
    local retval = torch.factory('cuda.CornernessCriteria')()
    retval.ptr = ffi.gc(
        C.createMinEigenValCorner(cv.argcheck(t, argRules)),
        Classes.Algorithm_dtor)
    return retval
end

do
    local CornersDetector = torch.class('cuda.CornersDetector', 'cv.Algorithm', cv.cuda)

    function CornersDetector:detect(t)
        local argRules = {
            {"image", required = true, operator = cv.wrap_tensor},
            {"corners", default = nil, operator = cv.wrap_tensor},
            {"mask", default = nil, operator = cv.wrap_tensor}
        }
        return cv.unwrap_tensors(C.CornersDetector_detect(cv.cuda._info(), 
            self.ptr, cv.argcheck(t,argRules)))
    end
end

function cv.cuda.createGoodFeaturesToTrackDetector(t)
    local argRules = {
        {"srcType", required = true},
        {"maxCorners", default = 1000},
        {"qualityLevel", default = 0.01},
        {"minDistance", default = 0.0},
        {"blockSize", default = 3},
        {"useHarrisDetector", default = false},
        {"harrisK", default = 0.04}
    }
    local retval = torch.factory('cuda.CornersDetector')()
    retval.ptr = ffi.gc(
        C.createGoodFeaturesToTrackDetector(cv.argcheck(t, argRules)), 
        Classes.Algorithm_dtor)
    return retval
end

do
    local TemplateMatching = torch.class('cuda.TemplateMatching', 'cv.Algorithm', cv.cuda)

    function TemplateMatching:match(t)
        local argRules = {
            {"image", required = true, operator = cv.wrap_tensor},
            {"templ", required = true, operator = cv.wrap_tensor},
            {"result", default = nil, operator = cv.wrap_tensor}
        }
        return cv.unwrap_tensors(C.TemplateMatching_match(cv.cuda._info(),
            self.ptr, cv.argcheck(t, argRules)))
    end
end

function cv.cuda.createTemplateMatching(t)
    local argRules = {
        {"srcType", required = true},
        {"method", required = true},
        {"user_block_size", default = {0, 0}, operator = cv.Size}
    }
    local retval = torch.factory('cuda.TemplateMatching')()
    retval.ptr = ffi.gc(
        C.createTemplateMatching(cv.argcheck(t, argRules)), 
        Classes.Algorithm_dtor)
    return retval
end

function cv.cuda.bilateralFilter(t)
    local argRules = {
        {"src", required = true, operator = cv.wrap_tensor},
        {"dst", default = nil, operator = cv.wrap_tensor},
        {"kernel_size", required = true},
        {"sigma_color", required = true},
        {"sigma_spatial", required = true},
        {"kernel_size", required = true},
        {"borderMode", default = cv.BORDER_DEFAULT}
    }
    return cv.unwrap_tensors(C.bilateralFilter(cv.cuda._info(), cv.argcheck(t, argRules)))
end

function cv.cuda.blendLinear(t)
    local argRules = {
        {"img1", required = true, operator = cv.wrap_tensor},
        {"img2", required = true, operator = cv.wrap_tensor},
        {"weights1", required = true, operator = cv.wrap_tensor},
        {"weights2", required = true, operator = cv.wrap_tensor},
        {"result", default = nil, operator = cv.wrap_tensor}
    }
    return cv.unwrap_tensors(C.blendLinear(cv.cuda._info(), cv.argcheck(t, argRules)))
end

return cv.cuda

