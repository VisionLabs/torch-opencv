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



return cv.cuda
