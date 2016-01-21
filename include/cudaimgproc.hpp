#include <CUDACommon.hpp>
#include <include/Classes.hpp>
#include <opencv2/cudaimgproc.hpp>

// CornernessCriteria

struct CornernessCriteriaPtr {
    void *ptr;
    inline cuda::CornernessCriteria * operator->() { return static_cast<cuda::CornernessCriteria *>(ptr); }
    inline CornernessCriteriaPtr(cuda::CornernessCriteria *ptr) { this->ptr = ptr; }
    inline cuda::CornernessCriteria & operator*() { return *static_cast<cuda::CornernessCriteria *>(this->ptr); }
};

// CornersDetector

struct CornersDetectorPtr {
    void *ptr;
    inline cuda::CornersDetector * operator->() { return static_cast<cuda::CornersDetector *>(ptr); }
    inline CornersDetectorPtr(cuda::CornersDetector *ptr) { this->ptr = ptr; }
    inline cuda::CornersDetector & operator*() { return *static_cast<cuda::CornersDetector *>(this->ptr); }
};

// TemplateMatching

struct TemplateMatchingPtr {
    void *ptr;
    inline cuda::TemplateMatching * operator->() { return static_cast<cuda::TemplateMatching *>(ptr); }
    inline TemplateMatchingPtr(cuda::TemplateMatching *ptr) { this->ptr = ptr; }
    inline cuda::TemplateMatching & operator*() { return *static_cast<cuda::TemplateMatching *>(this->ptr); }
};

extern "C"
struct TensorWrapper cvtColor(struct cutorchInfo info,
                                         struct TensorWrapper src, struct TensorWrapper dst, int code, int dstCn);

extern "C"
struct TensorWrapper demosaicing(struct cutorchInfo info,
                                 struct TensorWrapper src, struct TensorWrapper dst, int code, int dcn);

extern "C"
void swapChannels(
        struct cutorchInfo info, struct TensorWrapper image,
        struct TensorWrapper dstOrder);

extern "C"
struct TensorWrapper gammaCorrection(struct cutorchInfo info,
                                     struct TensorWrapper src, struct TensorWrapper dst, bool forward);

extern "C"
struct TensorWrapper alphaComp(struct cutorchInfo info,
                               struct TensorWrapper img1, struct TensorWrapper img2,
                               struct TensorWrapper dst, int alpha_op);

extern "C"
struct TensorWrapper calcHist(
        struct cutorchInfo info, struct TensorWrapper src, struct TensorWrapper hist);

extern "C"
struct TensorWrapper equalizeHist(struct cutorchInfo info,
                                  struct TensorWrapper src, struct TensorWrapper dst);

extern "C"
struct TensorWrapper evenLevels(struct cutorchInfo info,
                                struct TensorWrapper levels, int nLevels, int lowerLevel, int upperLevel);

extern "C"
struct TensorWrapper histEven(struct cutorchInfo info,
                              struct TensorWrapper src, struct TensorWrapper hist,
                              int histSize, int lowerLevel, int upperLevel);

extern "C"
struct TensorArray histEven_4(struct cutorchInfo info,
                              struct TensorWrapper src, struct TensorArray hist, struct TensorWrapper histSize,
                              struct TensorWrapper lowerLevel, struct TensorWrapper upperLevel);

extern "C"
struct TensorWrapper histRange(struct cutorchInfo info,
                               struct TensorWrapper src, struct TensorWrapper hist,
                               struct TensorWrapper levels);

extern "C"
struct TensorArray histRange_4(struct cutorchInfo info,
                               struct TensorWrapper src, struct TensorArray hist, struct TensorWrapper levels);

extern "C"
struct CornernessCriteriaPtr createHarrisCorner(
        int srcType, int blockSize, int ksize, double k, int borderType);

extern "C"
struct CornernessCriteriaPtr createMinEigenValCorner(
        int srcType, int blockSize, int ksize, int borderType);

extern "C"
struct TensorWrapper CornernessCriteria_compute(
        struct cutorchInfo info, struct CornernessCriteriaPtr ptr,
        struct TensorWrapper src, struct TensorWrapper dst);

extern "C"
struct CornersDetectorPtr createGoodFeaturesToTrackDetector(
        int srcType, int maxCorners, double qualityLevel, double minDistance,
        int blockSize, bool useHarrisDetector, double harrisK);

extern "C"
struct TensorWrapper CornersDetector_detect(
        struct cutorchInfo info, struct CornersDetectorPtr ptr, struct TensorWrapper image,
        struct TensorWrapper corners, struct TensorWrapper mask);

extern "C"
struct TemplateMatchingPtr createTemplateMatching(
        int srcType, int method, struct SizeWrapper user_block_size);

extern "C"
struct TensorWrapper TemplateMatching_match(
        struct cutorchInfo info, struct TemplateMatchingPtr ptr, struct TensorWrapper image,
        struct TensorWrapper templ, struct TensorWrapper result);

extern "C"
struct TensorWrapper bilateralFilter(struct cutorchInfo info,
                                     struct TensorWrapper src, struct TensorWrapper dst, int kernel_size,
                                     float sigma_color, float sigma_spatial, int borderMode);

extern "C"
struct TensorWrapper blendLinear(struct cutorchInfo info,
                                 struct TensorWrapper img1, struct TensorWrapper img2, struct TensorWrapper weights1,
                                 struct TensorWrapper weights2, struct TensorWrapper result);