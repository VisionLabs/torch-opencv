#include <CUDACommon.hpp>
#include <include/Classes.hpp>
#include <opencv2/cudafeatures2d.hpp>

// DescriptorMatcher

struct DescriptorMatcherPtr {
    void *ptr;
    inline cuda::DescriptorMatcher * operator->() { return static_cast<cuda::DescriptorMatcher *>(ptr); }
    inline DescriptorMatcherPtr(cuda::DescriptorMatcher *ptr) { this->ptr = ptr; }
    inline cuda::DescriptorMatcher & operator*() { return *static_cast<cuda::DescriptorMatcher *>(this->ptr); }
};

// Feature2DAsync

struct Feature2DAsyncPtr {
    void *ptr;
    inline cuda::Feature2DAsync * operator->() { return static_cast<cuda::Feature2DAsync *>(ptr); }
    inline Feature2DAsyncPtr(cuda::Feature2DAsync *ptr) { this->ptr = ptr; }
    inline cuda::Feature2DAsync & operator*() { return *static_cast<cuda::Feature2DAsync *>(this->ptr); }
};

// FastFeatureDetector

struct FastFeatureDetectorPtr {
    void *ptr;
    inline cuda::FastFeatureDetector * operator->() { return static_cast<cuda::FastFeatureDetector *>(ptr); }
    inline FastFeatureDetectorPtr(cuda::FastFeatureDetector *ptr) { this->ptr = ptr; }
    inline cuda::FastFeatureDetector & operator*() { return *static_cast<cuda::FastFeatureDetector *>(this->ptr); }
};

// ORB

struct ORBPtr {
    void *ptr;
    inline cuda::ORB * operator->() { return static_cast<cuda::ORB *>(ptr); }
    inline ORBPtr(cuda::ORB *ptr) { this->ptr = ptr; }
    inline cuda::ORB & operator*() { return *static_cast<cuda::ORB *>(this->ptr); }
};

extern "C"
struct DescriptorMatcherPtr createBFMatcherCuda(int normType);

extern "C"
bool DescriptorMatcher_isMaskSupportedCuda(struct DescriptorMatcherPtr ptr);

extern "C"
void DescriptorMatcher_addCuda(
        struct DescriptorMatcherPtr ptr, struct TensorArray descriptors);

extern "C"
struct TensorArray DescriptorMatcher_getTrainDescriptorsCuda(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr);

extern "C"
void DescriptorMatcher_clearCuda(struct DescriptorMatcherPtr ptr);

extern "C"
bool DescriptorMatcher_emptyCuda(struct DescriptorMatcherPtr ptr);

extern "C"
void DescriptorMatcher_trainCuda(struct DescriptorMatcherPtr ptr);

extern "C"
struct TensorWrapper DescriptorMatcher_matchCuda(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, struct TensorWrapper mask);

extern "C"
struct TensorWrapper DescriptorMatcher_match_masksCuda(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper matches,
        struct TensorArray masks);

extern "C"
struct DMatchArray DescriptorMatcher_matchConvertCuda(
        struct DescriptorMatcherPtr ptr, struct TensorWrapper gpu_matches);

extern "C"
struct TensorWrapper DescriptorMatcher_knnMatchCuda(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, int k, struct TensorWrapper mask);

extern "C"
struct TensorWrapper DescriptorMatcher_knnMatch_masksCuda(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, int k, struct TensorArray masks);

extern "C"
struct DMatchArrayOfArrays DescriptorMatcher_knnMatchConvertCuda(
        struct DescriptorMatcherPtr ptr,
        struct TensorWrapper gpu_matches, bool compactResult);

extern "C"
struct TensorWrapper DescriptorMatcher_radiusMatchCuda(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, float maxDistance, struct TensorWrapper mask);

extern "C"
struct TensorWrapper DescriptorMatcher_radiusMatch_masksCuda(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, float maxDistance, struct TensorArray masks);

extern "C"
struct DMatchArrayOfArrays DescriptorMatcher_radiusMatchConvertCuda(
        struct DescriptorMatcherPtr ptr,
        struct TensorWrapper gpu_matches, bool compactResult);

extern "C"
void Feature2DAsync_dtorCuda(struct Feature2DAsyncPtr ptr);

extern "C"
struct TensorWrapper Feature2DAsync_detectAsyncCuda(
        struct cutorchInfo info, struct Feature2DAsyncPtr ptr, struct TensorWrapper image,
        struct TensorWrapper keypoints, struct TensorWrapper mask);

extern "C"
struct TensorArray Feature2DAsync_computeAsyncCuda(
        struct cutorchInfo info, struct Feature2DAsyncPtr ptr, struct TensorWrapper image,
        struct TensorWrapper keypoints, struct TensorWrapper descriptors);

extern "C"
struct TensorArray Feature2DAsync_detectAndComputeAsyncCuda(
        struct cutorchInfo info, struct Feature2DAsyncPtr ptr, struct TensorWrapper image,
        struct TensorWrapper mask, struct TensorWrapper keypoints,
        struct TensorWrapper descriptors, bool useProvidedKeypoints);

extern "C"
struct KeyPointArray Feature2DAsync_convertCuda(
        struct Feature2DAsyncPtr ptr, struct TensorWrapper gpu_keypoints);

extern "C"
struct FastFeatureDetectorPtr FastFeatureDetector_ctorCuda(
        int threshold, bool nonmaxSuppression, int type, int max_npoints);

extern "C"
void FastFeatureDetector_dtorCuda(struct FastFeatureDetectorPtr ptr);

extern "C"
void FastFeatureDetector_setMaxNumPointsCuda(struct FastFeatureDetectorPtr ptr, int val);

extern "C"
int FastFeatureDetector_getMaxNumPointsCuda(struct FastFeatureDetectorPtr ptr);

extern "C"
struct ORBPtr ORB_ctorCuda(
        int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel,
        int WTA_K, int scoreType, int patchSize, int fastThreshold, bool blurForDescriptor);

extern "C"
void ORB_setBlurForDescriptorCuda(struct ORBPtr ptr, bool val);

extern "C"
bool ORB_getBlurForDescriptorCuda(struct ORBPtr ptr);
