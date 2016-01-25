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
struct DescriptorMatcherPtr createBFMatcher(int normType);

extern "C"
bool DescriptorMatcher_isMaskSupported(struct DescriptorMatcherPtr ptr);

extern "C"
void DescriptorMatcher_add(
        struct DescriptorMatcherPtr ptr, struct TensorArray descriptors);

extern "C"
struct TensorArray DescriptorMatcher_getTrainDescriptors(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr);

extern "C"
void DescriptorMatcher_clear(struct DescriptorMatcherPtr ptr);

extern "C"
bool DescriptorMatcher_empty(struct DescriptorMatcherPtr ptr);

extern "C"
void DescriptorMatcher_train(struct DescriptorMatcherPtr ptr);

extern "C"
struct TensorWrapper DescriptorMatcher_match(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, struct TensorWrapper mask);

extern "C"
struct TensorWrapper DescriptorMatcher_match_masks(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper matches,
        struct TensorArray masks);

extern "C"
struct DMatchArray DescriptorMatcher_matchConvert(
        struct DescriptorMatcherPtr ptr, struct TensorWrapper gpu_matches);

extern "C"
struct TensorWrapper DescriptorMatcher_knnMatch(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, int k, struct TensorWrapper mask);

extern "C"
struct TensorWrapper DescriptorMatcher_knnMatch_masks(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, int k, struct TensorArray masks);

extern "C"
struct DMatchArrayOfArrays DescriptorMatcher_knnMatchConvert(
        struct DescriptorMatcherPtr ptr,
        struct TensorWrapper gpu_matches, bool compactResult);

extern "C"
struct TensorWrapper DescriptorMatcher_radiusMatch(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, float maxDistance, struct TensorWrapper mask);

extern "C"
struct TensorWrapper DescriptorMatcher_radiusMatch_masks(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, float maxDistance, struct TensorArray masks);

extern "C"
struct DMatchArrayOfArrays DescriptorMatcher_radiusMatchConvert(
        struct DescriptorMatcherPtr ptr,
        struct TensorWrapper gpu_matches, bool compactResult);

extern "C"
void Feature2DAsync_dtor(struct Feature2DAsyncPtr ptr);

extern "C"
struct TensorWrapper Feature2DAsync_detectAsync(
        struct cutorchInfo info, struct Feature2DAsyncPtr ptr, struct TensorWrapper image,
        struct TensorWrapper keypoints, struct TensorWrapper mask);

extern "C"
struct TensorArray Feature2DAsync_computeAsync(
        struct cutorchInfo info, struct Feature2DAsyncPtr ptr, struct TensorWrapper image,
        struct TensorWrapper keypoints, struct TensorWrapper descriptors);

extern "C"
struct TensorArray Feature2DAsync_detectAndComputeAsync(
        struct cutorchInfo info, struct Feature2DAsyncPtr ptr, struct TensorWrapper image,
        struct TensorWrapper mask, struct TensorWrapper keypoints,
        struct TensorWrapper descriptors, bool useProvidedKeypoints);

extern "C"
struct KeyPointArray Feature2DAsync_convert(
        struct Feature2DAsyncPtr ptr, struct TensorWrapper gpu_keypoints);

extern "C"
struct FastFeatureDetectorPtr FastFeatureDetector_ctor(
        int threshold, bool nonmaxSuppression, int type, int max_npoints);

extern "C"
void FastFeatureDetector_dtor(struct FastFeatureDetectorPtr ptr);

extern "C"
void FastFeatureDetector_setMaxNumPoints(struct FastFeatureDetectorPtr ptr, int val);

extern "C"
int FastFeatureDetector_getMaxNumPoints(struct FastFeatureDetectorPtr ptr);

extern "C"
struct ORBPtr ORB_ctor(
        int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel,
        int WTA_K, int scoreType, int patchSize, int fastThreshold, bool blurForDescriptor);

extern "C"
void ORB_setBlurForDescriptor(struct ORBPtr ptr, bool val);

extern "C"
bool ORB_getBlurForDescriptor(struct ORBPtr ptr);
