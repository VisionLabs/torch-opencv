#include <cudafeatures2d.hpp>

extern "C"
struct DescriptorMatcherPtr createBFMatcher(int normType) {
    return rescueObjectFromPtr(cuda::DescriptorMatcher::createBFMatcher(normType));
}

extern "C"
bool DescriptorMatcher_isMaskSupported(struct DescriptorMatcherPtr ptr) {
    return ptr->isMaskSupported();
}

extern "C"
void DescriptorMatcher_add(
        struct DescriptorMatcherPtr ptr, struct TensorArray descriptors) {
    ptr->add(descriptors.toGpuMatList());
}

extern "C"
void DescriptorMatcher_clear(struct DescriptorMatcherPtr ptr) {
    ptr->clear();
}

extern "C"
bool DescriptorMatcher_empty(struct DescriptorMatcherPtr ptr) {
    return ptr->empty();
}

extern "C"
void DescriptorMatcher_train(struct DescriptorMatcherPtr ptr) {
    ptr->train();
}

extern "C"
struct TensorWrapper DescriptorMatcher_match(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, struct TensorWrapper mask)
{
    cuda::GpuMat retval;
    if (!matches.isNull()) retval = matches.toGpuMat();
    ptr->matchAsync(
            queryDescriptors.toGpuMat(), trainDescriptors.toGpuMat(), retval,
            TO_GPUMAT_OR_NOARRAY(mask), prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper DescriptorMatcher_match_masks(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper matches,
        struct TensorArray masks)
{
    cuda::GpuMat retval;
    if (!matches.isNull()) retval = matches.toGpuMat();
    ptr->matchAsync(
            queryDescriptors.toGpuMat(), retval,
            TO_GPUMAT_LIST_OR_NOARRAY(masks), prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper DescriptorMatcher_knnMatch(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, int k, struct TensorWrapper mask)
{
    cuda::GpuMat retval;
    if (!matches.isNull()) retval = matches.toGpuMat();
    ptr->knnMatchAsync(
            queryDescriptors.toGpuMat(), trainDescriptors.toGpuMat(), retval,
            k, TO_GPUMAT_OR_NOARRAY(mask), prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper DescriptorMatcher_knnMatch_masks(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, int k, struct TensorArray masks)
{
    cuda::GpuMat retval;
    if (!matches.isNull()) retval = matches.toGpuMat();
    ptr->knnMatchAsync(
            queryDescriptors.toGpuMat(), trainDescriptors.toGpuMat(), retval,
            k, TO_GPUMAT_LIST_OR_NOARRAY(masks), prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct DMatchArrayOfArrays DescriptorMatcher_knnMatchConvert(
        struct DescriptorMatcherPtr ptr,
        struct TensorWrapper gpu_matches, bool compactResult)
{
    std::vector<std::vector<cv::DMatch>> retval;
    ptr->knnMatchConvert(gpu_matches.toGpuMat(), retval, compactResult);
    return DMatchArrayOfArrays(retval);
}

extern "C"
struct TensorWrapper DescriptorMatcher_radiusMatch(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, float maxDistance, struct TensorWrapper mask)
{
    cuda::GpuMat retval;
    if (!matches.isNull()) retval = matches.toGpuMat();
    ptr->radiusMatchAsync(
            queryDescriptors.toGpuMat(), trainDescriptors.toGpuMat(),
            retval, maxDistance, TO_GPUMAT_OR_NOARRAY(mask), prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper DescriptorMatcher_radiusMatch_masks(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, float maxDistance, struct TensorArray masks)
{
    cuda::GpuMat retval;
    if (!matches.isNull()) retval = matches.toGpuMat();
    ptr->radiusMatchAsync(
            queryDescriptors.toGpuMat(), trainDescriptors.toGpuMat(),
            retval, maxDistance, TO_GPUMAT_LIST_OR_NOARRAY(masks), prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct DMatchArrayOfArrays DescriptorMatcher_radiusMatchConvert(
        struct DescriptorMatcherPtr ptr,
        struct TensorWrapper gpu_matches, bool compactResult)
{
    std::vector<std::vector<cv::DMatch>> retval;
    ptr->radiusMatchConvert(gpu_matches.toGpuMat(), retval, compactResult);
    return DMatchArrayOfArrays(retval);
}