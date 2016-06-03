#include <cudaimgproc.hpp>

extern "C"
struct TensorWrapper cvtColorCuda(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst, int code, int dstCn)
{
    GpuMatT dstMat = dst.toGpuMatT();
    cuda::cvtColor(src.toGpuMat(), dstMat, code, dstCn, prepareStream(info));
    return TensorWrapper(dstMat, info.state);
}

extern "C"
struct TensorWrapper demosaicingCuda(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst, int code, int dcn)
{
    cuda::GpuMat retval = dst.toGpuMat();
    cuda::demosaicing(src.toGpuMat(), retval, code, dcn, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
void swapChannelsCuda(
        struct cutorchInfo info, struct TensorWrapper image,
        struct Vec4iWrapper dstOrder)
{
    cuda::swapChannels(image.toGpuMat(), &dstOrder.v0, prepareStream(info));
}

extern "C"
struct TensorWrapper gammaCorrectionCuda(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst, bool forward)
{
    cuda::GpuMat retval = dst.toGpuMat();
    cuda::gammaCorrection(src.toGpuMat(), retval, forward, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper alphaCompCuda(struct cutorchInfo info,
        struct TensorWrapper img1, struct TensorWrapper img2,
        struct TensorWrapper dst, int alpha_op)
{
    cuda::GpuMat retval = dst.toGpuMat();
    cuda::alphaComp(img1.toGpuMat(), img2.toGpuMat(), retval, alpha_op, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper calcHistCuda(
        struct cutorchInfo info, struct TensorWrapper src, struct TensorWrapper hist)
{
    cuda::GpuMat retval = hist.toGpuMat();
    cuda::calcHist(src.toGpuMat(), retval, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper equalizeHistCuda(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst)
{
    cuda::GpuMat retval = dst.toGpuMat();
    cuda::equalizeHist(src.toGpuMat(), retval);
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper evenLevelsCuda(struct cutorchInfo info,
        struct TensorWrapper levels, int nLevels, int lowerLevel, int upperLevel)
{
    cuda::GpuMat retval = levels.toGpuMat();
    cuda::evenLevels(retval, nLevels, lowerLevel, upperLevel, prepareStream(info));
    return TensorWrapper(retval, info.state);
}
//
//extern "C"
//struct TensorWrapper histEven(struct cutorchInfo info,
//        struct TensorWrapper src, struct TensorWrapper hist,
//        int histSize, int lowerLevel, int upperLevel)
//{
//    cuda::GpuMat retval;
//    if (!hist.isNull()) retval = hist.toGpuMat();
//    cuda::histEven(src.toGpuMat(), retval, hS, lowerLevel, upperLevel);
//    return TensorWrapper(retval, info.state);
//}
//
//extern "C"
//struct TensorArray histEven_4(struct cutorchInfo info,
//        struct TensorWrapper src, struct TensorArray hist, struct TensorWrapper histSize,
//        struct TensorWrapper lowerLevel, struct TensorWrapper upperLevel)
//{
//    std::vector<cuda::GpuMat> retval(4);
//    if (!hist.isNull()) retval = hist.toGpuMatList();
//    cuda::histEven(
//            src.toGpuMat(), retval.data(),
//            static_cast<int*>(histSize.toMat().data),
//            static_cast<int*>(lowerLevel.toMat().data),
//            static_cast<int*>(upperLevel.toMat().data));
//    return TensorArray(retval, info.state);
//}
//
//extern "C"
//struct TensorWrapper histRange(struct cutorchInfo info,
//        struct TensorWrapper src, struct TensorWrapper hist,
//        struct TensorWrapper levels)
//{
//    cuda::GpuMat retval;
//    if (!hist.isNull()) retval = hist.toGpuMat();
//    cuda::histRange(src.toGpuMat(), retval, levels.toMat(), prepareStream(info));
//    return TensorWrapper(retval, info.state);
//}
//
//extern "C"
//struct TensorArray histRange_4(struct cutorchInfo info,
//        struct TensorWrapper src, struct TensorArray hist, struct TensorWrapper levels)
//{
//    std::vector<cuda::GpuMat> retval(4);
//    if (!hist.isNull()) retval = hist.toGpuMatList();
//    cuda::histRange(
//            src.toGpuMat(), retval.data(),
//            static_cast<int*>(levels.toMat().data),
//            prepareStream(info));
//    return TensorArray(retval, info.state);
//}

extern "C"
struct CornernessCriteriaPtr createHarrisCornerCuda(
        int srcType, int blockSize, int ksize, double k, int borderType)
{
    return rescueObjectFromPtr(cuda::createHarrisCorner(
            srcType, blockSize, ksize, k, borderType));
}

extern "C"
struct CornernessCriteriaPtr createMinEigenValCornerCuda(
        int srcType, int blockSize, int ksize, int borderType)
{
    return rescueObjectFromPtr(cuda::createMinEigenValCorner(
            srcType, blockSize, ksize, borderType));
}

extern "C"
struct TensorWrapper CornernessCriteria_computeCuda(
        struct cutorchInfo info, struct CornernessCriteriaPtr ptr,
        struct TensorWrapper src, struct TensorWrapper dst)
{
    cuda::GpuMat retval = dst.toGpuMat();
    ptr->compute(src.toGpuMat(), retval, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct CornersDetectorPtr createGoodFeaturesToTrackDetectorCuda(
        int srcType, int maxCorners, double qualityLevel, double minDistance,
        int blockSize, bool useHarrisDetector, double harrisK)
{
    return rescueObjectFromPtr(cuda::createGoodFeaturesToTrackDetector(
            srcType, maxCorners, qualityLevel, minDistance,
            blockSize, useHarrisDetector, harrisK));
}

extern "C"
struct TensorWrapper CornersDetector_detectCuda(
        struct cutorchInfo info, struct CornersDetectorPtr ptr, struct TensorWrapper image,
        struct TensorWrapper corners, struct TensorWrapper mask)
{
    cuda::GpuMat retval = corners.toGpuMat();
    ptr->detect(image.toGpuMat(), retval, TO_GPUMAT_OR_NOARRAY(mask));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TemplateMatchingPtr createTemplateMatchingCuda(
        int srcType, int method, struct SizeWrapper user_block_size)
{
    return rescueObjectFromPtr(cuda::createTemplateMatching(
            srcType, method, user_block_size));
}

extern "C"
struct TensorWrapper TemplateMatching_matchCuda(
        struct cutorchInfo info, struct TemplateMatchingPtr ptr, struct TensorWrapper image,
        struct TensorWrapper templ, struct TensorWrapper result)
{
    cuda::GpuMat retval = result.toGpuMat();
    ptr->match(image.toGpuMat(), templ.toGpuMat(), retval, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper bilateralFilterCuda(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst, int kernel_size,
        float sigma_color, float sigma_spatial, int borderMode)
{
    cuda::GpuMat retval = dst.toGpuMat();
    cuda::bilateralFilter(
            src.toGpuMat(), retval, kernel_size, sigma_color,
            sigma_color, borderMode, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper blendLinearCuda(struct cutorchInfo info,
        struct TensorWrapper img1, struct TensorWrapper img2, struct TensorWrapper weights1,
        struct TensorWrapper weights2, struct TensorWrapper result)
{
    cuda::GpuMat retval = result.toGpuMat();
    cuda::blendLinear(
            img1.toGpuMat(), img2.toGpuMat(), weights1.toGpuMat(), weights2.toGpuMat(),
            retval, prepareStream(info));
    return TensorWrapper(retval, info.state);
}
