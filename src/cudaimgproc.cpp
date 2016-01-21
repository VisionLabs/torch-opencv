#include <cudaimgproc.hpp>

extern "C" struct TensorWrapper cvtColor(cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst, int code, int dstCn)
{
    cuda::GpuMat retval;
    if (!dst.isNull()) retval = dst.toGpuMat();
    cuda::cvtColor(src.toGpuMat(), retval, code, dstCn, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper demosaicing(cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst, int code, int dcn)
{
    cuda::GpuMat retval;
    if (!dst.isNull()) retval = dst.toGpuMat();
    cuda::demosaicing(_src.toGpuMat(), retval, code, dcn, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
void swapChannels(
        cutorchInfo info, struct TensorWrapper image, 
        struct TensorWrapper dstOrder)
{
    std::vector<int> dstOrderVec = dstOrder.toMat();
    cuda::swapChannels(image.toGpuMat(), dstOrderVec.data(), prepareStream(info));
}

extern "C"
struct TensorWrapper gammaCorrection(cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst, bool forward)
{
    cuda::GpuMat retval;
    if (!dst.isNull()) retval = dst.toGpuMat();
    cuda::gammaCorrection(src.toGpuMat(), retval, forward, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper alphaComp(cutorchInfo info,
        struct TensorWrapper img1, struct TensorWrapper img2,
        struct TensorWrapper dst, int alpha_op)
{
    cuda::GpuMat retval;
    if (!dst.isNull()) retval = dst.toGpuMat();
    cuda::alphaComp(img1.toGpuMat(), img2.toGpuMat(), retval, alpha_op, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper calcHist(
        cutorchInfo info, struct TensorWrapper src, struct TensorWrapper hist)
{
    cuda::GpuMat retval;
    if (!dst.isNull()) retval = dst.toGpuMat();
    cuda::calcHist(src.toGpuMat(), retval, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper equalizeHist(cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst)
{
    cuda::GpuMat retval;
    if (!dst.isNull()) retval = dst.toGpuMat();
    cuda::equalizeHist(src.toGpuMat(), retval);
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper evenLevels(struct cutorchInfo info,
        struct TensorWrapper levels, int nLevels, int lowerLevel, int upperLevel)
{
    cuda::GpuMat retval;
    if (!levels.isNull()) retval = levels.toGpuMat();
    cuda::evenLevels(retval, nLevels, lowerLevel, upperLevel, prepareStream(info));
}

extern "C"
struct TensorWrapper histEven(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper hist,
        int histSize, int lowerLevel, int upperLevel)
{
    cuda::GpuMat retval;
    if (!hist.isNull()) retval = hist.toGpuMat();
    cuda::histEven(src.toGpuMat(), retval, hS, lowerLevel, upperLevel);
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorArray histEven_4(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorArray hist, struct TensorWrapper histSize,
        struct TensorWrapper lowerLevel, struct TensorWrapper upperLevel)
{
    std::vector<cuda::GpuMat> retval(4);
    if (!hist.isNull()) retval = hist.toGpuMatList();
    cuda::histEven(
            src.toGpuMat(), retval.data(),
            static_cast<int*>(histSize.toMat().data),
            static_cast<int*>(lowerLevel.toMat().data),
            static_cast<int*>(upperLevel.toMat().data));
    return TensorArray(retval, info.state);
}

extern "C"
struct TensorWrapper histRange(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper hist,
        struct TensorWrapper levels)
{
    cuda::GpuMat retval;
    if (!hist.isNull()) retval = hist.toGpuMat();
    cuda::histRange(src.toGpuMat(), retval, levels.toMat(), prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorArray histRange_4(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorArray hist, struct TensorWrapper levels)
{
    std::vector<cuda::GpuMat> retval(4);
    if (!hist.isNull()) retval = hist.toGpuMatList();
    cuda::histRange(
            src.toGpuMat(), retval.data(),
            static_cast<int*>(levels.toMat().data),
            prepareStream(info));
    return TensorArray(retval, info.state);
}

extern "C"
struct CornernessCriteriaPtr createHarrisCorner(
        int srcType, int blockSize, int ksize, double k, int borderType)
{
    return rescueObjectFromPtr(cuda::createHarrisCorner(
            srcType, blockSize, ksize, k, borderType));
}

extern "C"
struct CornernessCriteriaPtr createMinEigenValCorner(
        int srcType, int blockSize, int ksize, int borderType)
{
    return rescueObjectFromPtr(cuda::createMinEigenValCorner(
            srcType, blockSize, ksize, borderType));
}

extern "C"
struct TensorWrapper CornernessCriteria_compute(
        struct cutorchInfo info, struct CornernessCriteriaPtr ptr,
        struct TensorWrapper src, struct TensorWrapper dst)
{
    cuda::GpuMat retval;
    if (!dst.isNull()) retval = dst.toGpuMat();
    ptr->compute(src.toGpuMat(), retval, prepareStream(info));
}

