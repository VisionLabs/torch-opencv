#include <cudafilters.hpp>

/*  Whenever we call an OpenCV-CUDA function from Lua, it's necessary
 *  to tell OpenCV which device and stream currently in use by cutorch.
 *  For this, as `cv.cuda<whatever` is required and loaded, it stores a single
 *  `cv::cuda::Stream` object. When invoking an OpenCV function, we must
 *  refresh that object and pass through a reference to it. */

// Create that object
FakeStream fakeStream;

cuda::Stream & prepareStream(cutorchInfo info) {
    cuda::setDevice(info.deviceID - 1);
    fakeStream.impl_ = cv::makePtr<FakeStreamImpl>(info.state->currentStream);
    return *reinterpret_cast<cuda::Stream *>(&fakeStream);
}

extern "C"
struct TensorWrapper Filter_apply(cutorchInfo info,
    struct FilterPtr ptr, struct TensorWrapper src, struct TensorWrapper dst)
{
    cuda::GpuMat retval;
    if (!dst.isNull()) retval = dst.toGpuMat();
    ptr->apply(src.toGpuMat(), retval, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct FilterPtr createBoxFilter(
        int srcType, int dstType, struct SizeWrapper ksize, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal)
{
    return rescueObjectFromPtr(cuda::createBoxFilter(
            srcType, dstType, ksize, anchor, borderMode, borderVal));
}

extern "C"
struct FilterPtr createLinearFilter(
        int srcType, int dstType, struct TensorWrapper kernel, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal)
{
    return rescueObjectFromPtr(cuda::createLinearFilter(
            srcType, dstType, kernel.toMat(), anchor, borderMode,  borderVal));
}

extern "C"
struct FilterPtr createLaplacianFilter(
        int srcType, int dstType, int ksize, double scale,
        int borderMode, struct ScalarWrapper borderVal)
{
    return rescueObjectFromPtr(cuda::createLaplacianFilter(
            srcType, dstType, ksize, scale, borderMode, borderVal));
}

extern "C"
struct FilterPtr createSeparableLinearFilter(
        int srcType, int dstType, struct TensorWrapper rowKernel,
        struct TensorWrapper columnKernel, struct PointWrapper anchor,
        int rowBorderMode, int columnBorderMode)
{
    return rescueObjectFromPtr(cuda::createSeparableLinearFilter(
            srcType, dstType, rowKernel.toMat(), columnKernel.toMat(), anchor,
            rowBorderMode, columnBorderMode));
}

extern "C"
struct FilterPtr createDerivFilter(
        int srcType, int dstType, int dx, int dy, int ksize, bool normalize,
        double scale, int rowBorderMode, int columnBorderMode)
{
    return rescueObjectFromPtr(cuda::createDerivFilter(
            srcType, dstType, dx, dy, ksize, normalize, scale, rowBorderMode, columnBorderMode));
}

extern "C"
struct FilterPtr createSobelFilter(
        int srcType, int dstType, int dx, int dy, int ksize,
        double scale, int rowBorderMode, int columnBorderMode)
{
    return rescueObjectFromPtr(cuda::createSobelFilter(
            srcType, dstType, dx, dy, ksize, scale, rowBorderMode, columnBorderMode));
}

extern "C"
struct FilterPtr createScharrFilter(
        int srcType, int dstType, int dx, int dy,
        double scale, int rowBorderMode, int columnBorderMode)
{
    return rescueObjectFromPtr(cuda::createScharrFilter(
            srcType, dstType, dx, dy, scale, rowBorderMode, columnBorderMode));
}

extern "C"
struct FilterPtr createGaussianFilter(
        int srcType, int dstType, struct SizeWrapper ksize,
        double sigma1, double sigma2, int rowBorderMode, int columnBorderMode)
{
    return rescueObjectFromPtr(cuda::createGaussianFilter(
            srcType, dstType, ksize, sigma1, sigma2, rowBorderMode, columnBorderMode));
}

extern "C"
struct FilterPtr createMorphologyFilter(
        int op, int srcType, struct TensorWrapper kernel,
        struct PointWrapper anchor, int iterations)
{
    return rescueObjectFromPtr(cuda::createMorphologyFilter(
            op, srcType, kernel.toMat(), anchor, iterations));
}

extern "C"
struct FilterPtr createBoxMaxFilter(
        int srcType, struct SizeWrapper ksize, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal)
{
    return rescueObjectFromPtr(cuda::createBoxMaxFilter(
            srcType, ksize, anchor, borderMode, borderVal));
}

extern "C"
struct FilterPtr createBoxMinFilter(
        int srcType, struct SizeWrapper ksize, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal)
{
    return rescueObjectFromPtr(cuda::createBoxMaxFilter(
            srcType, ksize, anchor, borderMode, borderVal));
}

extern "C"
struct FilterPtr createRowSumFilter(
        int srcType, int dstType, int ksize, int anchor,
        int borderMode, struct ScalarWrapper borderVal)
{
    return rescueObjectFromPtr(cuda::createRowSumFilter(
            srcType, dstType, ksize, anchor, borderMode, borderVal));
}

extern "C"
struct FilterPtr createColumnSumFilter(
        int srcType, int dstType, int ksize, int anchor,
        int borderMode, struct ScalarWrapper borderVal)
{
    return rescueObjectFromPtr(cuda::createColumnSumFilter(
            srcType, dstType, ksize, anchor, borderMode, borderVal));
}
