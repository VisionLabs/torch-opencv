#include <cudafilters.hpp>

extern "C"
struct TensorWrapper Filter_applyCuda(cutorchInfo info,
    struct FilterPtr ptr, struct TensorWrapper src, struct TensorWrapper dst)
{
    cuda::GpuMat retval;
    if (!dst.isNull()) retval = dst.toGpuMat();
    ptr->apply(src.toGpuMat(), retval, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct FilterPtr createBoxFilterCuda(
        int srcType, int dstType, struct SizeWrapper ksize, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal)
{
    return rescueObjectFromPtr(cuda::createBoxFilter(
            srcType, dstType, ksize, anchor, borderMode, borderVal));
}

extern "C"
struct FilterPtr createLinearFilterCuda(
        int srcType, int dstType, struct TensorWrapper kernel, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal)
{
    return rescueObjectFromPtr(cuda::createLinearFilter(
            srcType, dstType, kernel.toMat(), anchor, borderMode,  borderVal));
}

extern "C"
struct FilterPtr createLaplacianFilterCuda(
        int srcType, int dstType, int ksize, double scale,
        int borderMode, struct ScalarWrapper borderVal)
{
    return rescueObjectFromPtr(cuda::createLaplacianFilter(
            srcType, dstType, ksize, scale, borderMode, borderVal));
}

extern "C"
struct FilterPtr createSeparableLinearFilterCuda(
        int srcType, int dstType, struct TensorWrapper rowKernel,
        struct TensorWrapper columnKernel, struct PointWrapper anchor,
        int rowBorderMode, int columnBorderMode)
{
    return rescueObjectFromPtr(cuda::createSeparableLinearFilter(
            srcType, dstType, rowKernel.toMat(), columnKernel.toMat(), anchor,
            rowBorderMode, columnBorderMode));
}

extern "C"
struct FilterPtr createDerivFilterCuda(
        int srcType, int dstType, int dx, int dy, int ksize, bool normalize,
        double scale, int rowBorderMode, int columnBorderMode)
{
    return rescueObjectFromPtr(cuda::createDerivFilter(
            srcType, dstType, dx, dy, ksize, normalize, scale, rowBorderMode, columnBorderMode));
}

extern "C"
struct FilterPtr createSobelFilterCuda(
        int srcType, int dstType, int dx, int dy, int ksize,
        double scale, int rowBorderMode, int columnBorderMode)
{
    return rescueObjectFromPtr(cuda::createSobelFilter(
            srcType, dstType, dx, dy, ksize, scale, rowBorderMode, columnBorderMode));
}

extern "C"
struct FilterPtr createScharrFilterCuda(
        int srcType, int dstType, int dx, int dy,
        double scale, int rowBorderMode, int columnBorderMode)
{
    return rescueObjectFromPtr(cuda::createScharrFilter(
            srcType, dstType, dx, dy, scale, rowBorderMode, columnBorderMode));
}

extern "C"
struct FilterPtr createGaussianFilterCuda(
        int srcType, int dstType, struct SizeWrapper ksize,
        double sigma1, double sigma2, int rowBorderMode, int columnBorderMode)
{
    return rescueObjectFromPtr(cuda::createGaussianFilter(
            srcType, dstType, ksize, sigma1, sigma2, rowBorderMode, columnBorderMode));
}

extern "C"
struct FilterPtr createMorphologyFilterCuda(
        int op, int srcType, struct TensorWrapper kernel,
        struct PointWrapper anchor, int iterations)
{
    return rescueObjectFromPtr(cuda::createMorphologyFilter(
            op, srcType, kernel.toMat(), anchor, iterations));
}

extern "C"
struct FilterPtr createBoxMaxFilterCuda(
        int srcType, struct SizeWrapper ksize, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal)
{
    return rescueObjectFromPtr(cuda::createBoxMaxFilter(
            srcType, ksize, anchor, borderMode, borderVal));
}

extern "C"
struct FilterPtr createBoxMinFilterCuda(
        int srcType, struct SizeWrapper ksize, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal)
{
    return rescueObjectFromPtr(cuda::createBoxMaxFilter(
            srcType, ksize, anchor, borderMode, borderVal));
}

extern "C"
struct FilterPtr createRowSumFilterCuda(
        int srcType, int dstType, int ksize, int anchor,
        int borderMode, struct ScalarWrapper borderVal)
{
    return rescueObjectFromPtr(cuda::createRowSumFilter(
            srcType, dstType, ksize, anchor, borderMode, borderVal));
}

extern "C"
struct FilterPtr createColumnSumFilterCuda(
        int srcType, int dstType, int ksize, int anchor,
        int borderMode, struct ScalarWrapper borderVal)
{
    return rescueObjectFromPtr(cuda::createColumnSumFilter(
            srcType, dstType, ksize, anchor, borderMode, borderVal));
}
