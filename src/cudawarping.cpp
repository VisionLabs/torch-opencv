#include <cudawarping.hpp>

extern "C"
struct TensorWrapper remapCuda(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper map1,
        struct TensorWrapper map2, int interpolation, struct TensorWrapper dst,
        int borderMode, struct ScalarWrapper borderValue)
{
    if (dst.isNull()) {
        cuda::GpuMat retval;
        cuda::remap(src.toGpuMat(), retval, map1.toGpuMat(), map2.toGpuMat(),
                    interpolation, borderMode, borderValue, prepareStream(info));
        return TensorWrapper(retval, info.state);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cuda::GpuMat source = src.toGpuMat();
        cuda::remap(source, source, map1.toGpuMat(), map2.toGpuMat(), interpolation,
                    borderMode, borderValue, prepareStream(info));
    } else {
        cuda::remap(src.toGpuMat(), dst.toGpuMat(), map1.toGpuMat(), map2.toGpuMat(),
                    interpolation, borderMode, borderValue, prepareStream(info));
    }
    return dst;
}

extern "C"
struct TensorWrapper resizeCuda(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dsize, double fx, double fy,
        int interpolation)
{
    GpuMatT dstMat = dst.toGpuMatT();
    cuda::resize(src.toGpuMat(), dstMat, dsize, fx, fy, interpolation, prepareStream(info));
    return TensorWrapper(dstMat, info.state);
}


extern "C"
struct TensorWrapper warpAffineCuda(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper M, struct SizeWrapper dsize,
        int flags, int borderMode, struct ScalarWrapper borderValue)
{
    if (dst.isNull()) {
        cuda::GpuMat retval;
        cuda::warpAffine(src.toGpuMat(), retval, M.toGpuMat(), dsize, flags,
                         borderMode, borderValue, prepareStream(info));
        return TensorWrapper(retval, info.state);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cuda::GpuMat source = src.toGpuMat();
        cuda::warpAffine(source, source, M.toGpuMat(), dsize, flags, borderMode,
                         borderValue, prepareStream(info));
    } else {
        cuda::warpAffine(src.toGpuMat(), dst.toGpuMat(), M.toGpuMat(), dsize,
                         flags, borderMode, borderValue, prepareStream(info));
    }
    return dst;
}

extern "C"
struct TensorArray buildWarpAffineMapsCuda(
        struct cutorchInfo info, struct TensorWrapper M, bool inverse,
        struct SizeWrapper dsize, struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    std::vector<cuda::GpuMat> retval(2);
    if (!xmap.isNull()) retval[0] = xmap.toGpuMat();
    if (!ymap.isNull()) retval[1] = ymap.toGpuMat();

    cuda::buildWarpAffineMaps(M.toGpuMat(), inverse, dsize,
                              retval[0], retval[1], prepareStream(info));

    return TensorArray(retval, info.state);
}

extern "C"
struct TensorWrapper warpPerspectiveCuda(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper M, struct SizeWrapper dsize,
        int flags, int borderMode, struct ScalarWrapper borderValue)
{
    if (dst.isNull()) {
        cuda::GpuMat retval;
        cuda::warpPerspective(src.toGpuMat(), retval, M.toGpuMat(), dsize,
                              flags, borderMode, borderValue, prepareStream(info));
        return TensorWrapper(retval, info.state);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cuda::GpuMat source = src.toGpuMat();
        cuda::warpPerspective(source, source, M.toGpuMat(), dsize, flags,
                              borderMode, borderValue, prepareStream(info));
    } else {
        cuda::warpPerspective(src.toGpuMat(), dst.toGpuMat(), M.toGpuMat(),
                              dsize, flags, borderMode, borderValue, prepareStream(info));
    }
    return dst;
}

extern "C"
struct TensorArray buildWarpPerspectiveMapsCuda(
        struct cutorchInfo info, struct TensorWrapper M, bool inverse,
        struct SizeWrapper dsize, struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    std::vector<cuda::GpuMat> retval(2);
    if (!xmap.isNull()) retval[0] = xmap.toGpuMat();
    if (!ymap.isNull()) retval[1] = ymap.toGpuMat();

    cuda::buildWarpPerspectiveMaps(M.toGpuMat(), inverse, dsize,
                                   retval[0], retval[1], prepareStream(info));

    return TensorArray(retval, info.state);
}

extern "C"
struct TensorWrapper rotateCuda(
        struct cutorchInfo info, struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dsize, double angle, double xShift, double yShift, int interpolation)
{
    cuda::GpuMat dstMat;
    if (!dst.isNull()) dstMat = dst.toGpuMat();
    cuda::rotate(src.toGpuMat(), dstMat, dsize, angle, xShift, yShift,
                 interpolation, prepareStream(info));
    return TensorWrapper(dstMat, info.state);
}

extern "C"
struct TensorWrapper pyrDownCuda(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst)
{
    if (dst.isNull()) {
        cuda::GpuMat retval;
        cuda::pyrDown(src.toGpuMat(), retval, prepareStream(info));
        return TensorWrapper(retval, info.state);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cuda::GpuMat source = src.toGpuMat();
        cuda::pyrDown(source, source, prepareStream(info));
    } else {
        cuda::pyrDown(src.toGpuMat(), dst.toGpuMat(), prepareStream(info));
    }
    return dst;
}

extern "C"
struct TensorWrapper pyrUpCuda(struct cutorchInfo info,
        struct TensorWrapper src, struct TensorWrapper dst)
{
    if (dst.isNull()) {
        cuda::GpuMat retval;
        cuda::pyrUp(src.toGpuMat(), retval, prepareStream(info));
        return TensorWrapper(retval, info.state);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cuda::GpuMat source = src.toGpuMat();
        cuda::pyrUp(source, source, prepareStream(info));
    } else {
        cuda::pyrUp(src.toGpuMat(), dst.toGpuMat(), prepareStream(info));
    }
    return dst;
}