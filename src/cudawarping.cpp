#include <cudawarping.hpp>
#include <include/highgui.hpp>

extern "C"
struct TensorWrapper remap(struct THCState *state,
        struct TensorWrapper src, struct TensorWrapper map1,
        struct TensorWrapper map2, int interpolation, struct TensorWrapper dst,
        int borderMode, struct ScalarWrapper borderValue)
{
    if (dst.isNull()) {
        cuda::GpuMat retval;
        cuda::remap(src.toGpuMat(), retval, map1.toGpuMat(), map2.toGpuMat(), interpolation, borderMode, borderValue);
        return TensorWrapper(retval, state);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cuda::GpuMat source = src.toGpuMat();
        cuda::remap(source, source, map1.toGpuMat(), map2.toGpuMat(), interpolation, borderMode, borderValue);
    } else {
        cuda::remap(src.toGpuMat(), dst.toGpuMat(), map1.toGpuMat(), map2.toGpuMat(), interpolation, borderMode, borderValue);
    }
    return dst;
}

extern "C"
struct TensorWrapper resize(struct THCState *state,
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dsize, double fx, double fy,
        int interpolation)
{
    cuda::GpuMat retval;
    if (!dst.isNull()) retval = dst.toGpuMat();
    cuda::resize(src.toGpuMat(), retval, dsize, fx, fy, interpolation);
    return TensorWrapper(retval, state);
}


extern "C"
struct TensorWrapper warpAffine(struct THCState *state,
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper M, struct SizeWrapper dsize,
        int flags, int borderMode, struct ScalarWrapper borderValue)
{
    if (dst.isNull()) {
        cuda::GpuMat retval;
        cuda::warpAffine(src.toGpuMat(), retval, M.toGpuMat(), dsize, flags, borderMode, borderValue);
        return TensorWrapper(retval, state);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cuda::GpuMat source = src.toGpuMat();
        cuda::warpAffine(source, source, M.toGpuMat(), dsize, flags, borderMode, borderValue);
    } else {
        cuda::warpAffine(src.toGpuMat(), dst.toGpuMat(), M.toGpuMat(), dsize, flags, borderMode, borderValue);
    }
    return dst;
}

extern "C"
struct TensorArray buildWarpAffineMaps(
        struct THCState *state, struct TensorWrapper M, bool inverse,
        struct SizeWrapper dsize, struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    std::vector<cuda::GpuMat> retval(2);
    if (!xmap.isNull()) retval[0] = xmap.toGpuMat();
    if (!ymap.isNull()) retval[1] = ymap.toGpuMat();

    cuda::buildWarpAffineMaps(M.toGpuMat(), inverse, dsize, retval[0], retval[1]);

    return TensorArray(retval, state);
}

extern "C"
struct TensorWrapper warpPerspective(struct THCState *state,
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper M, struct SizeWrapper dsize,
        int flags, int borderMode, struct ScalarWrapper borderValue)
{
    if (dst.isNull()) {
        cuda::GpuMat retval;
        cuda::warpPerspective(src.toGpuMat(), retval, M.toGpuMat(), dsize, flags, borderMode, borderValue);
        return TensorWrapper(retval, state);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cuda::GpuMat source = src.toGpuMat();
        cuda::warpPerspective(source, source, M.toGpuMat(), dsize, flags, borderMode, borderValue);
    } else {
        cuda::warpPerspective(src.toGpuMat(), dst.toGpuMat(), M.toGpuMat(), dsize, flags, borderMode, borderValue);
    }
    return dst;
}

extern "C"
struct TensorArray buildWarpPerspectiveMaps(
        struct THCState *state, struct TensorWrapper M, bool inverse,
        struct SizeWrapper dsize, struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    std::vector<cuda::GpuMat> retval(2);
    if (!xmap.isNull()) retval[0] = xmap.toGpuMat();
    if (!ymap.isNull()) retval[1] = ymap.toGpuMat();

    cuda::buildWarpPerspectiveMaps(M.toGpuMat(), inverse, dsize, retval[0], retval[1]);

    return TensorArray(retval, state);
}

extern "C"
struct TensorWrapper rotate(
        struct THCState *state, struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dsize, double angle, double xShift, double yShift, int interpolation)
{
    cuda::GpuMat dstMat;
    if (!dst.isNull()) dstMat = dst.toGpuMat();
    cuda::rotate(src.toGpuMat(), dstMat, dsize, angle, xShift, yShift, interpolation);
    return TensorWrapper(dstMat, state);
}

extern "C"
struct TensorWrapper pyrDown(struct THCState *state,
        struct TensorWrapper src, struct TensorWrapper dst)
{
    if (dst.isNull()) {
        cuda::GpuMat retval;
        cuda::pyrDown(src.toGpuMat(), retval);
        return TensorWrapper(retval, state);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cuda::GpuMat source = src.toGpuMat();
        cuda::pyrDown(source, source);
    } else {
        cuda::pyrDown(src.toGpuMat(), dst.toGpuMat());
    }
    return dst;
}

extern "C"
struct TensorWrapper pyrUp(struct THCState *state,
        struct TensorWrapper src, struct TensorWrapper dst)
{
    if (dst.isNull()) {
        cuda::GpuMat retval;
        cuda::pyrUp(src.toGpuMat(), retval);
        return TensorWrapper(retval, state);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cuda::GpuMat source = src.toGpuMat();
        cuda::pyrUp(source, source);
    } else {
        cuda::pyrUp(src.toGpuMat(), dst.toGpuMat());
    }
    return dst;
}