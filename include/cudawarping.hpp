#include <CUDACommon.hpp>
#include <include/Classes.hpp>
#include <opencv2/cudawarping.hpp>

extern "C"
struct TensorWrapper remap(struct cutorchInfo info,
                           struct TensorWrapper src, struct TensorWrapper map1,
                           struct TensorWrapper map2, int interpolation, struct TensorWrapper dst,
                           int borderMode, struct ScalarWrapper borderValue);

extern "C"
struct TensorWrapper resize(struct cutorchInfo info,
                            struct TensorWrapper src, struct TensorWrapper dst,
                            struct SizeWrapper dsize, double fx, double fy,
                            int interpolation);

extern "C"
struct TensorWrapper warpAffine(struct cutorchInfo info,
                                struct TensorWrapper src, struct TensorWrapper dst,
                                struct TensorWrapper M, struct SizeWrapper dsize,
                                int flags, int borderMode, struct ScalarWrapper borderValue);

extern "C"
struct TensorArray buildWarpAffineMaps(
        struct cutorchInfo info, struct TensorWrapper M, bool inverse,
        struct SizeWrapper dsize, struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C"
struct TensorWrapper warpPerspective(struct cutorchInfo info,
                                     struct TensorWrapper src, struct TensorWrapper dst,
                                     struct TensorWrapper M, struct SizeWrapper dsize,
                                     int flags, int borderMode, struct ScalarWrapper borderValue);

extern "C"
struct TensorArray buildWarpPerspectiveMaps(
        struct cutorchInfo info, struct TensorWrapper M, bool inverse,
        struct SizeWrapper dsize, struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C"
struct TensorWrapper rotate(
        struct cutorchInfo info, struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dsize, double angle, double xShift, double yShift, int interpolation);

extern "C"
struct TensorWrapper pyrDown(struct cutorchInfo info,
                             struct TensorWrapper src, struct TensorWrapper dst);

extern "C"
struct TensorWrapper pyrUp(struct cutorchInfo info,
                           struct TensorWrapper src, struct TensorWrapper dst);
