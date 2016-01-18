#include <CUDACommon.hpp>
#include <include/Classes.hpp>
#include <opencv2/cudafilters.hpp>

// Filter

struct FilterPtr {
    void *ptr;
    inline cuda::Filter * operator->() { return static_cast<cuda::Filter *>(ptr); }
    inline FilterPtr(cuda::Filter *ptr) { this->ptr = ptr; }
    inline cuda::Filter & operator*() { return *static_cast<cuda::Filter *>(this->ptr); }
};

extern "C"
struct TensorWrapper Filter_apply(struct THCState *state,
    struct FilterPtr ptr, struct TensorWrapper src, struct TensorWrapper dst);

extern "C"
struct FilterPtr createBoxFilter(
        int srcType, int dstType, struct SizeWrapper ksize, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal);

extern "C"
struct FilterPtr createLinearFilter(
        int srcType, int dstType, struct TensorWrapper kernel, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal);

extern "C"
struct FilterPtr createLaplacianFilter(
        int srcType, int dstType, int ksize, double scale,
        int borderMode, struct ScalarWrapper borderVal);

extern "C"
struct FilterPtr createSeparableLinearFilter(
        int srcType, int dstType, struct TensorWrapper rowKernel,
        struct TensorWrapper columnKernel, struct PointWrapper anchor,
        int rowBorderMode, int columnBorderMode);

extern "C"
struct FilterPtr createDerivFilter(
        int srcType, int dstType, int dx, int dy, int ksize, bool normalize,
        double scale, int rowBorderMode, int columnBorderMode);

extern "C"
struct FilterPtr createSobelFilter(
        int srcType, int dstType, int dx, int dy, int ksize,
        double scale, int rowBorderMode, int columnBorderMode);

extern "C"
struct FilterPtr createScharrFilter(
        int srcType, int dstType, int dx, int dy,
        double scale, int rowBorderMode, int columnBorderMode);

extern "C"
struct FilterPtr createGaussianFilter(
        int srcType, int dstType, struct SizeWrapper ksize,
        double sigma1, double sigma2, int rowBorderMode, int columnBorderMode);

extern "C"
struct FilterPtr createMorphologyFilter(
        int op, int srcType, struct TensorWrapper kernel,
        struct PointWrapper anchor, int iterations);

extern "C"
struct FilterPtr createBoxMaxFilter(
        int srcType, struct SizeWrapper ksize, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal);

extern "C"
struct FilterPtr createBoxMinFilter(
        int srcType, struct SizeWrapper ksize, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal);

extern "C"
struct FilterPtr createRowSumFilter(
        int srcType, int dstType, int ksize, int anchor,
        int borderMode, struct ScalarWrapper borderVal);

extern "C"
struct FilterPtr createColumnSumFilter(
        int srcType, int dstType, int ksize, int anchor,
        int borderMode, struct ScalarWrapper borderVal);
