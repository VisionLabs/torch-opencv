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
struct TensorWrapper Filter_applyCuda(struct cutorchInfo info,
    struct FilterPtr ptr, struct TensorWrapper src, struct TensorWrapper dst);

extern "C"
struct FilterPtr createBoxFilterCuda(
        int srcType, int dstType, struct SizeWrapper ksize, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal);

extern "C"
struct FilterPtr createLinearFilterCuda(
        int srcType, int dstType, struct TensorWrapper kernel, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal);

extern "C"
struct FilterPtr createLaplacianFilterCuda(
        int srcType, int dstType, int ksize, double scale,
        int borderMode, struct ScalarWrapper borderVal);

extern "C"
struct FilterPtr createSeparableLinearFilterCuda(
        int srcType, int dstType, struct TensorWrapper rowKernel,
        struct TensorWrapper columnKernel, struct PointWrapper anchor,
        int rowBorderMode, int columnBorderMode);

extern "C"
struct FilterPtr createDerivFilterCuda(
        int srcType, int dstType, int dx, int dy, int ksize, bool normalize,
        double scale, int rowBorderMode, int columnBorderMode);

extern "C"
struct FilterPtr createSobelFilterCuda(
        int srcType, int dstType, int dx, int dy, int ksize,
        double scale, int rowBorderMode, int columnBorderMode);

extern "C"
struct FilterPtr createScharrFilterCuda(
        int srcType, int dstType, int dx, int dy,
        double scale, int rowBorderMode, int columnBorderMode);

extern "C"
struct FilterPtr createGaussianFilterCuda(
        int srcType, int dstType, struct SizeWrapper ksize,
        double sigma1, double sigma2, int rowBorderMode, int columnBorderMode);

extern "C"
struct FilterPtr createMorphologyFilterCuda(
        int op, int srcType, struct TensorWrapper kernel,
        struct PointWrapper anchor, int iterations);

extern "C"
struct FilterPtr createBoxMaxFilterCuda(
        int srcType, struct SizeWrapper ksize, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal);

extern "C"
struct FilterPtr createBoxMinFilterCuda(
        int srcType, struct SizeWrapper ksize, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal);

extern "C"
struct FilterPtr createRowSumFilterCuda(
        int srcType, int dstType, int ksize, int anchor,
        int borderMode, struct ScalarWrapper borderVal);

extern "C"
struct FilterPtr createColumnSumFilterCuda(
        int srcType, int dstType, int ksize, int anchor,
        int borderMode, struct ScalarWrapper borderVal);
