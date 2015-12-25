#include <CUDACommon.hpp>
#include <include/Classes.hpp>
#include <opencv2/cudaarithm.hpp>

struct LookUpTablePtr {
    void *ptr;

    inline cuda::LookUpTable * operator->() { return static_cast<cuda::LookUpTable *>(ptr); }
    inline LookUpTablePtr(cuda::LookUpTable *ptr) { this->ptr = ptr; }
    inline operator const cuda::LookUpTable &() { return *static_cast<cuda::LookUpTable *>(ptr); }
};

struct ConvolutionPtr {
    void *ptr;

    inline cuda::Convolution * operator->() { return static_cast<cuda::Convolution *>(ptr); }
    inline ConvolutionPtr(cuda::Convolution *ptr) { this->ptr = ptr; }
    inline operator const cuda::Convolution &() { return *static_cast<cuda::Convolution *>(ptr); }
};

extern "C"
struct TensorWrapper min(
        struct THCState *state, struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper dst);

extern "C"
struct TensorWrapper max(
        struct THCState *state, struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper dst);

extern "C"
struct TensorPlusDouble threshold(
        struct THCState *state, struct TensorWrapper src,
        struct TensorWrapper dst, double thresh, double maxval, int type);

extern "C"
struct TensorWrapper magnitude(
        struct THCState *state, struct TensorWrapper xy, struct TensorWrapper magnitude);

extern "C"
struct TensorWrapper magnitudeSqr(
        struct THCState *state, struct TensorWrapper xy, struct TensorWrapper magnitude);

extern "C"
struct TensorWrapper magnitude2(
        struct THCState *state, struct TensorWrapper x, struct TensorWrapper y, struct TensorWrapper magnitude);

extern "C"
struct TensorWrapper magnitudeSqr2(
        struct THCState *state, struct TensorWrapper x, struct TensorWrapper y, struct TensorWrapper magnitudeSqr);

extern "C"
struct TensorWrapper phase(
        struct THCState *state, struct TensorWrapper x, struct TensorWrapper y,
        struct TensorWrapper angle, bool angleInDegrees);

extern "C"
struct TensorArray cartToPolar(
        struct THCState *state, struct TensorWrapper x, struct TensorWrapper y,
        struct TensorWrapper magnitude, struct TensorWrapper angle, bool angleInDegrees);

extern "C"
struct TensorArray polarToCart(
        struct THCState *state, struct TensorWrapper magnitude, struct TensorWrapper angle,
        struct TensorWrapper x, struct TensorWrapper y, bool angleInDegrees);

extern "C"
struct LookUpTablePtr LookUpTable_ctor(
        struct THCState *state, struct TensorWrapper lut);

extern "C"
struct TensorWrapper LookUpTable_transform(
        struct THCState *state, struct LookUpTablePtr ptr,
        struct TensorWrapper src, struct TensorWrapper dst);

extern "C"
struct TensorWrapper rectStdDev(
        struct THCState *state, struct TensorWrapper src, struct TensorWrapper sqr,
        struct TensorWrapper dst, struct RectWrapper rect);

extern "C"
struct TensorWrapper normalize(
        struct THCState *state, struct TensorWrapper src, struct TensorWrapper dst,
        double alpha, double beta, int norm_type, int dtype);

extern "C"
struct TensorWrapper integral(
        struct THCState *state, struct TensorWrapper src, struct TensorWrapper sum);

extern "C"
struct TensorWrapper sqrIntegral(
        struct THCState *state, struct TensorWrapper src, struct TensorWrapper sum);

extern "C"
struct TensorWrapper mulSpectrums(
        struct THCState *state, struct TensorWrapper src1, struct TensorWrapper src2,
        struct TensorWrapper dst, int flags, bool conjB);

extern "C"
struct TensorWrapper mulAndScaleSpectrums(
        struct THCState *state, struct TensorWrapper src1, struct TensorWrapper src2,
        struct TensorWrapper dst, int flags, float scale, bool conjB);

extern "C"
struct TensorWrapper dft(
        struct THCState *state, struct TensorWrapper src,
        struct TensorWrapper dst, struct SizeWrapper dft_size, int flags);

extern "C"
struct ConvolutionPtr Convolution_ctor(
        struct THCState *state, struct SizeWrapper user_block_size);

extern "C"
struct TensorWrapper Convolution_convolve(
        struct THCState *state, struct ConvolutionPtr ptr, struct TensorWrapper image,
        struct TensorWrapper templ, struct TensorWrapper result, bool ccor);