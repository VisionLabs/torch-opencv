#include <cudaarithm.hpp>
#include <opencv2/highgui.hpp>

extern "C"
struct TensorWrapper min(
        struct THCState *state, struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper dst)
{
    if (dst.isNull()) {
        cuda::GpuMat result;
        cuda::min(src1.toGpuMat(), src2.toGpuMat(), result);
        return TensorWrapper(result, state);
    } else {
        cuda::min(src1.toGpuMat(), src2.toGpuMat(), dst.toGpuMat());
        return dst;
    }
}

extern "C"
struct TensorWrapper max(
        struct THCState *state, struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper dst)
{
    if (dst.isNull()) {
        cuda::GpuMat result;
        cuda::max(src1.toGpuMat(), src2.toGpuMat(), result);
        return TensorWrapper(result, state);
    } else {
        cuda::max(src1.toGpuMat(), src2.toGpuMat(), dst.toGpuMat());
        return dst;
    }
}

extern "C"
struct TensorPlusDouble threshold(
        struct THCState *state, struct TensorWrapper src, 
        struct TensorWrapper dst, double thresh, double maxval, int type)
{
    TensorPlusDouble retval;

    if (dst.isNull()) {
        cuda::GpuMat result;
        retval.val = cuda::threshold(src.toGpuMat(), result, thresh, maxval, type);
        new (&retval.tensor) TensorWrapper(result, state);
    } else {
        retval.val = cuda::threshold(src.toGpuMat(), dst.toGpuMat(), thresh, maxval, type);
        retval.tensor = dst;
    }
    return retval;
}

extern "C"
struct TensorWrapper magnitude(
        struct THCState *state, struct TensorWrapper xy, struct TensorWrapper magnitude)
{
    if (magnitude.isNull()) {
        cuda::GpuMat result;
        cuda::magnitude(xy.toGpuMat(), result);
        return TensorWrapper(result, state);
    } else {
        cuda::magnitude(xy.toGpuMat(), magnitude.toGpuMat());
        return magnitude;
    }
}

extern "C"
struct TensorWrapper magnitudeSqr(
        struct THCState *state, struct TensorWrapper xy, struct TensorWrapper magnitude)
{
    if (magnitude.isNull()) {
        cuda::GpuMat result;
        cuda::magnitudeSqr(xy.toGpuMat(), result);
        return TensorWrapper(result, state);
    } else {
        cuda::magnitudeSqr(xy.toGpuMat(), magnitude.toGpuMat());
        return magnitude;
    }
}

extern "C"
struct TensorWrapper magnitude2(
        struct THCState *state, struct TensorWrapper x, struct TensorWrapper y, struct TensorWrapper magnitude)
{
    if (magnitude.isNull()) {
        cuda::GpuMat result;
        cuda::magnitude(x.toGpuMat(), y.toGpuMat(), result);
        return TensorWrapper(result, state);
    } else {
        cuda::magnitude(x.toGpuMat(), y.toGpuMat(), magnitude.toGpuMat());
        return magnitude;
    }
}

extern "C"
struct TensorWrapper magnitudeSqr2(
        struct THCState *state, struct TensorWrapper x, struct TensorWrapper y, struct TensorWrapper magnitudeSqr)
{
    if (magnitudeSqr.isNull()) {
        cuda::GpuMat result;
        cuda::magnitudeSqr(x.toGpuMat(), y.toGpuMat(), result);
        return TensorWrapper(result, state);
    } else {
        cuda::magnitudeSqr(x.toGpuMat(), y.toGpuMat(), magnitudeSqr.toGpuMat());
        return magnitudeSqr;
    }
}

extern "C"
struct TensorWrapper phase(
        struct THCState *state, struct TensorWrapper x, struct TensorWrapper y,
        struct TensorWrapper angle, bool angleInDegrees)
{
    if (angle.isNull()) {
        cuda::GpuMat result;
        cuda::phase(x.toGpuMat(), y.toGpuMat(), result, angleInDegrees);
        return TensorWrapper(result, state);
    } else {
        cuda::phase(x.toGpuMat(), y.toGpuMat(), angle.toGpuMat(), angleInDegrees);
        return angle;
    }
}

extern "C"
struct TensorArray cartToPolar(
        struct THCState *state, struct TensorWrapper x, struct TensorWrapper y,
        struct TensorWrapper magnitude, struct TensorWrapper angle, bool angleInDegrees)
{
    std::vector<cuda::GpuMat> result(2);
    if (!magnitude.isNull()) result[0] = magnitude.toGpuMat();
    if (!angle.isNull())     result[1] = angle.toGpuMat();

    cuda::cartToPolar(x.toGpuMat(), y.toGpuMat(), result[0], result[1], angleInDegrees);

    return TensorArray(result, state);
}

extern "C"
struct TensorArray polarToCart(
        struct THCState *state, struct TensorWrapper magnitude, struct TensorWrapper angle,
        struct TensorWrapper x, struct TensorWrapper y, bool angleInDegrees)
{
    std::vector<cuda::GpuMat> result;
    if (!x.isNull()) result[0] = x.toGpuMat();
    if (!y.isNull()) result[1] = y.toGpuMat();

    cuda::polarToCart(magnitude.toGpuMat(), angle.toGpuMat(), result[0], result[1], angleInDegrees);

    return TensorArray(result, state);
}

extern "C"
struct LookUpTablePtr LookUpTable_ctor(
        struct THCState *state, struct TensorWrapper lut)
{
    return rescueObjectFromPtr(cuda::createLookUpTable(lut.toGpuMat()));
}

extern "C"
struct TensorWrapper LookUpTable_transform(
        struct THCState *state, struct LookUpTablePtr ptr,
        struct TensorWrapper src, struct TensorWrapper dst)
{
    if (dst.isNull()) {
        cuda::GpuMat result;
        ptr->transform(src.toGpuMat(), result);
        return TensorWrapper(result, state);
    } else {
        ptr->transform(src.toGpuMat(), dst.toGpuMat());
        return dst;
    }
}

extern "C"
struct TensorWrapper rectStdDev(
        struct THCState *state, struct TensorWrapper src, struct TensorWrapper sqr,
        struct TensorWrapper dst, struct RectWrapper rect)
{
    if (dst.isNull()) {
        cv::Mat result;
        cuda::rectStdDev(src.toGpuMat(), sqr.toGpuMat(), result, rect);
        return TensorWrapper(result);
    } else {
        cuda::rectStdDev(src.toGpuMat(), sqr.toGpuMat(), dst.toGpuMat(), rect);
        return dst;
    }
}

extern "C"
struct TensorWrapper normalize(
        struct THCState *state, struct TensorWrapper src, struct TensorWrapper dst,
        double alpha, double beta, int norm_type, int dtype)
{
    if (dst.isNull()) {
        cv::Mat result;
        cuda::normalize(src.toGpuMat(), result, alpha, beta, norm_type, dtype);
        return TensorWrapper(result);
    } else {
        cuda::normalize(src.toGpuMat(), dst.toGpuMat(), alpha, beta, norm_type, dtype);
        return dst;
    }
}

extern "C"
struct TensorWrapper integral(
        struct THCState *state, struct TensorWrapper src, struct TensorWrapper sum)
{
    if (sum.isNull()) {
        cv::Mat result;
        cuda::integral(src.toGpuMat(), result);
        return TensorWrapper(result);
    } else {
        cuda::integral(src.toGpuMat(), sum.toGpuMat());
        return sum;
    }
}

extern "C"
struct TensorWrapper sqrIntegral(
        struct THCState *state, struct TensorWrapper src, struct TensorWrapper sum)
{
    if (sum.isNull()) {
        cv::Mat result;
        cuda::sqrIntegral(src.toGpuMat(), result);
        return TensorWrapper(result);
    } else {
        cuda::sqrIntegral(src.toGpuMat(), sum.toGpuMat());
        return sum;
    }
}

extern "C"
struct TensorWrapper mulSpectrums(
        struct THCState *state, struct TensorWrapper src1, struct TensorWrapper src2, 
        struct TensorWrapper dst, int flags, bool conjB)
{
    if (dst.isNull()) {
        cuda::GpuMat result;
        cuda::mulSpectrums(src1.toGpuMat(), src2.toGpuMat(), result, flags, conjB);
        return TensorWrapper(result, state);
    } else {
        cuda::mulSpectrums(src1.toGpuMat(), src2.toGpuMat(), dst.toGpuMat(), flags, conjB);
        return dst;
    }
}

extern "C"
struct TensorWrapper mulAndScaleSpectrums(
        struct THCState *state, struct TensorWrapper src1, struct TensorWrapper src2,
        struct TensorWrapper dst, int flags, float scale, bool conjB)
{
    if (dst.isNull()) {
        cuda::GpuMat result;
        cuda::mulAndScaleSpectrums(src1.toGpuMat(), src2.toGpuMat(), result, flags, scale, conjB);
        return TensorWrapper(result, state);
    } else {
        cuda::mulAndScaleSpectrums(src1.toGpuMat(), src2.toGpuMat(), dst.toGpuMat(), flags, scale, conjB);
        return dst;
    }
}

extern "C"
struct TensorWrapper dft(
        struct THCState *state, struct TensorWrapper src, 
        struct TensorWrapper dst, struct SizeWrapper dft_size, int flags)
{
    if (dst.isNull()) {
        cv::Mat result;
        cuda::dft(src.toGpuMat(), result, dft_size, flags);
        return TensorWrapper(result);
    } else {
        cuda::dft(src.toGpuMat(), dst.toGpuMat(), dft_size, flags);
        return dst;
    }
}

extern "C"
struct ConvolutionPtr Convolution_ctor(
        struct THCState *state, struct SizeWrapper user_block_size)
{
    return rescueObjectFromPtr(cuda::createConvolution(user_block_size));
}

extern "C"
struct TensorWrapper Convolution_convolve(
        struct THCState *state, struct ConvolutionPtr ptr, struct TensorWrapper image,
        struct TensorWrapper templ, struct TensorWrapper result, bool ccor)
{
    if (result.isNull()) {
        cuda::GpuMat resultMat;
        ptr->convolve(image.toGpuMat(), templ.toGpuMat(), resultMat, ccor);
        return TensorWrapper(resultMat, state);
    } else {
        ptr->convolve(image.toGpuMat(), templ.toGpuMat(), result.toGpuMat(), ccor);
        return result;
    }
}
