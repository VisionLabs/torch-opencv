#include <photo.hpp>

extern "C"
struct TensorWrapper inpaint(struct TensorWrapper src, struct TensorWrapper inpaintMask,
                            struct TensorWrapper dst, double inpaintRadius, int flags)
{
    cv::inpaint(src.toMat(), inpaintMask.toMat(), dst.toMat(), inpaintRadius, flags);
    return dst;
}