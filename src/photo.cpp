#include <photo.hpp>

extern "C" struct TensorWrapper inpaint(struct TensorWrapper src, struct TensorWrapper inpaintMask,
                                    struct TensorWrapper dst, double inpaintRadius, int flags)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::inpaint(src.toMat(), inpaintMask.toMat(), retval, inpaintRadius, flags);
        return TensorWrapper(retval);
    } else {
        cv::inpaint(src.toMat(), inpaintMask.toMat(), dst.toMat(), inpaintRadius, flags);
    }
    return dst;
}

extern "C" struct TensorWrapper fastNlMeansDenoisingCommon(struct TensorWrapper src, struct TensorWrapper dst,
                                    struct FloatArray h, int templateWindowSize,
                                    int searchWindowSize, int normType)
{
    if (normType == -1) {
        if (dst.isNull()) {
            cv::Mat retval;
            cv::fastNlMeansDenoising(src.toMat(), retval, *(h.data), templateWindowSize, searchWindowSize);
            return TensorWrapper(retval);
        } else {
            cv::fastNlMeansDenoising(src.toMat(), dst.toMat(), *(h.data), templateWindowSize, searchWindowSize);
            return dst;
        }
    }
    std::vector<float> hvec;
    hvec = h.toFloatList(hvec);

    if (dst.isNull()) {
        cv::Mat retval;
        cv::fastNlMeansDenoising(src.toMat(), retval, hvec, templateWindowSize, searchWindowSize, normType);
        return TensorWrapper(retval);
    } else {
        cv::fastNlMeansDenoising(src.toMat(), dst.toMat(), hvec, templateWindowSize, searchWindowSize, normType);
    }
    return dst;
}

extern "C" struct TensorWrapper fastNlMeansDenoisingColored(struct TensorWrapper src, struct TensorWrapper dst,
                                    float h, float hColor, int templateWindowSize, int searchWindowSize)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::fastNlMeansDenoisingColored(src.toMat(), retval, h, hColor, templateWindowSize, searchWindowSize);
        return TensorWrapper(retval);
    } else {
        cv::fastNlMeansDenoisingColored(src.toMat(), dst.toMat(), h, hColor, templateWindowSize, searchWindowSize);
    }
    return dst;
}

extern "C" struct TensorWrapper fastNlMeansDenoisingMultiCommon(struct TensorArray srcImgs, struct TensorWrapper dst,
                                    int imgToDenoiseIndex, int temporalWindowSize, struct FloatArray h,
                                    int templateWindowSize, int searchWindowSize, int normType)
{
    if (normType == -1) {
        if (dst.isNull()) {
            cv::Mat retval;
            cv::fastNlMeansDenoisingMulti(srcImgs.toMatList(), retval, imgToDenoiseIndex, temporalWindowSize, *(h.data),
                                        templateWindowSize, searchWindowSize);
        
            return TensorWrapper(retval);
        } else {
            cv::fastNlMeansDenoisingMulti(srcImgs.toMatList(), dst.toMat(), imgToDenoiseIndex, temporalWindowSize, *(h.data),
                                        templateWindowSize, searchWindowSize);
            return dst;
        }
    }
    std::vector<float> hvec;
    hvec = h.toFloatList(hvec);

    if (dst.isNull()) {
        cv::Mat retval;
        cv::fastNlMeansDenoisingMulti(srcImgs.toMatList(), retval, imgToDenoiseIndex, temporalWindowSize, hvec,
                                        templateWindowSize, searchWindowSize, normType);
        return TensorWrapper(retval);
    } else {
        cv::fastNlMeansDenoisingMulti(srcImgs.toMatList(), dst.toMat(), imgToDenoiseIndex, temporalWindowSize, hvec,
                                        templateWindowSize, searchWindowSize, normType);
    }
    return dst;
}

extern "C" struct TensorWrapper fastNlMeansDenoisingColoredMulti(struct TensorArray srcImgs, struct TensorWrapper dst,
                                    int imgToDenoiseIndex, int temporalWindowSize, float h,
                                    float hColor, int templateWindowSize, int searchWindowSize)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::fastNlMeansDenoisingColoredMulti(srcImgs.toMatList(), retval, imgToDenoiseIndex, temporalWindowSize, h,
                                    hColor, templateWindowSize, searchWindowSize);
    
        return TensorWrapper(retval);
    } else {
        cv::fastNlMeansDenoisingColoredMulti(srcImgs.toMatList(), dst.toMat(), imgToDenoiseIndex, temporalWindowSize, h,
                                    hColor, templateWindowSize, searchWindowSize);
    }
    return dst;
}

// Some frightening functions with patterns and structures

extern "C" struct TensorWrapper decolor(struct TensorWrapper src, struct TensorWrapper grayscale,
                                    struct TensorWrapper color_boost)
{
    if (grayscale.isNull()) {
        cv::Mat retval;
        cv::decolor(src.toMat(), retval, color_boost.toMat());
        return TensorWrapper(retval);
    } else {
        cv::decolor(src.toMat(), grayscale.toMat(), color_boost.toMat());
    }
    return grayscale;
}

extern "C" struct TensorWrapper seamlessClone(struct TensorWrapper src, struct TensorWrapper dst,
                                    struct TensorWrapper mask, struct PointWrapper p,
                                    struct TensorWrapper blend, int flags)
{
    if (blend.isNull()) {
        cv::Mat retval;
        cv::seamlessClone(src.toMat(), dst.toMat(), mask.toMat(), p, retval, flags);
        return TensorWrapper(retval);
    } else {
        cv::seamlessClone(src.toMat(), dst.toMat(), mask.toMat(), p, blend.toMat(), flags);
    }
    return dst;
}

extern "C" struct TensorWrapper colorChange(struct TensorWrapper src, struct TensorWrapper mask,
                                    struct TensorWrapper dst, float red_mul,
                                    float green_mul, float blue_mul)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::colorChange(src.toMat(), mask.toMat(), retval, red_mul, green_mul, blue_mul);
    
        return TensorWrapper(retval);
    } else {
        cv::colorChange(src.toMat(), mask.toMat(), dst.toMat(), red_mul, green_mul, blue_mul);
    }
    return dst;
}

extern "C" struct TensorWrapper illuminationChange(struct TensorWrapper src, struct TensorWrapper mask,
                                    struct TensorWrapper dst, float alpha, float beta)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::illuminationChange(src.toMat(), mask.toMat(), retval, alpha, beta);
    
        return TensorWrapper(retval);
    } else {
        cv::illuminationChange(src.toMat(), mask.toMat(), dst.toMat(), alpha, beta);
    }
    return dst;
}

extern "C" struct TensorWrapper textureFlattening(struct TensorWrapper src, struct TensorWrapper mask,
                                    struct TensorWrapper dst, float low_threshold, float high_threshold,
                                    int kernel_size)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::textureFlattening(src.toMat(), mask.toMat(), retval, low_threshold,
                            high_threshold, kernel_size);
    
        return TensorWrapper(retval);
    } else {
        cv::textureFlattening(src.toMat(), mask.toMat(), dst.toMat(), low_threshold,
                            high_threshold, kernel_size);
    }
    return dst;
}

extern "C" struct TensorWrapper edgePreservingFilter(struct TensorWrapper src, struct TensorWrapper dst,
                                    int flags, float sigma_s, float sigma_r)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::edgePreservingFilter(src.toMat(), retval, flags, sigma_s, sigma_r);
    
        return TensorWrapper(retval);
    } else {
        cv::edgePreservingFilter(src.toMat(), dst.toMat(), flags, sigma_s, sigma_r);
    }
    return dst;
}

extern "C" struct TensorWrapper detailEnhance(struct TensorWrapper src, struct TensorWrapper dst,
                                    float sigma_s, float sigma_r)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::detailEnhance(src.toMat(), retval, sigma_s, sigma_r);
    
        return TensorWrapper(retval);
    } else {
        cv::detailEnhance(src.toMat(), dst.toMat(), sigma_s, sigma_r);
    }
    return dst;
}

extern "C" struct TensorWrapper pencilSketch(struct TensorWrapper src, struct TensorWrapper dst1,
                                    struct TensorWrapper dst2, float sigma_s, float sigma_r, float shade_factor)
{
    cv::Mat retval1;
    cv::Mat retval2;

    if (!dst1.isNull()) {
        retval1 = dst1.toMat();
    }
    if (!dst2.isNull()) {
        retval2 = dst2.toMat();
    }
    cv::pencilSketch(src.toMat(), retval1, retval2, sigma_s, sigma_r, shade_factor);
    return TensorWrapper(retval2);
}

extern "C" struct TensorWrapper stylization(struct TensorWrapper src, struct TensorWrapper dst,
                                    float sigma_s, float sigma_r)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::stylization(src.toMat(), retval, sigma_s, sigma_r);
    
        return TensorWrapper(retval);
    } else {
        cv::stylization(src.toMat(), dst.toMat(), sigma_s, sigma_r);
    }
    return dst;
}