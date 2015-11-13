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

extern "C" struct TensorWrapper fastNlMeansDenoising1(struct TensorWrapper src, struct TensorWrapper dst,
                                    float h, int templateWindowSize, int searchWindowSize)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::fastNlMeansDenoising(src.toMat(), retval, h, templateWindowSize, searchWindowSize);
        return TensorWrapper(retval);
    } else {
        cv::fastNlMeansDenoising(src.toMat(), dst.toMat(), h, templateWindowSize, searchWindowSize);
    }
    return dst;
}

extern "C" struct TensorWrapper fastNlMeansDenoising2(struct TensorWrapper src, struct TensorWrapper dst,
                                    struct TensorWrapper h, int templateWindowSize,
                                    int searchWindowSize, int normType)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::fastNlMeansDenoising(src.toMat(), retval, h.toMat(), templateWindowSize, searchWindowSize, normType);
        return TensorWrapper(retval);
    } else {
        cv::fastNlMeansDenoising(src.toMat(), dst.toMat(), h.toMat(), templateWindowSize, searchWindowSize, normType);
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

extern "C" struct TensorWrapper fastNlMeansDenoisingMulti1(struct TensorArray srcImgs, struct TensorWrapper dst,
                                    int imgToDenoiseIndex, int temporalWindowSize, float h,
                                    int templateWindowSize, int searchWindowSize)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::fastNlMeansDenoisingMulti(srcImgs.toMatList(), retval, imgToDenoiseIndex, temporalWindowSize, h,
                                    templateWindowSize, searchWindowSize);

        return TensorWrapper(retval);
    } else {
        cv::fastNlMeansDenoisingMulti(srcImgs.toMatList(), dst.toMat(), imgToDenoiseIndex, temporalWindowSize, h,
                                    templateWindowSize, searchWindowSize);
    }
    return dst;
}

extern "C" struct TensorWrapper fastNlMeansDenoisingMulti2(struct TensorArray srcImgs, struct TensorWrapper dst,
                                    int imgToDenoiseIndex, int temporalWindowSize, struct TensorWrapper h,
                                    int templateWindowSize, int searchWindowSize, int normType)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::fastNlMeansDenoisingMulti(srcImgs.toMatList(), retval, imgToDenoiseIndex, temporalWindowSize, h.toMat(),
                                        templateWindowSize, searchWindowSize, normType);
        return TensorWrapper(retval);
    } else {
        cv::fastNlMeansDenoisingMulti(srcImgs.toMatList(), dst.toMat(), imgToDenoiseIndex, temporalWindowSize, h.toMat(),
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
        return dst;
    }
}

extern "C" struct TensorWrapper denoise_TVL1(struct TensorArray observations, struct TensorWrapper result,
                                    double lambda, int niters)
{
    if (result.isNull()) {
        cv::Mat retval;
        cv::denoise_TVL1(observations.toMatList(), retval, lambda, niters);
        return TensorWrapper(retval);
    } else {
        cv::Mat retval = result.toMat();
        cv::denoise_TVL1(observations.toMatList(), retval, lambda, niters);
    }
    return result;
}

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

/****************** Classes ******************/

// Tonemap

extern "C" struct TonemapPtr Tonemap_ctor(float gamma)
{
    return rescueObjectFromPtr(cv::createTonemap(gamma));
}

extern "C" struct TensorWrapper Tonemap_process(struct TonemapPtr ptr, struct TensorArray src, struct TensorWrapper dst)
{
    if (dst.isNull()) {
        cv::Mat retval;
        ptr->process(src.toMatList(), retval);
        return TensorWrapper(retval);
    } else {
        ptr->process(src.toMatList(), dst.toMat());
    }
    return dst;
}

extern "C" float Tonemap_getGamma(struct TonemapPtr ptr)
{
    return ptr->getGamma();
}

extern "C" void Tonemap_setGamma(struct TonemapPtr ptr, float gamma)
{
    ptr->setGamma(gamma);
}

// TonemapDrago

extern "C" struct TonemapDragoPtr TonemapDrago_ctor(float gamma, float saturation, float bias)
{
    return rescueObjectFromPtr(cv::createTonemapDrago(gamma, saturation, bias));
}

extern "C" float TonemapDrago_getSaturation(struct TonemapDragoPtr ptr)
{
    return ptr->getSaturation();
}

extern "C" void TonemapDrago_setSaturation(struct TonemapDragoPtr ptr, float saturation)
{
    ptr->setSaturation(saturation);
}

extern "C" float TonemapDrago_getBias(struct TonemapDragoPtr ptr)
{
    return ptr->getBias();
}

extern "C" void TonemapDrago_setBias(struct TonemapDragoPtr ptr, float bias)
{
    ptr->setBias(bias);
}

// TonemapDurand

extern "C" struct TonemapDurandPtr TonemapDurand_ctor(float gamma, float contrast, float saturation, float sigma_space, float sigma_color)
{
    return rescueObjectFromPtr(cv::createTonemapDurand(gamma, contrast, saturation, sigma_space, sigma_color));
}

extern "C" float TonemapDurand_getSaturation(struct TonemapDurandPtr ptr)
{
    return ptr->getSaturation();
}

extern "C" void TonemapDurand_setSaturation(struct TonemapDurandPtr ptr, float saturation)
{
    ptr->setSaturation(saturation);
}
extern "C" float TonemapDurand_getContrast(struct TonemapDurandPtr ptr)
{
    return ptr->getContrast();
}

extern "C" void TonemapDurand_setContrast(struct TonemapDurandPtr ptr, float contrast)
{
    ptr->setContrast(contrast);
}
extern "C" float TonemapDurand_getSigma_space(struct TonemapDurandPtr ptr)
{
    return ptr->getSigmaSpace();
}

extern "C" void TonemapDurand_setSigmaSpace(struct TonemapDurandPtr ptr, float sigma_space)
{
    ptr->setSigmaSpace(sigma_space);
}
extern "C" float TonemapDurand_getSigmaColor(struct TonemapDurandPtr ptr)
{
    return ptr->getSigmaColor();
}

extern "C" void TonemapDurand_setSigmaColor(struct TonemapDurandPtr ptr, float sigma_color)
{
    ptr->setSigmaColor(sigma_color);
}

// TonemapReinhard

extern "C" struct TonemapReinhardPtr TonemapReinhard_ctor(float gamma, float intensity, float light_adapt, float color_adapt)
{
    return rescueObjectFromPtr(cv::createTonemapReinhard(gamma, intensity, light_adapt, color_adapt));
}

extern "C" float TonemapReinhard_getIntensity(struct TonemapReinhardPtr ptr)
{
    return ptr->getIntensity();
}

extern "C" void TonemapReinhard_setIntensity(struct TonemapReinhardPtr ptr, float intensity)
{
    ptr->setIntensity(intensity);
}
extern "C" float TonemapReinhard_getLightAdaptation(struct TonemapReinhardPtr ptr)
{
    return ptr->getLightAdaptation();
}

extern "C" void TonemapReinhard_setLightAdaptation(struct TonemapReinhardPtr ptr, float light_adapt)
{
    ptr->setLightAdaptation(light_adapt);
}
extern "C" float TonemapReinhard_getColorAdaptation(struct TonemapReinhardPtr ptr)
{
    return ptr->getColorAdaptation();
}

extern "C" void TonemapReinhard_setColorAdaptation(struct TonemapReinhardPtr ptr, float color_adapt)
{
    ptr->setColorAdaptation(color_adapt);
}

// TonemapMantiuk

extern "C" struct TonemapMantiukPtr TonemapMantiuk_ctor(float gamma, float scale, float saturation)
{
    return rescueObjectFromPtr(cv::createTonemapMantiuk(gamma, scale, saturation));
}

extern "C" float TonemapMantiuk_getScale(struct TonemapMantiukPtr ptr)
{
    return ptr->getScale();
}

extern "C" void TonemapMantiuk_setScale(struct TonemapMantiukPtr ptr, float scale)
{
    ptr->setScale(scale);
}
extern "C" float TonemapMantiuk_getSaturation(struct TonemapMantiukPtr ptr)
{
    return ptr->getSaturation();
}

extern "C" void TonemapMantiuk_setSaturation(struct TonemapMantiukPtr ptr, float saturation)
{
    ptr->setSaturation(saturation);
}

// AlignExposures

struct TensorArray AlignExposures_process(struct AlignExposuresPtr ptr, struct TensorArray src, struct TensorArray dst,
                        struct TensorWrapper times, struct TensorWrapper response)
{
    if (dst.isNull()) {
        auto retval = new std::vector<cv::Mat>(src);
        ptr->process(src.toMatList(), *retval, times.toMat(), response.toMat());

        return TensorArray(*retval);
    } else {
        auto retval = new std::vector<cv::Mat>(dst);
        ptr->process(src.toMatList(), *retval, times.toMat(), response.toMat());
        dst = *retval;
    }
    return dst;
}

// AlignMTB

extern "C" struct AlignMTBPtr AlignMTB_ctor(int max_bits, int exclude_range, bool cut)
{
    return rescueObjectFromPtr(cv::createAlignMTB(max_bits, exclude_range, cut));
}

extern "C" struct TensorArray AlignMTB_process1(struct AlignMTBPtr ptr, struct TensorArray src, struct TensorArray dst,
                            struct TensorWrapper times, struct TensorWrapper response)
{
    if (dst.isNull()) {
        auto retval = new std::vector<cv::Mat>(src);
        ptr->process(src.toMatList(), *retval, times.toMat(), response.toMat());

        return TensorArray(*retval);
    } else {
        auto retval = new std::vector<cv::Mat>(dst);
        ptr->process(src.toMatList(), *retval, times.toMat(), response.toMat());
        dst = *retval;
    }
    return dst;
}

extern "C" struct TensorArray AlignMTB_process2(struct AlignMTBPtr ptr, struct TensorArray src, struct TensorArray dst)
{
    if (dst.isNull()) {
        auto retval = new std::vector<cv::Mat>(src);
        ptr->process(src.toMatList(), *retval);

        return TensorArray(*retval);
    } else {
        auto retval = new std::vector<cv::Mat>(dst);
        ptr->process(src.toMatList(), *retval);
        dst = *retval;
    }
    return dst;
}

extern "C" struct PointWrapper AlignMTB_calculateShift(struct AlignMTBPtr ptr, struct TensorWrapper img0, struct TensorWrapper img1)
{
    return ptr->calculateShift(img0.toMat(), img1.toMat());
}

extern "C" struct TensorWrapper AlignMTB_shiftMat(struct AlignMTBPtr ptr, struct TensorWrapper src,
                            struct TensorWrapper dst, struct PointWrapper shift)
{
    if (dst.isNull()) {
        cv::Mat retval;
        ptr->shiftMat(src.toMat(), retval, shift);
        return TensorWrapper(retval);
    } else {
        ptr->shiftMat(src.toMat(), dst.toMat(), shift);
    }
    return dst;
}

extern "C" void AlignMTB_computeBitmaps(struct AlignMTBPtr ptr, struct TensorWrapper img,
                            struct TensorWrapper tb, struct TensorWrapper eb)
{
    ptr->computeBitmaps(img.toMat(), tb.toMat(), eb.toMat());
}

extern "C" int AlignMTB_getMaxBits(struct AlignMTBPtr ptr)
{
    return ptr->getMaxBits();
}

extern "C" void AlignMTB_setMaxBits(struct AlignMTBPtr ptr, int max_bits)
{
    ptr->setMaxBits(max_bits);
}
extern "C" int AlignMTB_getExcludeRange(struct AlignMTBPtr ptr)
{
    return ptr->getExcludeRange();
}

extern "C" void AlignMTB_setExcludeRange(struct AlignMTBPtr ptr, int exclude_range)
{
    ptr->setExcludeRange(exclude_range);
}
extern "C" bool AlignMTB_getCut(struct AlignMTBPtr ptr)
{
    return ptr->getCut();
}

extern "C" void AlignMTB_setCut(struct AlignMTBPtr ptr, bool cut)
{
    ptr->setCut(cut);
}

// CalibreCRF

extern "C" struct TensorWrapper CalibrateCRF_process(struct CalibrateCRFPtr ptr, struct TensorArray src, struct TensorWrapper dst,
                            struct TensorWrapper times)
{
    if (dst.isNull()) {
        cv::Mat retval;
        ptr->process(src.toMatList(), retval, times.toMat());

        return TensorWrapper(retval);
    } else {
        ptr->process(src.toMatList(), dst.toMat(), times.toMat());
    }
    return dst;
}

// CalibrateDebevec

extern "C" struct CalibrateDebevecPtr CalibrateDebevec_ctor(int samples, float lambda, bool random)
{
    return rescueObjectFromPtr(cv::createCalibrateDebevec(samples, lambda, random));
}

extern "C" float CalibrateDebevec_getLambda(struct CalibrateDebevecPtr ptr)
{
    return ptr->getLambda();
}

extern "C" void CalibrateDebevec_setLambda(struct CalibrateDebevecPtr ptr, float lambda)
{
    ptr->setLambda(lambda);
}
extern "C" int CalibrateDebevec_getSamples(struct CalibrateDebevecPtr ptr)
{
    return ptr->getSamples();
}

extern "C" void CalibrateDebevec_setSamples(struct CalibrateDebevecPtr ptr, int samples)
{
    ptr->setSamples(samples);
}
extern "C" bool CalibrateDebevec_getRandom(struct CalibrateDebevecPtr ptr)
{
    return ptr->getRandom();
}

extern "C" void CalibrateDebevec_setRandom(struct CalibrateDebevecPtr ptr, bool random)
{
    ptr->setRandom(random);
}

// CalibrateRobertson

extern "C" struct CalibrateRobertsonPtr CalibrateRobertson_ctor(int max_iter, float threshold)
{
    return rescueObjectFromPtr(cv::createCalibrateRobertson(max_iter, threshold));
}

extern "C" int CalibrateRobertson_getMaxIter(struct CalibrateRobertsonPtr ptr)
{
    return ptr->getMaxIter();
}

extern "C" void CalibrateRobertson_setMaxIter(struct CalibrateRobertsonPtr ptr, int max_iter)
{
    ptr->setMaxIter(max_iter);
}
extern "C" float CalibrateRobertson_getThreshold(struct CalibrateRobertsonPtr ptr)
{
    return ptr->getThreshold();
}

extern "C" void CalibrateRobertson_setThreshold(struct CalibrateRobertsonPtr ptr, float threshold)
{
    ptr->setThreshold(threshold);
}

extern "C" struct TensorWrapper CalibrateRobertson_getRadiance(struct CalibrateRobertsonPtr ptr)
{
    return TensorWrapper(ptr->getRadiance());
}

// MergeExposures
extern "C" struct TensorWrapper MergeExposures_process(struct MergeExposuresPtr ptr, struct TensorArray src, struct TensorWrapper dst,
                            struct TensorWrapper times, struct TensorWrapper response)
{
    if (dst.isNull()) {
        cv::Mat retval;
        ptr->process(src.toMatList(), retval, times.toMat(), response.toMat());

        return TensorWrapper(retval);
    } else {
        ptr->process(src.toMatList(), dst.toMat(), times.toMat(), response.toMat());
    }
    return dst;
}

// MergeDebevec

extern "C" struct MergeDebevecPtr MergeDebevec_ctor()
{
    return rescueObjectFromPtr(cv::createMergeDebevec());
}

extern "C" struct TensorWrapper MergeDebevec_process1(struct MergeDebevecPtr ptr, struct TensorArray src, struct TensorWrapper dst,
                            struct TensorWrapper times, TensorWrapper response)
{
    if (dst.isNull()) {
        cv::Mat retval;
        ptr->process(src.toMatList(), retval, times.toMat(), response.toMat());

        return TensorWrapper(retval);
    } else {
        ptr->process(src.toMatList(), dst.toMat(), times.toMat(), response.toMat());
    }
    return dst;
}

extern "C" struct TensorWrapper MergeDebevec_process2(struct MergeDebevecPtr ptr, struct TensorArray src, struct TensorWrapper dst,
                            struct TensorWrapper times)
{
    if (dst.isNull()) {
        cv::Mat retval;
        ptr->process(src.toMatList(), retval, times.toMat());

        return TensorWrapper(retval);
    } else {
        ptr->process(src.toMatList(), dst.toMat(), times.toMat());
    }
    return dst;
}

// MergeMertens

extern "C" struct MergeMertensPtr MergeMertens_ctor(float contrast_weight, float saturation_weight, float exposure_weight)
{
    return rescueObjectFromPtr(cv::createMergeMertens(contrast_weight, saturation_weight, exposure_weight));
}

extern "C" struct TensorWrapper MergeMertens_process1(struct MergeMertensPtr ptr, struct TensorArray src, struct TensorWrapper dst,
                            struct TensorWrapper times, struct TensorWrapper response)
{
    if (dst.isNull()) {
        cv::Mat retval;
        ptr->process(src.toMatList(), retval, times.toMat(), response.toMat());

        return TensorWrapper(retval);
    } else {
        ptr->process(src.toMatList(), dst.toMat(), times.toMat(), response.toMat());
    }
    return dst;
}

extern "C" struct TensorWrapper MergeMertens_process2(struct MergeMertensPtr ptr, struct TensorArray src, struct TensorWrapper dst)
{
    if (dst.isNull()) {
        cv::Mat retval;
        ptr->process(src.toMatList(), retval);

        return TensorWrapper(retval);
    } else {
        ptr->process(src.toMatList(), dst.toMat());
    }
    return dst;
}

extern "C" float MergeMertens_getContrastWeight(struct MergeMertensPtr ptr)
{
    return ptr->getContrastWeight();
}

extern "C" void MergeMertens_setContrastWeight(struct MergeMertensPtr ptr, float contrast_weight)
{
    ptr->setContrastWeight(contrast_weight);
}
extern "C" float MergeMertens_getSaturationWeight(struct MergeMertensPtr ptr)
{
    return ptr->getSaturationWeight();
}

extern "C" void MergeMertens_setSaturationWeight(struct MergeMertensPtr ptr, float saturation_weight)
{
    ptr->setSaturationWeight(saturation_weight);
}
extern "C" float MergeMertens_getExposureWeight(struct MergeMertensPtr ptr)
{
    return ptr->getExposureWeight();
}

extern "C" void MergeMertens_setExposureWeight(struct MergeMertensPtr ptr, float exposure_weight)
{
    ptr->setExposureWeight(exposure_weight);
}