#include <photo.hpp>

extern "C" struct TensorWrapper inpaint(struct TensorWrapper src, struct TensorWrapper inpaintMask,
                                    struct TensorWrapper dst, double inpaintRadius, int flags)
{
    MatT dst_mat = dst.toMatT();
    cv::inpaint(src.toMat(), inpaintMask.toMat(), dst_mat, inpaintRadius, flags);
    return TensorWrapper(dst_mat);
}

extern "C" struct TensorWrapper fastNlMeansDenoising1(struct TensorWrapper src, struct TensorWrapper dst,
                                    float h, int templateWindowSize, int searchWindowSize)
{
    MatT dst_mat = dst.toMatT();
    cv::fastNlMeansDenoising(src.toMat(), dst_mat, h, templateWindowSize, searchWindowSize);
    return TensorWrapper(dst_mat);
}

extern "C" struct TensorWrapper fastNlMeansDenoising2(struct TensorWrapper src, struct TensorWrapper dst,
                                    struct TensorWrapper h, int templateWindowSize,
                                    int searchWindowSize, int normType)
{
    MatT dst_mat = dst.toMatT();
    cv::fastNlMeansDenoising(src.toMat(), dst_mat, h.toMat(), templateWindowSize, searchWindowSize, normType);
    return TensorWrapper(dst_mat);
}

extern "C" struct TensorWrapper fastNlMeansDenoisingColored(struct TensorWrapper src, struct TensorWrapper dst,
                                    float h, float hColor, int templateWindowSize, int searchWindowSize)
{
    MatT dst_mat = dst.toMatT();
    cv::fastNlMeansDenoisingColored(src.toMat(), dst_mat, h, hColor, templateWindowSize, searchWindowSize);
    return TensorWrapper(dst_mat);
}

extern "C" struct TensorWrapper fastNlMeansDenoisingMulti1(struct TensorArray srcImgs, struct TensorWrapper dst,
                                    int imgToDenoiseIndex, int temporalWindowSize, float h,
                                    int templateWindowSize, int searchWindowSize)
{
    MatT dst_mat = dst.toMatT();
    cv::fastNlMeansDenoisingMulti(srcImgs.toMatList(), dst_mat, imgToDenoiseIndex, temporalWindowSize, h,
                                    templateWindowSize, searchWindowSize);
    return TensorWrapper(dst_mat);
}

extern "C" struct TensorWrapper fastNlMeansDenoisingMulti2(struct TensorArray srcImgs, struct TensorWrapper dst,
                                    int imgToDenoiseIndex, int temporalWindowSize, struct TensorWrapper h,
                                    int templateWindowSize, int searchWindowSize, int normType)
{
    MatT dst_mat = dst.toMatT();
    cv::fastNlMeansDenoisingMulti(srcImgs.toMatList(), dst_mat, imgToDenoiseIndex, temporalWindowSize, h.toMat(),
                                        templateWindowSize, searchWindowSize, normType);
    return TensorWrapper(dst_mat);
}

extern "C" struct TensorWrapper fastNlMeansDenoisingColoredMulti(struct TensorArray srcImgs, struct TensorWrapper dst,
                                    int imgToDenoiseIndex, int temporalWindowSize, float h,
                                    float hColor, int templateWindowSize, int searchWindowSize)
{
    MatT dst_mat = dst.toMatT();
    cv::fastNlMeansDenoisingColoredMulti(srcImgs.toMatList(), dst_mat, imgToDenoiseIndex, temporalWindowSize, h,
                                    hColor, templateWindowSize, searchWindowSize);
    return TensorWrapper(dst_mat);
}

extern "C" struct TensorWrapper denoise_TVL1(struct TensorArray observations, struct TensorWrapper result,
                                    double lambda, int niters) {
    cv::Mat result_mat;
    if (result.isNull()) {
        cv::denoise_TVL1(observations.toMatList(), result_mat, lambda, niters);
        return TensorWrapper(MatT(result_mat));
    } else {
        result_mat = result.toMat();
        cv::denoise_TVL1(observations.toMatList(), result_mat, lambda, niters);
    }
    return result;
}

extern "C" struct TensorArray decolor(struct TensorWrapper src, struct TensorWrapper grayscale,
                                    struct TensorWrapper color_boost)
{
    std::vector<MatT> retval(2);
    retval[0] = grayscale.toMatT();
    retval[1] = color_boost.toMatT();
    cv::decolor(src.toMat(), retval[0], retval[1]);
    return TensorArray(retval);
}

extern "C" struct TensorWrapper seamlessClone(struct TensorWrapper src, struct TensorWrapper dst,
                                    struct TensorWrapper mask, struct PointWrapper p,
                                    struct TensorWrapper blend, int flags)
{
    MatT blend_mat = blend.toMatT();
    cv::seamlessClone(src.toMat(), blend_mat, mask.toMat(), p, blend.toMat(), flags);
    return TensorWrapper(blend_mat);
}

extern "C" struct TensorWrapper colorChange(struct TensorWrapper src, struct TensorWrapper mask,
                                    struct TensorWrapper dst, float red_mul,
                                    float green_mul, float blue_mul)
{
    MatT dst_mat = dst.toMatT();
    cv::colorChange(src.toMat(), mask.toMat(), dst_mat, red_mul, green_mul, blue_mul);
    return TensorWrapper(dst_mat);
}

extern "C" struct TensorWrapper illuminationChange(struct TensorWrapper src, struct TensorWrapper mask,
                                    struct TensorWrapper dst, float alpha, float beta)
{
    MatT dst_mat = dst.toMatT();
    cv::illuminationChange(src.toMat(), mask.toMat(), dst_mat, alpha, beta);
    return TensorWrapper(dst_mat);
}

extern "C" struct TensorWrapper textureFlattening(struct TensorWrapper src, struct TensorWrapper mask,
                                    struct TensorWrapper dst, float low_threshold, float high_threshold,
                                    int kernel_size)
{
    MatT dst_mat = dst.toMatT();
    cv::textureFlattening(src.toMat(), mask.toMat(), dst_mat, low_threshold,
                            high_threshold, kernel_size);
    return TensorWrapper(dst_mat);
}

extern "C" struct TensorWrapper edgePreservingFilter(struct TensorWrapper src, struct TensorWrapper dst,
                                    int flags, float sigma_s, float sigma_r)
{
    MatT dst_mat = dst.toMatT();
    cv::edgePreservingFilter(src.toMat(), dst_mat, flags, sigma_s, sigma_r);
    return TensorWrapper(dst_mat);
}

extern "C" struct TensorWrapper detailEnhance(struct TensorWrapper src, struct TensorWrapper dst,
                                    float sigma_s, float sigma_r)
{
    MatT dst_mat = dst.toMatT();
    cv::detailEnhance(src.toMat(), dst_mat, sigma_s, sigma_r);
    return TensorWrapper(dst_mat);
}

extern "C" struct TensorArray pencilSketch(struct TensorWrapper src, struct TensorWrapper dst1,
                                    struct TensorWrapper dst2, float sigma_s, float sigma_r, float shade_factor)
{
    std::vector<MatT> retval(2);
    retval[0] = dst1.toMatT();
    retval[1] = dst2.toMatT();
    cv::pencilSketch(src.toMat(), retval[1], retval[2], sigma_s, sigma_r, shade_factor);
    return TensorArray(retval);
}

extern "C" struct TensorWrapper stylization(struct TensorWrapper src, struct TensorWrapper dst,
                                    float sigma_s, float sigma_r)
{
    MatT dst_mat = dst.toMatT();
    cv::stylization(src.toMat(), dst_mat, sigma_s, sigma_r);
    return TensorWrapper(dst_mat);
}

/****************** Classes ******************/

// Tonemap

extern "C" struct TonemapPtr Tonemap_ctor(float gamma)
{
    return rescueObjectFromPtr(cv::createTonemap(gamma));
}

extern "C" struct TensorWrapper Tonemap_process(struct TonemapPtr ptr, struct TensorArray src, struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    ptr->process(src.toMatList(), dst_mat);
    return TensorWrapper(dst_mat);
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
    std::vector<cv::Mat> dst_vec;
    if(dst.isNull()) {
        ptr->process(src.toMatList(), dst_vec, times.toMat(), response.toMat());
        std::vector<MatT> result = get_vec_MatT(dst_vec);
        return TensorArray(result);
    } else {
        dst_vec = dst.toMatList();
        ptr->process(src.toMatList(), dst_vec, times.toMat(), response.toMat());
    }
    return dst_vec;
}

// AlignMTB

extern "C" struct AlignMTBPtr AlignMTB_ctor(int max_bits, int exclude_range, bool cut)
{
    return rescueObjectFromPtr(cv::createAlignMTB(max_bits, exclude_range, cut));
}

extern "C" struct TensorArray AlignMTB_process1(struct AlignMTBPtr ptr, struct TensorArray src, struct TensorArray dst,
                            struct TensorWrapper times, struct TensorWrapper response)
{
    std::vector<cv::Mat> dst_vec;
    if (dst.isNull()) {
        ptr->process(src.toMatList(), dst_vec, times.toMat(), response.toMat());
        std::vector<MatT> result = get_vec_MatT(dst_vec);
        return TensorArray(result);
    } else {
        dst_vec = dst.toMatList();
        ptr->process(src.toMatList(), dst_vec, times.toMat(), response.toMat());
    }
    return dst;
}

extern "C" struct TensorArray AlignMTB_process2(struct AlignMTBPtr ptr, struct TensorArray src, struct TensorArray dst)
{
    std::vector<cv::Mat> dst_vec;
    if (dst.isNull()) {
        ptr->process(src.toMatList(), dst_vec);
        std::vector<MatT> result = get_vec_MatT(dst_vec);
        return TensorArray(result);
    } else {
        dst_vec = dst.toMatList();
        ptr->process(src.toMatList(), dst_vec);
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
    MatT dst_mat = dst.toMatT();
    ptr->shiftMat(src.toMat(), dst_mat, shift);
    return TensorWrapper(dst_mat);
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
    MatT dst_mat = dst.toMatT();
    ptr->process(src.toMatList(), dst_mat, times.toMat());
    return TensorWrapper(dst_mat);
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
    return TensorWrapper(MatT(ptr->getRadiance()));
}

// MergeExposures
extern "C" struct TensorWrapper MergeExposures_process(struct MergeExposuresPtr ptr, struct TensorArray src, struct TensorWrapper dst,
                            struct TensorWrapper times, struct TensorWrapper response)
{
    MatT dst_mat = dst.toMatT();
    ptr->process(src.toMatList(), dst_mat, times.toMat(), response.toMat());
    return TensorWrapper(dst_mat);
}

// MergeDebevec

extern "C" struct MergeDebevecPtr MergeDebevec_ctor()
{
    return rescueObjectFromPtr(cv::createMergeDebevec());
}

extern "C" struct TensorWrapper MergeDebevec_process1(struct MergeDebevecPtr ptr, struct TensorArray src, struct TensorWrapper dst,
                            struct TensorWrapper times, TensorWrapper response)
{
    MatT dst_mat = dst.toMatT();
    ptr->process(src.toMatList(), dst_mat, times.toMat(), response.toMat());
    return TensorWrapper(dst_mat);
}

extern "C" struct TensorWrapper MergeDebevec_process2(struct MergeDebevecPtr ptr, struct TensorArray src, struct TensorWrapper dst,
                            struct TensorWrapper times)
{
    MatT dst_mat = dst.toMatT();
    ptr->process(src.toMatList(), dst_mat, times.toMat());
    return TensorWrapper(dst_mat);
}

// MergeMertens

extern "C" struct MergeMertensPtr MergeMertens_ctor(float contrast_weight, float saturation_weight, float exposure_weight)
{
    return rescueObjectFromPtr(cv::createMergeMertens(contrast_weight, saturation_weight, exposure_weight));
}

extern "C" struct TensorWrapper MergeMertens_process1(struct MergeMertensPtr ptr, struct TensorArray src, struct TensorWrapper dst,
                            struct TensorWrapper times, struct TensorWrapper response)
{
    MatT dst_mat = dst.toMatT();
    ptr->process(src.toMatList(), dst_mat, times.toMat(), response.toMat());
    return TensorWrapper(dst_mat);
}

extern "C" struct TensorWrapper MergeMertens_process2(struct MergeMertensPtr ptr, struct TensorArray src, struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    ptr->process(src.toMatList(), dst_mat);
    return TensorWrapper(dst_mat);
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

// MergeRobertson

extern "C" struct MergeRobertsonPtr MergeRobertson_ctor()
{
    return rescueObjectFromPtr(cv::createMergeRobertson());
}

extern "C" struct TensorWrapper MergeRobertson_process1(struct MergeRobertsonPtr ptr, struct TensorArray src, struct TensorWrapper dst,
                            struct TensorWrapper times, struct TensorWrapper response)
{
    MatT dst_mat = dst.toMatT();
        ptr->process(src.toMatList(), dst_mat, times.toMat(), response.toMat());
    return TensorWrapper(dst_mat);
}

extern "C" struct TensorWrapper MergeRobertson_process2(struct MergeRobertsonPtr ptr, struct TensorArray src,
                            struct TensorWrapper dst, struct TensorWrapper times)
{
    MatT dst_mat = dst.toMatT();
    ptr->process(src.toMatList(), dst_mat, times.toMat());
    return TensorWrapper(dst_mat);
}
