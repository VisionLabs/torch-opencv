#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/photo.hpp>

extern "C" struct TensorWrapper inpaint(struct TensorWrapper src, struct TensorWrapper inpaintMask,
                                    struct TensorWrapper dst, double inpaintRadius, int flags);

extern "C" struct TensorWrapper fastNlMeansDenoising1(struct TensorWrapper src, struct TensorWrapper dst,
                                    float h, int templateWindowSize,
                                    int searchWindowSize);

extern "C" struct TensorWrapper fastNlMeansDenoising2(struct TensorWrapper src, struct TensorWrapper dst,
                                    struct TensorWrapper h, int templateWindowSize,
                                    int searchWindowSize, int normType);

extern "C" struct TensorWrapper fastNlMeansDenoisingColored(struct TensorWrapper src, struct TensorWrapper dst,
                                    float h, float hColor, int templateWindowSize, int searchWindowSize);

extern "C" struct TensorWrapper fastNlMeansDenoisingMulti1(struct TensorArray srcImgs, struct TensorWrapper dst,
                                    int imgToDenoiseIndex, int temporalWindowSize, float h,
                                    int templateWindowSize, int searchWindowSize);

extern "C" struct TensorWrapper fastNlMeansDenoisingMulti2(struct TensorArray srcImgs, struct TensorWrapper dst,
                                    int imgToDenoiseIndex, int temporalWindowSize, struct TensorWrapper h,
                                    int templateWindowSize, int searchWindowSize, int normType);

extern "C" struct TensorWrapper fastNlMeansDenoisingColoredMulti(struct TensorArray srcImgs, struct TensorWrapper dst,
                                    int imgToDenoiseIndex, int temporalWindowSize, float h,
                                    float hColor, int templateWindowSize, int searchWindowSize);

extern "C" struct TensorWrapper denoise_TVL1(struct TensorArray observations, struct TensorWrapper result,
                                    double lambda, int niters);

extern "C" struct TensorWrapper decolor(struct TensorWrapper src, struct TensorWrapper grayscale,
                                    struct TensorWrapper color_boost);

extern "C" struct TensorWrapper seamlessClone(struct TensorWrapper src, struct TensorWrapper dst,
                                    struct TensorWrapper mask, struct PointWrapper p,
                                    struct TensorWrapper blend, int flags);

extern "C" struct TensorWrapper colorChange(struct TensorWrapper src, struct TensorWrapper mask,
                                    struct TensorWrapper dst, float red_mul,
                                    float green_mul, float blue_mul);

extern "C" struct TensorWrapper illuminationChange(struct TensorWrapper src, struct TensorWrapper mask,
                                    struct TensorWrapper dst, float alpha, float beta);

extern "C" struct TensorWrapper textureFlattening(struct TensorWrapper src, struct TensorWrapper mask,
                                    struct TensorWrapper dst, float low_threshold, float high_threshold,
                                    int kernel_size);

extern "C" struct TensorWrapper edgePreservingFilter(struct TensorWrapper src, struct TensorWrapper dst,
                                    int flags, float sigma_s, float sigma_r);

extern "C" struct TensorWrapper detailEnhance(struct TensorWrapper src, struct TensorWrapper dst,
                                    float sigma_s, float sigma_r);

extern "C" struct TensorWrapper pencilSketch(struct TensorWrapper src, struct TensorWrapper dst1,
                                    struct TensorWrapper dst2, float sigma_s, float sigma_r, float shade_factor);

extern "C" struct TensorWrapper stylization(struct TensorWrapper src, struct TensorWrapper dst,
                                    float sigma_s, float sigma_r);

/****************** Classes ******************/

// Tonemap

struct TonemapPtr {
    void *ptr;

    inline cv::Tonemap * operator->() { return static_cast<cv::Tonemap *>(ptr); }
    inline TonemapPtr(cv::Tonemap *ptr) { this->ptr = ptr; }
};

extern "C" struct TonemapPtr Tonemap_ctor(float gamma);

extern "C" struct TensorWrapper Tonemap_process(struct TonemapPtr ptr, struct TensorArray src, struct TensorWrapper dst);

extern "C" float Tonemap_getGamma(struct TonemapPtr ptr);

extern "C" void Tonemap_setGamma(struct TonemapPtr ptr, float gamma);

// TonemapDrago

struct TonemapDragoPtr {
    void *ptr;

    inline cv::TonemapDrago * operator->() { return static_cast<cv::TonemapDrago *>(ptr); }
    inline TonemapDragoPtr(cv::TonemapDrago *ptr) { this->ptr = ptr; }
};

extern "C" struct TonemapDragoPtr TonemapDrago_ctor(float gamma, float saturation, float bias);

extern "C" float TonemapDrago_getSaturation(struct TonemapDragoPtr ptr);

extern "C" void TonemapDrago_setSaturation(struct TonemapDragoPtr ptr, float saturation);

extern "C" float TonemapDrago_getBias(struct TonemapDragoPtr ptr);

extern "C" void TonemapDrago_setBias(struct TonemapDragoPtr ptr, float bias);

// TonemapDurand

struct TonemapDurandPtr {
    void *ptr;
    inline cv::TonemapDurand * operator->() { return static_cast<cv::TonemapDurand *>(ptr); }
    inline TonemapDurandPtr(cv::TonemapDurand *ptr) { this->ptr = ptr; }
};

extern "C" struct TonemapDurandPtr TonemapDurand_ctor(float gamma, float contrast, float saturation, float sigma_space, float sigma_color);

extern "C" float TonemapDurand_getSaturation(struct TonemapDurandPtr ptr);

extern "C" void TonemapDurand_setSaturation(struct TonemapDurandPtr ptr, float saturation);

extern "C" float TonemapDurand_getContrast(struct TonemapDurandPtr ptr);

extern "C" void TonemapDurand_setContrast(struct TonemapDurandPtr ptr, float contrast);

extern "C" float TonemapDurand_getSigmaSpace(struct TonemapDurandPtr ptr);

extern "C" void TonemapDurand_setSigmaSpace(struct TonemapDurandPtr ptr, float sigma_space);

extern "C" float TonemapDurand_getSigmaColor(struct TonemapDurandPtr ptr);

extern "C" void TonemapDurand_setSigmaColor(struct TonemapDurandPtr ptr, float sigma_color);

// TonemapReinhard

struct TonemapReinhardPtr {
    void *ptr;
    inline cv::TonemapReinhard * operator->() { return static_cast<cv::TonemapReinhard *>(ptr); }
    inline TonemapReinhardPtr(cv::TonemapReinhard *ptr) { this->ptr = ptr; }
};

extern "C" struct TonemapReinhardPtr TonemapReinhard_ctor(float gamma, float intensity, float light_adapt, float color_adapt);

extern "C" float TonemapReinhard_getIntensity(struct TonemapReinhardPtr ptr);

extern "C" void TonemapReinhard_setIntensity(struct TonemapReinhardPtr ptr, float intensity);

extern "C" float TonemapReinhard_getLightAdaptation(struct TonemapReinhardPtr ptr);

extern "C" void TonemapReinhard_setLightAdaptation(struct TonemapReinhardPtr ptr, float light_adapt);

extern "C" float TonemapReinhard_getColorAdaptation(struct TonemapReinhardPtr ptr);

extern "C" void TonemapReinhard_setColorAdaptation(struct TonemapReinhardPtr ptr, float color_adapt);

// TonemapMantiuk

struct TonemapMantiukPtr {
    void *ptr;
    inline cv::TonemapMantiuk * operator->() { return static_cast<cv::TonemapMantiuk *>(ptr); }
    inline TonemapMantiukPtr(cv::TonemapMantiuk *ptr) { this->ptr = ptr; }
};

extern "C" struct TonemapMantiukPtr TonemapMantiuk_ctor(float gamma, float scale, float saturation);

extern "C" float TonemapMantiuk_getScale(struct TonemapMantiukPtr ptr);

extern "C" void TonemapMantiuk_setScale(struct TonemapMantiukPtr ptr, float scale);

extern "C" float TonemapMantiuk_getSaturation(struct TonemapMantiukPtr ptr);

extern "C" void TonemapMantiuk_setSaturation(struct TonemapMantiukPtr ptr, float saturation);

// AlignExposures

struct AlignExposuresPtr {
    void *ptr;
    inline cv::AlignExposures * operator->() { return static_cast<cv::AlignExposures *>(ptr); }
    inline AlignExposuresPtr(cv::AlignExposures *ptr) { this->ptr = ptr; }
};

extern "C" struct TensorArray AlignExposures_process(struct AlignExposuresPtr ptr, struct TensorArray src, struct TensorArray dst,
                        struct TensorWrapper times, struct TensorWrapper response);

// AlignMTB

struct AlignMTBPtr {
    void *ptr;
    inline cv::AlignMTB * operator->() { return static_cast<cv::AlignMTB *>(ptr); }
    inline AlignMTBPtr(cv::AlignMTB *ptr) { this->ptr = ptr; }
};

extern "C" struct AlignMTBPtr AlignMTB_ctor(int max_bits, int exclude_range, bool cut);

extern "C" struct TensorArray AlignMTB_process1(struct AlignMTBPtr ptr, struct TensorArray src, struct TensorArray dst);

extern "C" struct TensorArray AlignMTB_process2(struct AlignMTBPtr ptr, struct TensorArray src, struct TensorArray dst,
                            struct TensorWrapper times, struct TensorWrapper response);

extern "C" struct PointWrapper AlignMTB_calculateShift(struct AlignMTBPtr ptr, struct TensorWrapper img0, struct TensorWrapper img1);

extern "C" struct TensorWrapper AlignMTB_shiftMat(struct AlignMTBPtr ptr, struct TensorWrapper src,
                            struct TensorWrapper dst, struct PointWrapper shift);

extern "C" void AlignMTB_computeBitmaps(struct AlignMTBPtr ptr, struct TensorWrapper img,
                            struct TensorWrapper tb, struct TensorWrapper eb);

extern "C" int AlignMTB_getMaxBits(struct AlignMTBPtr ptr);

extern "C" void AlignMTB_setMaxBits(struct AlignMTBPtr ptr, int max_bits);

extern "C" int AlignMTB_getExcludeRange(struct AlignMTBPtr ptr);

extern "C" void AlignMTB_setExcludeRange(struct AlignMTBPtr ptr, int exclude_range);

extern "C" bool AlignMTB_getCut(struct AlignMTBPtr ptr);

extern "C" void AlignMTB_setCut(struct AlignMTBPtr ptr, bool cut);

// CalibrateCRF

struct CalibrateCRFPtr {
    void *ptr;
    inline cv::CalibrateCRF * operator->() { return static_cast<cv::CalibrateCRF *>(ptr); }
    inline CalibrateCRFPtr(cv::CalibrateCRF *ptr) { this->ptr = ptr; }
};

extern "C" struct TensorArray CalibrateCRF_process(struct CalibrateCRFPtr ptr, struct TensorArray src, struct TensorArray dst,
                            struct TensorWrapper times);

// CalibrateDebevec

struct CalibrateDebevecPtr {
    void *ptr;
    inline cv::CalibrateDebevec * operator->() { return static_cast<cv::CalibrateDebevec *>(ptr); }
    inline CalibrateDebevecPtr(cv::CalibrateDebevec *ptr) { this->ptr = ptr; }
};

extern "C" struct CalibrateDebevecPtr CalibrateDebevec_ctor(int samples, float lambda, bool random);

extern "C" float CalibrateDebevec_getLambda(struct CalibrateDebevecPtr ptr);

extern "C" void CalibrateDebevec_setLambda(struct CalibrateDebevecPtr ptr, float lambda);

extern "C" int CalibrateDebevec_getSamples(struct CalibrateDebevecPtr ptr);

extern "C" void CalibrateDebevec_setSamples(struct CalibrateDebevecPtr ptr, int samples);

extern "C" bool CalibrateDebevec_getRandom(struct CalibrateDebevecPtr ptr);

extern "C" void CalibrateDebevec_setRandom(struct CalibrateDebevecPtr ptr, bool random);