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