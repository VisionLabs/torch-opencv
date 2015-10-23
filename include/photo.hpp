#include <Common.hpp>
#include <opencv2/photo.hpp>

extern "C" struct TensorWrapper inpaint(struct TensorWrapper src, struct TensorWrapper inpaintMask,
                                    struct TensorWrapper dst, double inpaintRadius, int flags);

extern "C" struct TensorWrapper fastNlMeansDenoisingCommon(struct TensorWrapper src, struct TensorWrapper dst,
                                    struct FloatArray h, int templateWindowSize,
                                    int searchWindowSize, int normType);

extern "C" struct TensorWrapper fastNlMeansDenoisingMultiCommon(struct TensorArray srcImgs, struct TensorWrapper dst,
                                    int imgToDenoiseIndex, int temporalWindowSize, struct FloatArray h,
                                    int templateWindowSize, int searchWindowSize, int normType);

extern "C" struct TensorWrapper fastNlMeansDenoisingColoredMulti(struct TensorArray srcImgs, struct TensorWrapper dst,
                                    int imgToDenoiseIndex, int temporalWindowSize, float h,
                                    float hColor, int templateWindowSize, int searchWindowSize);

// Some frightening functions with patterns and structures




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