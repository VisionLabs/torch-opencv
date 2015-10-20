#include <Common.hpp>
#include <opencv2/photo.hpp>

extern "C" struct TensorWrapper inpaint(struct TensorWrapper src, struct TensorWrapper inpaintMask,
                                    struct TensorWrapper dst, double inpaintRadius, int flags);

extern "C" struct TensorWrapper fastNlMeansDenoising(struct TensorWrapper src, struct TensorWrapper dst,
                                    float h, int templateWindowSize, int searchWindowSize);

extern "C" struct TensorWrapper fastNlMeansDenoisingColored(struct TensorWrapper src, struct TensorWrapper dst,
                                    float h, float hColor, int templateWindowSize, int searchWindowSize);

extern "C" struct TensorWrapper fastNlMeansDenoisingMulti(struct TensorArray srcImgs, struct TensorWrapper dst,
                                    int imgToDenoiseIndex, int temporalWindowSize, float h,
                                    int templateWindowSize, int searchWindowSize);

extern "C" struct TensorWrapper fastNlMeansDenoisingColoredMulti(struct TensorArray srcImgs, struct TensorWrapper dst,
                            int imgToDenoiseIndex, int temporalWindowSize, float h,
                            float hColor, int templateWindowSize, int searchWindowSize);