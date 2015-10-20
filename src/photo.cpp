#include <photo.hpp>

extern "C" struct TensorWrapper inpaint(struct TensorWrapper src, struct TensorWrapper inpaintMask,
                                    struct TensorWrapper dst, double inpaintRadius, int flags)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::inpaint(src.toMat(), inpaintMask.toMat(), retval, inpaintRadius, flags);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::inpaint(source, inpaintMask.toMat(), source, inpaintRadius, flags);
    } else {
        cv::inpaint(src.toMat(), inpaintMask.toMat(), dst.toMat(), inpaintRadius, flags);
    }
    return dst;
}

extern "C" struct TensorWrapper fastNlMeansDenoising(struct TensorWrapper src, struct TensorWrapper dst,
                                    float h, int templateWindowSize, int searchWindowSize)
{
    if (dst.isNull()) {
        cv::Mat retval;
        cv::fastNlMeansDenoising(src.toMat(), retval, h, templateWindowSize, searchWindowSize);
        return TensorWrapper(retval);
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::fastNlMeansDenoising(source, source, h, templateWindowSize, searchWindowSize);
    } else {
        cv::fastNlMeansDenoising(src.toMat(), dst.toMat(), h, templateWindowSize, searchWindowSize);
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
    } else if (dst.tensorPtr == src.tensorPtr) {
        // in-place
        cv::Mat source = src.toMat();
        cv::fastNlMeansDenoisingColored(source, source, h, hColor, templateWindowSize, searchWindowSize);
    } else {
        cv::fastNlMeansDenoisingColored(src.toMat(), dst.toMat(), h, hColor, templateWindowSize, searchWindowSize);
    }
    return dst;
}

extern "C" struct TensorWrapper fastNlMeansDenoisingMulti(struct TensorArray srcImgs, struct TensorWrapper dst,
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