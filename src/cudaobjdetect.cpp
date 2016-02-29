#include <cudaobjdetect.hpp>

struct HOGPtr HOG_ctor(
        struct SizeWrapper win_size, struct SizeWrapper block_size,
        struct SizeWrapper block_stride, struct SizeWrapper cell_size, int nbins)
{
    return rescueObjectFromPtr(cuda::HOG::create(
            win_size, block_size, block_stride, cell_size, nbins));
}

extern "C"
void HOG_setWinSigma(struct HOGPtr ptr, double val)
{
    ptr->setWinSigma(val);
}

extern "C"
double HOG_getWinSigma(struct HOGPtr ptr)
{
    return ptr->getWinSigma();
}

extern "C"
void HOG_setL2HysThreshold(struct HOGPtr ptr, double val)
{
    ptr->setL2HysThreshold(val);
}

extern "C"
double HOG_getL2HysThreshold(struct HOGPtr ptr)
{
    return ptr->getL2HysThreshold();
}

extern "C"
void HOG_setGammaCorrection(struct HOGPtr ptr, bool val)
{
    ptr->setGammaCorrection(val);
}

extern "C"
bool HOG_getGammaCorrection(struct HOGPtr ptr)
{
    return ptr->getGammaCorrection();
}

extern "C"
void HOG_setNumLevels(struct HOGPtr ptr, int val)
{
    ptr->setNumLevels(val);
}

extern "C"
int HOG_getNumLevels(struct HOGPtr ptr)
{
    return ptr->getNumLevels();
}

extern "C"
void HOG_setHitThreshold(struct HOGPtr ptr, double val)
{
    ptr->setHitThreshold(val);
}

extern "C"
double HOG_getHitThreshold(struct HOGPtr ptr)
{
    return ptr->getHitThreshold();
}

extern "C"
void HOG_setWinStride(struct HOGPtr ptr, struct SizeWrapper val)
{
    ptr->setWinStride(val);
}

extern "C"
struct SizeWrapper HOG_getWinStride(struct HOGPtr ptr)
{
    return ptr->getWinStride();
}

extern "C"
void HOG_setScaleFactor(struct HOGPtr ptr, double val)
{
    ptr->setScaleFactor(val);
}

extern "C"
double HOG_getScaleFactor(struct HOGPtr ptr)
{
    return ptr->getScaleFactor();
}

extern "C"
void HOG_setGroupThreshold(struct HOGPtr ptr, int val)
{
    ptr->setGroupThreshold(val);
}

extern "C"
int HOG_getGroupThreshold(struct HOGPtr ptr)
{
    return ptr->getGroupThreshold();
}

extern "C"
void HOG_setDescriptorFormat(struct HOGPtr ptr, int val)
{
    ptr->setDescriptorFormat(val);
}

extern "C"
int HOG_getDescriptorFormat(struct HOGPtr ptr)
{
    return ptr->getDescriptorFormat();
}

extern "C"
size_t HOG_getDescriptorSize(struct HOGPtr ptr)
{
    return ptr->getDescriptorSize();
}

extern "C"
size_t HOG_getBlockHistogramSize(struct HOGPtr ptr)
{
    return ptr->getBlockHistogramSize();
}

extern "C"
void HOG_setSVMDetector(struct HOGPtr ptr, struct TensorWrapper val)
{
    ptr->setSVMDetector(val.toMat());
}

extern "C"
struct TensorWrapper HOG_getDefaultPeopleDetector(struct HOGPtr ptr)
{
    return TensorWrapper(ptr->getDefaultPeopleDetector());
}

extern "C"
struct TensorPlusPointArray HOG_detect(
        cutorchInfo info, struct HOGPtr ptr, struct TensorWrapper img)
{
    std::vector<cv::Point> found_locations;
    std::vector<double> confidences;
    ptr->detect(img.toGpuMat(), found_locations, &confidences);

    TensorPlusPointArray retval;
    new (&retval.points) PointArray(found_locations);
    new (&retval.tensor) TensorWrapper(cv::Mat(confidences));
    return retval;
}

extern "C"
struct TensorPlusRectArray HOG_detectMultiScale(
        cutorchInfo info, struct HOGPtr ptr, struct TensorWrapper img)
{
    std::vector<cv::Rect> found_locations;
    std::vector<double> confidences;
    ptr->detectMultiScale(img.toGpuMat(), found_locations, &confidences);

    TensorPlusRectArray retval;
    new (&retval.rects) RectArray(found_locations);
    new (&retval.tensor) TensorWrapper(cv::Mat(confidences));
    return retval;
}

extern "C"
struct TensorWrapper HOG_compute(
        cutorchInfo info, struct HOGPtr ptr, struct TensorWrapper img,
        struct TensorWrapper descriptors)
{
    GpuMatT descriptorsMat = descriptors.toGpuMatT();
    ptr->compute(img.toGpuMat(), descriptorsMat, prepareStream(info));
    return TensorWrapper(descriptorsMat, info.state);
}

