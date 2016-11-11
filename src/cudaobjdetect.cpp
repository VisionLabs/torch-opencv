#include <cudaobjdetect.hpp>

extern "C"
struct HOGPtr HOG_ctorCuda(
        struct SizeWrapper win_size, struct SizeWrapper block_size,
        struct SizeWrapper block_stride, struct SizeWrapper cell_size, int nbins)
{
    return rescueObjectFromPtr(cuda::HOG::create(
            win_size, block_size, block_stride, cell_size, nbins));
}

extern "C"
void HOG_setWinSigmaCuda(struct HOGPtr ptr, double val)
{
    ptr->setWinSigma(val);
}

extern "C"
double HOG_getWinSigmaCuda(struct HOGPtr ptr)
{
    return ptr->getWinSigma();
}

extern "C"
void HOG_setL2HysThresholdCuda(struct HOGPtr ptr, double val)
{
    ptr->setL2HysThreshold(val);
}

extern "C"
double HOG_getL2HysThresholdCuda(struct HOGPtr ptr)
{
    return ptr->getL2HysThreshold();
}

extern "C"
void HOG_setGammaCorrectionCuda(struct HOGPtr ptr, bool val)
{
    ptr->setGammaCorrection(val);
}

extern "C"
bool HOG_getGammaCorrectionCuda(struct HOGPtr ptr)
{
    return ptr->getGammaCorrection();
}

extern "C"
void HOG_setNumLevelsCuda(struct HOGPtr ptr, int val)
{
    ptr->setNumLevels(val);
}

extern "C"
int HOG_getNumLevelsCuda(struct HOGPtr ptr)
{
    return ptr->getNumLevels();
}

extern "C"
void HOG_setHitThresholdCuda(struct HOGPtr ptr, double val)
{
    ptr->setHitThreshold(val);
}

extern "C"
double HOG_getHitThresholdCuda(struct HOGPtr ptr)
{
    return ptr->getHitThreshold();
}

extern "C"
void HOG_setWinStrideCuda(struct HOGPtr ptr, struct SizeWrapper val)
{
    ptr->setWinStride(val);
}

extern "C"
struct SizeWrapper HOG_getWinStrideCuda(struct HOGPtr ptr)
{
    return ptr->getWinStride();
}

extern "C"
void HOG_setScaleFactorCuda(struct HOGPtr ptr, double val)
{
    ptr->setScaleFactor(val);
}

extern "C"
double HOG_getScaleFactorCuda(struct HOGPtr ptr)
{
    return ptr->getScaleFactor();
}

extern "C"
void HOG_setGroupThresholdCuda(struct HOGPtr ptr, int val)
{
    ptr->setGroupThreshold(val);
}

extern "C"
int HOG_getGroupThresholdCuda(struct HOGPtr ptr)
{
    return ptr->getGroupThreshold();
}

extern "C"
void HOG_setDescriptorFormatCuda(struct HOGPtr ptr, int val)
{
    ptr->setDescriptorFormat(val);
}

extern "C"
int HOG_getDescriptorFormatCuda(struct HOGPtr ptr)
{
    return ptr->getDescriptorFormat();
}

extern "C"
size_t HOG_getDescriptorSizeCuda(struct HOGPtr ptr)
{
    return ptr->getDescriptorSize();
}

extern "C"
size_t HOG_getBlockHistogramSizeCuda(struct HOGPtr ptr)
{
    return ptr->getBlockHistogramSize();
}

extern "C"
void HOG_setSVMDetectorCuda(struct HOGPtr ptr, struct TensorWrapper val)
{
    ptr->setSVMDetector(val.toMat());
}

extern "C"
struct TensorWrapper HOG_getDefaultPeopleDetectorCuda(struct HOGPtr ptr)
{
    return TensorWrapper(ptr->getDefaultPeopleDetector());
}

extern "C"
struct TensorPlusPointArray HOG_detectCuda(
        struct cutorchInfo info, struct HOGPtr ptr, struct TensorWrapper img)
{
    std::vector<cv::Point> found_locations;
    std::vector<double> confidences;
    cuda::GpuMat imgMat = img.toGpuMat();
    cuda::GpuMat imgMatByte;
    imgMat.convertTo(imgMatByte, CV_8U, 255.0); // Sorry guys :( #156
    ptr->detect(imgMatByte, found_locations, &confidences);

    TensorPlusPointArray retval;
    new (&retval.points) PointArray(found_locations);
    new (&retval.tensor) TensorWrapper(cv::Mat(confidences));
    return retval;
}

extern "C"
struct TensorPlusRectArray HOG_detectMultiScaleCuda(
        struct cutorchInfo info, struct HOGPtr ptr, struct TensorWrapper img)
{
    std::vector<cv::Rect> found_locations;
    std::vector<double> confidences;
    cuda::GpuMat imgMat = img.toGpuMat();
    cuda::GpuMat imgMatByte;
    imgMat.convertTo(imgMatByte, CV_8U, 255.0); // Sorry guys :( #156

    if (ptr->getGroupThreshold() == 0) {
        ptr->detectMultiScale(imgMatByte, found_locations, &confidences);
    } else {
        ptr->detectMultiScale(imgMatByte, found_locations, nullptr);
    }

    TensorPlusRectArray retval;
    new (&retval.rects) RectArray(found_locations);
    new (&retval.tensor) TensorWrapper(cv::Mat(confidences));
    return retval;
}

extern "C"
struct TensorWrapper HOG_computeCuda(
        struct cutorchInfo info, struct HOGPtr ptr, struct TensorWrapper img,
        struct TensorWrapper descriptors)
{
    GpuMatT descriptorsMat = descriptors.toGpuMatT();
    cuda::GpuMat imgMat = img.toGpuMat();
    cuda::GpuMat imgMatByte;
    imgMat.convertTo(imgMatByte, CV_8U, 255.0); // Sorry guys :( #156

    ptr->compute(imgMatByte, descriptorsMat, prepareStream(info));
    return TensorWrapper(descriptorsMat, info.state);
}

extern "C"
struct CascadeClassifierPtr CascadeClassifier_ctor_filenameCuda(const char *filename)
{
    return rescueObjectFromPtr(cuda::CascadeClassifier::create(filename));
}

extern "C"
struct CascadeClassifierPtr CascadeClassifier_ctor_fileCuda(struct FileStoragePtr file)
{
    return rescueObjectFromPtr(cuda::CascadeClassifier::create(*file));
}

extern "C"
void CascadeClassifier_setMaxObjectSizeCuda(struct CascadeClassifierPtr ptr, struct SizeWrapper val)
{
    ptr->setMaxObjectSize(val);
}

extern "C"
struct SizeWrapper CascadeClassifier_getMaxObjectSizeCuda(struct CascadeClassifierPtr ptr)
{
    return ptr->getMaxObjectSize();
}

extern "C"
void CascadeClassifier_setMinObjectSizeCuda(struct CascadeClassifierPtr ptr, struct SizeWrapper val)
{
    ptr->setMinObjectSize(val);
}

extern "C"
struct SizeWrapper CascadeClassifier_getMinObjectSizeCuda(struct CascadeClassifierPtr ptr)
{
    return ptr->getMinObjectSize();
}

extern "C"
void CascadeClassifier_setScaleFactorCuda(struct CascadeClassifierPtr ptr, double val)
{
    ptr->setScaleFactor(val);
}

extern "C"
double CascadeClassifier_getScaleFactorCuda(struct CascadeClassifierPtr ptr)
{
    return ptr->getScaleFactor();
}

extern "C"
void CascadeClassifier_setMinNeighborsCuda(struct CascadeClassifierPtr ptr, int val)
{
    ptr->setMinNeighbors(val);
}

extern "C"
int CascadeClassifier_getMinNeighborsCuda(struct CascadeClassifierPtr ptr)
{
    return ptr->getMinNeighbors();
}

extern "C"
void CascadeClassifier_setFindLargestObjectCuda(struct CascadeClassifierPtr ptr, bool val)
{
    ptr->setFindLargestObject(val);
}

extern "C"
bool CascadeClassifier_getFindLargestObjectCuda(struct CascadeClassifierPtr ptr)
{
    return ptr->getFindLargestObject();
}

extern "C"
void CascadeClassifier_setMaxNumObjectsCuda(struct CascadeClassifierPtr ptr, int val)
{
    ptr->setMaxNumObjects(val);
}

extern "C"
int CascadeClassifier_getMaxNumObjectsCuda(struct CascadeClassifierPtr ptr)
{
    return ptr->getMaxNumObjects();
}

extern "C"
struct SizeWrapper CascadeClassifier_getClassifierSizeCuda(struct CascadeClassifierPtr ptr)
{
    return ptr->getClassifierSize();
}

extern "C"
struct TensorWrapper CascadeClassifier_detectMultiScaleCuda(
        struct cutorchInfo info, struct CascadeClassifierPtr ptr,
        struct TensorWrapper image, struct TensorWrapper objects)
{
    GpuMatT objectsMat = objects.toGpuMatT();
    cuda::GpuMat imageMat = image.toGpuMat();
    cuda::GpuMat imageByte;
    imageMat.convertTo(imageByte, CV_8U, 255.0); // Sorry guys :(
    ptr->detectMultiScale(imageByte, objectsMat, prepareStream(info));
    return TensorWrapper(objectsMat, info.state);
}

extern "C"
struct RectArray CascadeClassifier_convertCuda(
        struct CascadeClassifierPtr ptr, struct TensorWrapper gpu_objects)
{
    auto mat = gpu_objects.toGpuMat(CV_32S);
    std::vector<cv::Rect> objects;
    ptr->convert(mat, objects);
    return RectArray(objects);
}
