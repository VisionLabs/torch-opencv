#include <objdetect.hpp>

extern "C"
struct TensorPlusRectArray groupRectangles(struct RectArray rectList, int groupThreshold, double eps)
{
    // TODO avoid copying
    std::vector<cv::Rect> rectListVec = rectList;
    std::vector<int> weights;
    cv::groupRectangles(rectListVec, weights, groupThreshold, eps);

    struct TensorPlusRectArray retval;
    new (&retval.rects) RectArray(rectListVec);
    new (&retval.tensor) TensorWrapper(cv::Mat(weights));

    return retval;
}

// BaseCascadeClassifier

extern "C"
bool BaseCascadeClassifier_empty(struct BaseCascadeClassifierPtr ptr)
{
    return ptr->empty();
}

extern "C"
bool BaseCascadeClassifier_load(struct BaseCascadeClassifierPtr ptr, const char *filename)
{
    return ptr->load(filename);
}

extern "C"
bool BaseCascadeClassifier_isOldFormatCascade(struct BaseCascadeClassifierPtr ptr)
{
    return ptr->isOldFormatCascade();
}

extern "C"
struct SizeWrapper BaseCascadeClassifier_getOriginalWindowSize(struct BaseCascadeClassifierPtr ptr)
{
    return ptr->getOriginalWindowSize();
}

extern "C"
int BaseCascadeClassifier_getFeatureType(struct BaseCascadeClassifierPtr ptr)
{
    return ptr->getFeatureType();
}

// CascadeClassifier

extern "C"
struct CascadeClassifierPtr CascadeClassifier_ctor_default()
{
    return new cv::CascadeClassifier();
}

extern "C"
struct CascadeClassifierPtr CascadeClassifier_ctor(const char *filename)
{
    return new cv::CascadeClassifier(filename);
}

extern "C"
bool CascadeClassifier_read(struct CascadeClassifierPtr ptr, struct FileNodePtr node)
{
    return ptr->read(*node);
}

extern "C"
struct RectArray CascadeClassifier_detectMultiScale(struct CascadeClassifierPtr ptr,
        struct TensorWrapper image, double scaleFactor, int minNeighbors, int flags,
        struct SizeWrapper minSize, struct SizeWrapper maxSize)
{
    std::vector<cv::Rect> retval;
    ptr->detectMultiScale(
            image.toMat(), retval, scaleFactor, minNeighbors, flags, minSize, maxSize);
    return RectArray(retval);
}

extern "C"
struct TensorPlusRectArray CascadeClassifier_detectMultiScale2(struct CascadeClassifierPtr ptr,
        struct TensorWrapper image, double scaleFactor, int minNeighbors, int flags,
        struct SizeWrapper minSize, struct SizeWrapper maxSize)
{
    struct TensorPlusRectArray retval;
    std::vector<cv::Rect> objects;
    std::vector<int> numDetections;

    ptr->detectMultiScale(
            image.toMat(), objects, numDetections, scaleFactor,
            minNeighbors, flags, minSize, maxSize);

    new (&retval.rects) RectArray(objects);
    new (&retval.tensor) TensorWrapper(cv::Mat(numDetections));

    return retval;
}

extern "C"
struct TensorArrayPlusRectArray CascadeClassifier_detectMultiScale3(
        struct CascadeClassifierPtr ptr, struct TensorWrapper image, double scaleFactor,
        int minNeighbors, int flags, struct SizeWrapper minSize, struct SizeWrapper maxSize,
        bool outputRejectLevels)
{
    struct TensorArrayPlusRectArray retval;
    std::vector<cv::Rect> objects;
    std::vector<int> rejectLevels;
    std::vector<double> levelWeights;

    ptr->detectMultiScale(
            image.toMat(), objects, rejectLevels, levelWeights,
            scaleFactor, minNeighbors, flags, minSize, maxSize);

    new (&retval.rects) RectArray(objects);
    std::vector<cv::Mat> matArray(2);
    matArray[0] = cv::Mat(rejectLevels);
    matArray[1] = cv::Mat(levelWeights);
    new (&retval.tensors) TensorArray(matArray);

    return retval;
}

extern "C"
bool CascadeClassifier_convert(
        struct CascadeClassifierPtr ptr, const char *oldcascade, const char *newcascade)
{
    return ptr->convert(oldcascade, newcascade);
}

extern "C"
struct HOGDescriptorPtr HOGDescriptor_ctor(
        struct SizeWrapper winSize, struct SizeWrapper blockSize, struct SizeWrapper blockStride,
        struct SizeWrapper cellSize, int nbins, int derivAperture, double winSigma,
        int histogramNormType, double L2HysThreshold, bool gammaCorrection,
        int nlevels, bool signedGradient)
{
    return new cv::HOGDescriptor(
            winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient);
}

extern "C"
void HOGDescriptor_dtor(struct HOGDescriptorPtr ptr)
{
    delete static_cast<cv::HOGDescriptor *>(ptr.ptr);
}

extern "C"
size_t HOGDescriptor_getDescriptorSize(struct HOGDescriptorPtr ptr)
{
    return ptr->getDescriptorSize();
}

extern "C"
bool HOGDescriptor_checkDetectorSize(struct HOGDescriptorPtr ptr)
{
    return ptr->checkDetectorSize();
}

extern "C"
double HOGDescriptor_getWinSigma(struct HOGDescriptorPtr ptr)
{
    return ptr->getWinSigma();
}

extern "C"
void HOGDescriptor_setSVMDetector(struct HOGDescriptorPtr ptr, struct TensorWrapper _svmdetector)
{
    ptr->setSVMDetector(_svmdetector.toMat());
}

extern "C"
bool HOGDescriptor_load(
        struct HOGDescriptorPtr ptr, const char *filename, const char *objname)
{
    return ptr->load(filename, objname);
}

extern "C"
void HOGDescriptor_save(
        struct HOGDescriptorPtr ptr, const char *filename, const char *objname)
{
    ptr->save(filename, objname);
}

extern "C"
struct TensorWrapper HOGDescriptor_compute(
        struct HOGDescriptorPtr ptr, struct TensorWrapper img, struct SizeWrapper winStride,
        struct SizeWrapper padding, struct PointArray locations)
{
    std::vector<float> retval;
    ptr->compute(img.toMat(), retval, winStride, padding, locations);
    return TensorWrapper(cv::Mat(retval));
}

extern "C"
struct TensorPlusPointArray HOGDescriptor_detect(
        struct HOGDescriptorPtr ptr, struct TensorWrapper img, double hitThreshold,
        struct SizeWrapper winStride, struct SizeWrapper padding, struct PointArray searchLocations)
{
    struct TensorPlusPointArray retval;
    std::vector<cv::Point> foundLocations;
    std::vector<double> weights;

    ptr->detect(img.toMat(), foundLocations, weights,
                hitThreshold, winStride, padding, searchLocations);

    new (&retval.points) PointArray(foundLocations);
    new (&retval.tensor) TensorWrapper(cv::Mat(weights));

    return retval;
}

extern "C"
struct TensorPlusRectArray HOGDescriptor_detectMultiScale(
        struct HOGDescriptorPtr ptr, struct TensorWrapper img, double hitThreshold,
        struct SizeWrapper winStride, struct SizeWrapper padding, double scale,
        double finalThreshold, bool useMeanshiftGrouping)
{
    struct TensorPlusRectArray retval;
    std::vector<cv::Rect> foundLocations;
    std::vector<double> foundWeights;

    ptr->detectMultiScale(img.toMat(), foundLocations, foundWeights,
                hitThreshold, winStride, padding, scale, finalThreshold, useMeanshiftGrouping);

    new (&retval.rects) RectArray(foundLocations);
    new (&retval.tensor) TensorWrapper(cv::Mat(foundWeights));

    return retval;
}

extern "C"
struct TensorArray HOGDescriptor_computeGradient(
        struct HOGDescriptorPtr ptr, struct TensorWrapper img,
        struct SizeWrapper paddingTL, struct SizeWrapper paddingBR)
{
    std::vector<cv::Mat> retval;
    ptr->computeGradient(img.toMat(), retval[0], retval[1], paddingTL, paddingBR);
    return TensorArray(retval);
}

extern "C"
struct TensorWrapper HOGDescriptor_getDefaultPeopleDetector()
{
    return TensorWrapper(cv::Mat(cv::HOGDescriptor::getDefaultPeopleDetector()));
}

extern "C"
struct TensorWrapper HOGDescriptor_getDaimlerPeopleDetector()
{
    return TensorWrapper(cv::Mat(cv::HOGDescriptor::getDaimlerPeopleDetector()));
}
