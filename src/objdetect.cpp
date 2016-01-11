#include <objdetect.hpp>

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

bool BaseCascadeClassifier_empty(struct BaseCascadeClassifierPtr ptr)
{
    return ptr->empty();
}

bool BaseCascadeClassifier_load(struct BaseCascadeClassifierPtr ptr, const char *filename)
{
    return ptr->load(filename);
}

bool BaseCascadeClassifier_isOldFormatCascade(struct BaseCascadeClassifierPtr ptr)
{
    return ptr->isOldFormatCascade();
}

struct SizeWrapper BaseCascadeClassifier_getOriginalWindowSize(struct BaseCascadeClassifierPtr ptr)
{
    return ptr->getOriginalWindowSize();
}

int BaseCascadeClassifier_getFeatureType(struct BaseCascadeClassifierPtr ptr)
{
    return ptr->getFeatureType();
}

// CascadeClassifier

struct CascadeClassifierPtr CascadeClassifier_ctor_default()
{
    return new cv::CascadeClassifier();
}

struct CascadeClassifierPtr CascadeClassifier_ctor(const char *filename)
{
    return new cv::CascadeClassifier(filename);
}

bool CascadeClassifier_read(struct CascadeClassifierPtr ptr, struct FileNodePtr node)
{
    return ptr->read(*node);
}

struct RectArray CascadeClassifier_detectMultiScale(struct CascadeClassifierPtr ptr,
        struct TensorWrapper image, double scaleFactor, int minNeighbors, int flags,
        struct SizeWrapper minSize, struct SizeWrapper maxSize)
{
    std::vector<cv::Rect> retval;
    ptr->detectMultiScale(
            image.toMat(), retval, scaleFactor, minNeighbors, flags, minSize, maxSize);
    return RectArray(retval);
}

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

bool CascadeClassifier_convert(
        struct CascadeClassifierPtr ptr, const char *oldcascade, const char *newcascade)
{
    return ptr->convert(oldcascade, newcascade);
}