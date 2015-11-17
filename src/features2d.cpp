#include <features2d.hpp>

KeyPointWrapper::KeyPointWrapper(const cv::KeyPoint & other) {
    this->pt = other.pt;
    this->size = other.size;
    this->angle = other.angle;
    this->response = other.response;
    this->octave = other.octave;
    this->class_id = other.class_id;
}

KeyPointArray::KeyPointArray(const std::vector<cv::KeyPoint> & v)
{
    // TODO: IMPORTANT! Prevent memory leak here
    this->size = v.size();
    this->data = static_cast<KeyPointWrapper *>(
            malloc(sizeof(KeyPointWrapper) * this->size));
    for (int i = 0; i < this->size; ++i) {
        this->data[i] = v[i];
    }
}

KeyPointArray::operator std::vector<cv::KeyPoint>()
{
    std::vector<cv::KeyPoint> retval(this->size);
    for (int i = 0; i < this->size; ++i) {
        retval[i] = this->data[i];
    }
    return retval;
}

// KeyPointsFilter

extern "C" struct KeyPointsFilterPtr KeyPointsFilter_ctor()
{
    return new cv::KeyPointsFilter();
}

extern "C" void KeyPointsFilter_dtor(struct KeyPointsFilterPtr ptr)
{
    delete static_cast<cv::KeyPointsFilter *>(ptr.ptr);
}

extern "C" struct KeyPointArray KeyPointsFilter_runByImageBorder(
        struct KeyPointArray keypoints,
        struct SizeWrapper imageSize, int borderSize)
{
    std::vector<cv::KeyPoint> keypointsVector(keypoints);
    cv::KeyPointsFilter::runByImageBorder(keypointsVector, imageSize, borderSize);
    return KeyPointArray(keypointsVector);
}

extern "C" struct KeyPointArray AGAST(
        struct TensorWrapper image,
        int threshold, bool nonmaxSuppression)
{
    std::vector<cv::KeyPoint> result;
    cv::AGAST(image.toMat(), result, threshold, nonmaxSuppression);
    return KeyPointArray(result);
}