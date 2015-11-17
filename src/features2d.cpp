#include <features2d.hpp>

/*KeyPointWrapper::KeyPointWrapper(const cv::KeyPoint & other) {
    this->pt = other.pt;
    this->size = other.size;
    this->angle = other.angle;
    this->response = other.response;
    this->octave = other.octave;
    this->class_id = other.class_id;
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

extern "C" void KeyPointsFilter_runByImageBorder(struct KeyPointsFilterPtr ptr, std::vector<cv::KeyPoint>& keypoints,
                    struct SizeWrapper imageSize, int borderSize)
{
    ptr->runByImageBorder(keypoints, imageSize, borderSize);
}*/

extern "C" std::vector<cv::KeyPoint> AGAST(struct TensorWrapper image, std::vector<cv::KeyPoint> keypoints,
                int threshold, bool nonmaxSuppression)
{
    std::vector<cv::KeyPoint> retval;
    cv::AGAST(image.toMat(), retval, threshold, nonmaxSuppression);
    return retval;
}