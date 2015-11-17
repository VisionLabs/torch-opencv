#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/features2d.hpp>

struct KeyPointWrapper {
    struct Point2fWrapper pt;
    float size, angle, response;
    int octave, class_id;
       
    inline operator cv::KeyPoint() { return cv::KeyPoint(pt, size, angle, response, octave, class_id); }
    KeyPointWrapper(const cv::KeyPoint & other);
};

struct KeyPointArray {
    struct KeyPointWrapper *data;
    int size;

    KeyPointArray(const std::vector<cv::KeyPoint> & v);
    operator std::vector<cv::KeyPoint>();
};

// KeyPointsFilter

struct KeyPointsFilterPtr {
    void *ptr;

    inline cv::KeyPointsFilter * operator->() { return static_cast<cv::KeyPointsFilter *>(ptr); }
    inline KeyPointsFilterPtr(cv::KeyPointsFilter *ptr) { this->ptr = ptr; }
    inline cv::KeyPointsFilter & operator*() { return *static_cast<cv::KeyPointsFilter *>(this->ptr); }
};

extern "C" struct KeyPointsFilterPtr KeyPointsFilter_ctor();

extern "C" void KeyPointsFilter_dtor(struct KeyPointsFilterPtr ptr);

extern "C" struct KeyPointArray KeyPointsFilter_runByImageBorder(
        struct KeyPointArray keypoints,
        struct SizeWrapper imageSize, int borderSize);

extern "C" struct KeyPointArray AGAST(
        struct TensorWrapper image, int threshold, bool nonmaxSuppression);