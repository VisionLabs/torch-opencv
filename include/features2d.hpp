#include <Common.hpp>
#include <Classes.hpp>
//#include <opencv2/superres/optical_flow.hpp>
#include <opencv2/features2d.hpp>

/*struct KeyPointWrapper {
    struct Point2fWrapper pt;
    float size, angle, response;
    int octave, class_id;
       
    inline operator cv::KeyPoint() { return cv::KeyPoint(pt, size, angle, response, octave, class_id); }
    KeyPointWrapper(const cv::KeyPoint & other);
};

struct KeyPointArray {
    struct KeyPointWrapper *data;
    int size;
};*/

// KeyPointsFilter

/*struct KeyPointsFilterPtr {
    void *ptr;

    inline cv::KeyPointsFilter * operator->() { return static_cast<cv::KeyPointsFilter *>(ptr); }
    inline KeyPointsFilterPtr(cv::KeyPointsFilter *ptr) { this->ptr = ptr; }
    inline cv::KeyPointsFilter & operator*() { return *static_cast<cv::KeyPointsFilter *>(this->ptr); }
};

extern "C" struct KeyPointsFilterPtr KeyPointsFilter_ctor();

extern "C" void KeyPointsFilter_dtor(struct KeyPointsFilterPtr ptr);

extern "C" void KeyPointsFilter_runByImageBorder(struct KeyPointsFilterPtr ptr, std::vector<cv::KeyPoint>& keypoints,
                    struct SizeWrapper imageSize, int borderSize);*/

/*extern "C" std::vector<cv::KeyPoint> AGAST(struct TensorWrapper image, std::vector<cv::KeyPoint> keypoints,
                int threshold, bool nonmaxSuppression);*/