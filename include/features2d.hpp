#include <Common.hpp>
#include <Classes.hpp>
//#include <opencv2/superres/optical_flow.hpp>
#include <opencv2/features2d.hpp>

// KeyPointsFilter

struct KeyPointsFilterPtr {
    void *ptr;

    inline cv::KeyPointsFilter * operator->() { return static_cast<cv::KeyPointsFilter *>(ptr); }
    inline KeyPointsFilterPtr(cv::KeyPointsFilter *ptr) { this->ptr = ptr; }
    inline cv::KeyPointsFilter & operator*() { return *static_cast<cv::KeyPointsFilter *>(this->ptr); }
};

extern "C" struct KeyPointsFilterPtr KeyPointsFilter_ctor();

extern "C" void KeyPointsFilter_dtor(struct KeyPointsFilterPtr ptr);