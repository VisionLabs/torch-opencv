#include <features2d.hpp>

extern "C"
struct KeyPointsFilterPtr KeyPointsFilter_ctor() {
    return new cv::KeyPointsFilter();
}

extern "C"
void KeyPointsFilter_dtor(struct KeyPointsFilterPtr ptr) {
    delete static_cast<cv::KeyPointsFilter *>(ptr.ptr);
}