#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/superres.hpp>

namespace superres = cv::superres;

struct FrameSourcePtr {
    void *ptr;

    inline superres::FrameSource * operator->() { return static_cast<superres::FrameSource *>(ptr); }
    inline FrameSourcePtr(superres::FrameSource *ptr) { this->ptr = ptr; }
};

struct SuperResolutionPtr {
    void *ptr;

    inline superres::SuperResolution * operator->() { return static_cast<superres::SuperResolution *>(ptr); }
    inline SuperResolutionPtr(superres::SuperResolution *ptr) { this->ptr = ptr; }
};