#include <superres.hpp>

extern "C"
struct FrameSourcePtr createFrameSource()
{
    return rescueObjectFromPtr(superres::createFrameSource_Empty());
}

extern "C"
struct FrameSourcePtr createFrameSource_Video(const char *fileName)
{
    return rescueObjectFromPtr(superres::createFrameSource_Video(fileName));
}

extern "C"
struct FrameSourcePtr createFrameSource_Video_CUDA(const char *fileName)
{
    return rescueObjectFromPtr(superres::createFrameSource_Video_CUDA(fileName));
}

extern "C"
struct FrameSourcePtr createFrameSource_Camera(int deviceId)
{
    return rescueObjectFromPtr(superres::createFrameSource_Camera(deviceId));
}

extern "C"
struct TensorWrapper FrameSource_nextFrame(struct FrameSourcePtr ptr, struct TensorWrapper frame)
{
    if (frame.isNull()) {
        cv::Mat retval;
        ptr->nextFrame(retval);
        return TensorWrapper(retval);
    } else {
        ptr->nextFrame(frame.toMat());
        return frame;
    }
}

extern "C"
void FrameSource_reset(struct FrameSourcePtr ptr)
{
    ptr->reset();
}
