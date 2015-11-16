#include <superres.hpp>
#include <H5FSpkg.h>

// FrameSource

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

// SuperResolution

extern "C"
struct SuperResolutionPtr createSuperResolution_BTVL1()
{
    return rescueObjectFromPtr(superres::createSuperResolution_BTVL1());
}

extern "C"
struct SuperResolutionPtr createSuperResolution_BTVL1_CUDA()
{
    return rescueObjectFromPtr(superres::createSuperResolution_BTVL1_CUDA());
}

extern "C"
struct TensorWrapper SuperResolution_nextFrame(struct SuperResolutionPtr ptr, struct TensorWrapper frame)
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
void SuperResolution_reset(struct SuperResolutionPtr ptr)
{
    ptr->reset();
}

extern "C"
void SuperResolution_setInput(struct SuperResolutionPtr ptr, struct FrameSourcePtr frameSource)
{
    cv::Ptr<superres::FrameSource> frameSourcePtr(
            static_cast<superres::FrameSource *>(frameSource.ptr));
    rescueObjectFromPtr(frameSourcePtr);
    ptr->setInput(frameSourcePtr);
}

extern "C"
void SuperResolution_collectGarbage(struct SuperResolutionPtr ptr)
{
    ptr->collectGarbage();
}

extern "C"
void SuperResolution_setScale(struct SuperResolutionPtr ptr, int val)
{
    ptr->setScale(val);
}

extern "C"
int SuperResolution_getScale(struct SuperResolutionPtr ptr)
{
    return ptr->getScale();
}

extern "C"
void SuperResolution_setIterations(struct SuperResolutionPtr ptr, int val)
{
    ptr->setIterations(val);
}

extern "C"
int SuperResolution_getIterations(struct SuperResolutionPtr ptr)
{
    return ptr->getIterations();
}

extern "C"
void SuperResolution_setTau(struct SuperResolutionPtr ptr, double val)
{
    ptr->setTau(val);
}

extern "C"
double SuperResolution_getTau(struct SuperResolutionPtr ptr)
{
    return ptr->getTau();
}

extern "C"
void SuperResolution_setLabmda(struct SuperResolutionPtr ptr, double val)
{
    ptr->setLabmda(val);
}

extern "C"
double SuperResolution_getLabmda(struct SuperResolutionPtr ptr)
{
    return ptr->getLabmda();
}

extern "C"
void SuperResolution_setAlpha(struct SuperResolutionPtr ptr, double val)
{
    ptr->setAlpha(val);
}

extern "C"
double SuperResolution_getAlpha(struct SuperResolutionPtr ptr)
{
    return ptr->getAlpha();
}

extern "C"
void SuperResolution_setKernelSize(struct SuperResolutionPtr ptr, int val)
{
    ptr->setKernelSize(val);
}

extern "C"
int SuperResolution_getKernelSize(struct SuperResolutionPtr ptr)
{
    return ptr->getKernelSize();
}

extern "C"
void SuperResolution_setBlurKernelSize(struct SuperResolutionPtr ptr, int val)
{
    ptr->setBlurKernelSize(val);
}

extern "C"
int SuperResolution_getBlurKernelSize(struct SuperResolutionPtr ptr)
{
    return ptr->getBlurKernelSize();
}

extern "C"
void SuperResolution_setBlurSigma(struct SuperResolutionPtr ptr, double val)
{
    ptr->setBlurSigma(val);
}

extern "C"
double SuperResolution_getBlurSigma(struct SuperResolutionPtr ptr)
{
    return ptr->getBlurSigma();
}

extern "C"
void SuperResolution_setTemporalAreaRadius(struct SuperResolutionPtr ptr, int val)
{
    ptr->setTemporalAreaRadius(val);
}

extern "C"
int SuperResolution_getTemporalAreaRadius(struct SuperResolutionPtr ptr)
{
    return ptr->getTemporalAreaRadius();
}

// TODO this
//extern "C"
//void SuperResolution_setOpticalFlow(struct SuperResolutionPtr ptr, int val)
//{
//    ptr->setOpticalFlow(val);
//}
//
//extern "C"
//int SuperResolution_getOpticalFlow(struct SuperResolutionPtr ptr)
//{
//    return ptr->getOpticalFlow();
//}

// FarnebackOpticalFlow

extern "C"
struct FarnebackOpticalFlowPtr FarnebackOpticalFlow_ctor()
{
    return rescueObjectFromPtr(superres::createOptFlow_Farneback());
}

extern "C"
void FarnebackOpticalFlow_setPyrScale(struct FarnebackOpticalFlowPtr ptr, double val)
{
    ptr->setPyrScale(val);
}

extern "C"
double FarnebackOpticalFlow_getPyrScale(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getPyrScale();
}

extern "C"
void FarnebackOpticalFlow_setLevelsNumber(struct FarnebackOpticalFlowPtr ptr, int val)
{
    ptr->setLevelsNumber(val);
}

extern "C"
int FarnebackOpticalFlow_getLevelsNumber(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getLevelsNumber();
}

extern "C"
void FarnebackOpticalFlow_setWindowSize(struct FarnebackOpticalFlowPtr ptr, int val)
{
    ptr->setWindowSize(val);
}

extern "C"
int FarnebackOpticalFlow_getWindowSize(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getWindowSize();
}

extern "C"
void FarnebackOpticalFlow_setIterations(struct FarnebackOpticalFlowPtr ptr, int val)
{
    ptr->setIterations(val);
}

extern "C"
int FarnebackOpticalFlow_getIterations(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getIterations();
}

extern "C"
void FarnebackOpticalFlow_setPolyN(struct FarnebackOpticalFlowPtr ptr, int val)
{
    ptr->setPolyN(val);
}

extern "C"
int FarnebackOpticalFlow_getPolyN(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getPolyN();
}

extern "C"
void FarnebackOpticalFlow_setPolySigma(struct FarnebackOpticalFlowPtr ptr, int val)
{
    ptr->setPolySigma(val);
}

extern "C"
double FarnebackOpticalFlow_getPolySigma(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getPolySigma();
}

extern "C"
void FarnebackOpticalFlow_setFlags(struct FarnebackOpticalFlowPtr ptr, int val)
{
    ptr->setFlags(val);
}

extern "C"
int FarnebackOpticalFlow_getFlags(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getFlags();
}

