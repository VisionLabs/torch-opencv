#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/superres.hpp>

namespace superres = cv::superres;

struct FrameSourcePtr {
    void *ptr;

    inline superres::FrameSource * operator->() { return static_cast<superres::FrameSource *>(ptr); }
    inline FrameSourcePtr(superres::FrameSource *ptr) { this->ptr = ptr; }
};

extern "C"
struct FrameSourcePtr createFrameSource();

extern "C"
struct FrameSourcePtr createFrameSource_Video(const char *fileName);

extern "C"
struct FrameSourcePtr createFrameSource_Video_CUDA(const char *fileName);

extern "C"
struct FrameSourcePtr createFrameSource_Camera(int deviceId);

extern "C"
struct TensorWrapper FrameSource_nextFrame(struct FrameSourcePtr ptr, struct TensorWrapper frame);

extern "C"
void FrameSource_reset(struct FrameSourcePtr ptr);

struct SuperResolutionPtr {
    void *ptr;

    inline superres::SuperResolution * operator->() { return static_cast<superres::SuperResolution *>(ptr); }
    inline SuperResolutionPtr(superres::SuperResolution *ptr) { this->ptr = ptr; }
};

extern "C"
struct SuperResolutionPtr createSuperResolution_BTVL1();

extern "C"
struct SuperResolutionPtr createSuperResolution_BTVL1_CUDA();

extern "C"
struct TensorWrapper SuperResolution_nextFrame(struct SuperResolutionPtr ptr, struct TensorWrapper frame);

extern "C"
void SuperResolution_reset(struct SuperResolutionPtr ptr);

extern "C"
void SuperResolution_setInput(struct SuperResolutionPtr ptr, struct FrameSourcePtr frameSource);

extern "C"
void SuperResolution_collectGarbage(struct SuperResolutionPtr ptr);

extern "C"
void SuperResolution_setScale(struct SuperResolutionPtr ptr, int val);

extern "C"
int SuperResolution_getScale(struct SuperResolutionPtr ptr);

extern "C"
void SuperResolution_setIterations(struct SuperResolutionPtr ptr, int val);

extern "C"
int SuperResolution_getIterations(struct SuperResolutionPtr ptr);

extern "C"
void SuperResolution_setTau(struct SuperResolutionPtr ptr, double val);

extern "C"
double SuperResolution_getTau(struct SuperResolutionPtr ptr);

extern "C"
void SuperResolution_setLabmda(struct SuperResolutionPtr ptr, double val);

extern "C"
double SuperResolution_getLabmda(struct SuperResolutionPtr ptr);

extern "C"
void SuperResolution_setAlpha(struct SuperResolutionPtr ptr, double val);

extern "C"
double SuperResolution_getAlpha(struct SuperResolutionPtr ptr);

extern "C"
void SuperResolution_setKernelSize(struct SuperResolutionPtr ptr, int val);

extern "C"
int SuperResolution_getKernelSize(struct SuperResolutionPtr ptr);

extern "C"
void SuperResolution_setBlurKernelSize(struct SuperResolutionPtr ptr, int val);

extern "C"
int SuperResolution_getBlurKernelSize(struct SuperResolutionPtr ptr);

extern "C"
void SuperResolution_setBlurSigma(struct SuperResolutionPtr ptr, double val);

extern "C"
double SuperResolution_getBlurSigma(struct SuperResolutionPtr ptr);

extern "C"
void SuperResolution_setTemporalAreaRadius(struct SuperResolutionPtr ptr, int val);

extern "C"
int SuperResolution_getTemporalAreaRadius(struct SuperResolutionPtr ptr);

struct FarnebackOpticalFlowPtr {
    void *ptr;

    inline superres::FarnebackOpticalFlow * operator->() { return static_cast<superres::FarnebackOpticalFlow *>(ptr); }
    inline FarnebackOpticalFlowPtr(superres::FarnebackOpticalFlow *ptr) { this->ptr = ptr; }
};

struct DualTVL1OpticalFlowPtr {
    void *ptr;

    inline superres::DualTVL1OpticalFlow * operator->() { return static_cast<superres::DualTVL1OpticalFlow *>(ptr); }
    inline DualTVL1OpticalFlowPtr(superres::DualTVL1OpticalFlow *ptr) { this->ptr = ptr; }
};

struct BroxOpticalFlowPtr {
    void *ptr;

    inline superres::BroxOpticalFlow * operator->() { return static_cast<superres::BroxOpticalFlow *>(ptr); }
    inline BroxOpticalFlowPtr(superres::BroxOpticalFlow *ptr) { this->ptr = ptr; }
};

struct PyrLKOpticalFlowPtr {
    void *ptr;

    inline superres::PyrLKOpticalFlow * operator->() { return static_cast<superres::PyrLKOpticalFlow *>(ptr); }
    inline PyrLKOpticalFlowPtr(superres::PyrLKOpticalFlow *ptr) { this->ptr = ptr; }
};

