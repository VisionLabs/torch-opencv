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

extern "C"
void SuperResolution_setOpticalFlow(struct SuperResolutionPtr ptr, struct DenseOpticalFlowExtPtr val);

extern "C"
struct DenseOpticalFlowExtPtr SuperResolution_getOpticalFlow(struct SuperResolutionPtr ptr);

struct DenseOpticalFlowExtPtr {
    void *ptr;

    inline superres::DenseOpticalFlowExt * operator->() { return static_cast<superres::DenseOpticalFlowExt *>(ptr); }
    inline DenseOpticalFlowExtPtr(superres::DenseOpticalFlowExt *ptr) { this->ptr = ptr; }
};

extern "C"
struct TensorArray DenseOpticalFlowExt_calc(
        struct DenseOpticalFlowExtPtr ptr, struct TensorWrapper frame0, struct TensorWrapper frame1,
        struct TensorWrapper flow1, struct TensorWrapper flow2);

extern "C"
void DenseOpticalFlowExt_collectGarbage(struct DenseOpticalFlowExtPtr ptr);

extern "C"
void SuperResolution_setOpticalFlow(struct SuperResolutionPtr ptr, struct DenseOpticalFlowExtPtr val);

extern "C"
struct DenseOpticalFlowExtPtr SuperResolution_getOpticalFlow(struct SuperResolutionPtr ptr);

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

// FarnebackOpticalFlow

extern "C"
struct FarnebackOpticalFlowPtr FarnebackOpticalFlow_ctor();

extern "C"
void FarnebackOpticalFlow_setPyrScale(struct FarnebackOpticalFlowPtr ptr, double val);

extern "C"
double FarnebackOpticalFlow_getPyrScale(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setLevelsNumber(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C"
int FarnebackOpticalFlow_getLevelsNumber(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setWindowSize(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C"
int FarnebackOpticalFlow_getWindowSize(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setIterations(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C"
int FarnebackOpticalFlow_getIterations(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setPolyN(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C"
int FarnebackOpticalFlow_getPolyN(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setPolySigma(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C"
double FarnebackOpticalFlow_getPolySigma(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setFlags(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C"
int FarnebackOpticalFlow_getFlags(struct FarnebackOpticalFlowPtr ptr);

extern "C"
struct DualTVL1OpticalFlowPtr createOptFlow_DualTVL1();

extern "C"
struct DualTVL1OpticalFlowPtr createOptFlow_DualTVL1_CUDA();

extern "C"
void DualTVL1OpticalFlow_setTau(struct DualTVL1OpticalFlowPtr ptr, double val);

extern "C"
double DualTVL1OpticalFlow_getTau(struct DualTVL1OpticalFlowPtr ptr);

extern "C"
void DualTVL1OpticalFlow_setLambda(struct DualTVL1OpticalFlowPtr ptr, double val);

extern "C"
double DualTVL1OpticalFlow_getLambda(struct DualTVL1OpticalFlowPtr ptr);

extern "C"
void DualTVL1OpticalFlow_setTheta(struct DualTVL1OpticalFlowPtr ptr, double val);

extern "C"
double DualTVL1OpticalFlow_getTheta(struct DualTVL1OpticalFlowPtr ptr);

extern "C"
void DualTVL1OpticalFlow_setScalesNumber(struct DualTVL1OpticalFlowPtr ptr, int val);

extern "C"
int DualTVL1OpticalFlow_getScalesNumber(struct DualTVL1OpticalFlowPtr ptr);

extern "C"
void DualTVL1OpticalFlow_setWarpingsNumber(struct DualTVL1OpticalFlowPtr ptr, int val);

extern "C"
int DualTVL1OpticalFlow_getWarpingsNumber(struct DualTVL1OpticalFlowPtr ptr);

extern "C"
void DualTVL1OpticalFlow_setEpsilon(struct DualTVL1OpticalFlowPtr ptr, double val);

extern "C"
double DualTVL1OpticalFlow_getEpsilon(struct DualTVL1OpticalFlowPtr ptr);

extern "C"
void DualTVL1OpticalFlow_setIterations(struct DualTVL1OpticalFlowPtr ptr, int val);

extern "C"
int DualTVL1OpticalFlow_getIterations(struct DualTVL1OpticalFlowPtr ptr);

extern "C"
void DualTVL1OpticalFlow_setUseInitialFlow(struct DualTVL1OpticalFlowPtr ptr, bool val);

extern "C"
bool DualTVL1OpticalFlow_getUseInitialFlow(struct DualTVL1OpticalFlowPtr ptr);

extern "C"
struct BroxOpticalFlowPtr createOptFlow_Brox_CUDA();

extern "C"
void BroxOpticalFlow_setAlpha(struct BroxOpticalFlowPtr ptr, double val);

extern "C"
double BroxOpticalFlow_getAlpha(struct BroxOpticalFlowPtr ptr);

extern "C"
void BroxOpticalFlow_setGamma(struct BroxOpticalFlowPtr ptr, double val);

extern "C"
double BroxOpticalFlow_getGamma(struct BroxOpticalFlowPtr ptr);

extern "C"
void BroxOpticalFlow_setScaleFactor(struct BroxOpticalFlowPtr ptr, double val);

extern "C"
double BroxOpticalFlow_getScaleFactor(struct BroxOpticalFlowPtr ptr);

extern "C"
void BroxOpticalFlow_setInnerIterations(struct BroxOpticalFlowPtr ptr, int val);

extern "C"
int BroxOpticalFlow_getInnerIterations(struct BroxOpticalFlowPtr ptr);

extern "C"
void BroxOpticalFlow_setOuterIterations(struct BroxOpticalFlowPtr ptr, int val);

extern "C"
int BroxOpticalFlow_getOuterIterations(struct BroxOpticalFlowPtr ptr);

extern "C"
void BroxOpticalFlow_setSolverIterations(struct BroxOpticalFlowPtr ptr, int val);

extern "C"
int BroxOpticalFlow_getSolverIterations(struct BroxOpticalFlowPtr ptr);

extern "C"
struct PyrLKOpticalFlowPtr createOptFlow_PyrLK_CUDA();

extern "C"
void PyrLKOpticalFlow_setWindowSize(struct PyrLKOpticalFlowPtr ptr, int val);

extern "C"
int PyrLKOpticalFlow_getWindowSize(struct PyrLKOpticalFlowPtr ptr);

extern "C"
void PyrLKOpticalFlow_setMaxLevel(struct PyrLKOpticalFlowPtr ptr, int val);

extern "C"
int PyrLKOpticalFlow_getMaxLevel(struct PyrLKOpticalFlowPtr ptr);

extern "C"
void PyrLKOpticalFlow_setIterations(struct PyrLKOpticalFlowPtr ptr, int val);

extern "C"
int PyrLKOpticalFlow_getIterations(struct PyrLKOpticalFlowPtr ptr);
