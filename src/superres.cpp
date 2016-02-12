#include <superres.hpp>

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
void FrameSource_dtor(struct FrameSourcePtr ptr)
{
    delete static_cast<superres::FrameSource *>(ptr.ptr);
}

extern "C"
struct TensorWrapper FrameSource_nextFrame(struct FrameSourcePtr ptr, struct TensorWrapper frame)
{
    MatT frame_mat;
    if(!frame.isNull()) frame_mat = frame.toMatT();
    ptr->nextFrame(frame_mat);
    return TensorWrapper(frame_mat);
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
    MatT frame_mat;
    if(!frame.isNull()) frame_mat = frame.toMatT();
    ptr->nextFrame(frame_mat);
    return TensorWrapper(frame_mat);
}

extern "C"
void SuperResolution_reset(struct SuperResolutionPtr ptr)
{
    ptr->reset();
}

extern "C"
void SuperResolution_setInput(struct SuperResolutionPtr ptr, struct FrameSourcePtr frameSource)
{
    cv::Ptr<superres::FrameSource> tempPtr(
            static_cast<superres::FrameSource *>(frameSource.ptr));
    rescueObjectFromPtr(tempPtr);
    ptr->setInput(tempPtr);
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

/* A dirty and unsafe hack for resolving virtual inheritance issue.
 * If it stops working one day, we will have to:
 *
 * 1) create an enum that names all the DenseOpticalFlowExt virtual children (like FarnebackOpticalFlow)
 * 2) when creating such a child (createOptFlow_Farneback), return not only void * ptr to it, but also enum's value
 * 3) put this enum value to a Lua class (along with void * ptr)
 * 4) in SuperResolution_setOpticalFlow, cast the void * ptr to one of these children types depending on the enum value
 * */
class FakeOpticalFlow : public virtual superres::DenseOpticalFlowExt {};

extern "C"
void SuperResolution_setOpticalFlow(struct SuperResolutionPtr ptr, struct DenseOpticalFlowExtPtr val)
{
    cv::Ptr<FakeOpticalFlow> tempPtr(
            static_cast<FakeOpticalFlow *>(val.ptr));
    rescueObjectFromPtr(tempPtr);
    ptr->setOpticalFlow(tempPtr);
}

extern "C"
struct DenseOpticalFlowExtPtr SuperResolution_getOpticalFlow(struct SuperResolutionPtr ptr)
{
    return rescueObjectFromPtr(ptr->getOpticalFlow());
}

// DenseOpticalFlowExt

extern "C"
struct TensorArray DenseOpticalFlowExt_calc(
        struct DenseOpticalFlowExtPtr ptr, struct TensorWrapper frame0, struct TensorWrapper frame1,
        struct TensorWrapper flow1, struct TensorWrapper flow2)
{
    std::vector<MatT> retval(2);
    if (!flow1.isNull()) retval[0] = flow1.toMatT();
    if (!flow2.isNull()) retval[1] = flow2.toMatT();
    ptr->calc(frame0.toMat(), frame1.toMat(), retval[0], retval[1]);
    return TensorArray(retval);
}

extern "C"
void DenseOpticalFlowExt_collectGarbage(struct DenseOpticalFlowExtPtr ptr)
{
    ptr->collectGarbage();
}

// FarnebackOpticalFlow

extern "C"
struct FarnebackOpticalFlowPtr createOptFlow_Farneback()
{
    return rescueObjectFromPtr(superres::createOptFlow_Farneback());
}

extern "C"
struct FarnebackOpticalFlowPtr createOptFlow_Farneback_CUDA()
{
    return rescueObjectFromPtr(superres::createOptFlow_Farneback_CUDA());
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

// DualTVL1OpticalFlow

extern "C"
struct DualTVL1OpticalFlowPtr createOptFlow_DualTVL1()
{
    return rescueObjectFromPtr(superres::createOptFlow_DualTVL1());
}

extern "C"
struct DualTVL1OpticalFlowPtr createOptFlow_DualTVL1_CUDA()
{
    return rescueObjectFromPtr(superres::createOptFlow_DualTVL1_CUDA());
}

extern "C"
void DualTVL1OpticalFlow_setTau(struct DualTVL1OpticalFlowPtr ptr, double val)
{
    ptr->setTau(val);
}

extern "C"
double DualTVL1OpticalFlow_getTau(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getTau();
}

extern "C"
void DualTVL1OpticalFlow_setLambda(struct DualTVL1OpticalFlowPtr ptr, double val)
{
    ptr->setLambda(val);
}

extern "C"
double DualTVL1OpticalFlow_getLambda(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getLambda();
}

extern "C"
void DualTVL1OpticalFlow_setTheta(struct DualTVL1OpticalFlowPtr ptr, double val)
{
    ptr->setTheta(val);
}

extern "C"
double DualTVL1OpticalFlow_getTheta(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getTheta();
}

extern "C"
void DualTVL1OpticalFlow_setScalesNumber(struct DualTVL1OpticalFlowPtr ptr, int val)
{
    ptr->setScalesNumber(val);
}

extern "C"
int DualTVL1OpticalFlow_getScalesNumber(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getScalesNumber();
}

extern "C"
void DualTVL1OpticalFlow_setWarpingsNumber(struct DualTVL1OpticalFlowPtr ptr, int val)
{
    ptr->setWarpingsNumber(val);
}

extern "C"
int DualTVL1OpticalFlow_getWarpingsNumber(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getWarpingsNumber();
}

extern "C"
void DualTVL1OpticalFlow_setEpsilon(struct DualTVL1OpticalFlowPtr ptr, double val)
{
    ptr->setEpsilon(val);
}

extern "C"
double DualTVL1OpticalFlow_getEpsilon(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getEpsilon();
}

extern "C"
void DualTVL1OpticalFlow_setIterations(struct DualTVL1OpticalFlowPtr ptr, int val)
{
    ptr->setIterations(val);
}

extern "C"
int DualTVL1OpticalFlow_getIterations(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getIterations();
}

extern "C"
void DualTVL1OpticalFlow_setUseInitialFlow(struct DualTVL1OpticalFlowPtr ptr, bool val)
{
    ptr->setUseInitialFlow(val);
}

extern "C"
bool DualTVL1OpticalFlow_getUseInitialFlow(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getUseInitialFlow();
}

// BroxOpticalFlow

extern "C"
struct BroxOpticalFlowPtr createOptFlow_Brox_CUDA()
{
    return rescueObjectFromPtr(superres::createOptFlow_Brox_CUDA());
}


extern "C"
void BroxOpticalFlow_setAlpha(struct BroxOpticalFlowPtr ptr, double val)
{
    ptr->setAlpha(val);
}

extern "C"
double BroxOpticalFlow_getAlpha(struct BroxOpticalFlowPtr ptr)
{
    return ptr->getAlpha();
}

extern "C"
void BroxOpticalFlow_setGamma(struct BroxOpticalFlowPtr ptr, double val)
{
    ptr->setGamma(val);
}

extern "C"
double BroxOpticalFlow_getGamma(struct BroxOpticalFlowPtr ptr)
{
    return ptr->getGamma();
}

extern "C"
void BroxOpticalFlow_setScaleFactor(struct BroxOpticalFlowPtr ptr, double val)
{
    ptr->setScaleFactor(val);
}

extern "C"
double BroxOpticalFlow_getScaleFactor(struct BroxOpticalFlowPtr ptr)
{
    return ptr->getScaleFactor();
}

extern "C"
void BroxOpticalFlow_setInnerIterations(struct BroxOpticalFlowPtr ptr, int val)
{
    ptr->setInnerIterations(val);
}

extern "C"
int BroxOpticalFlow_getInnerIterations(struct BroxOpticalFlowPtr ptr)
{
    return ptr->getInnerIterations();
}

extern "C"
void BroxOpticalFlow_setOuterIterations(struct BroxOpticalFlowPtr ptr, int val)
{
    ptr->setOuterIterations(val);
}

extern "C"
int BroxOpticalFlow_getOuterIterations(struct BroxOpticalFlowPtr ptr)
{
    return ptr->getOuterIterations();
}

extern "C"
void BroxOpticalFlow_setSolverIterations(struct BroxOpticalFlowPtr ptr, int val)
{
    ptr->setSolverIterations(val);
}

extern "C"
int BroxOpticalFlow_getSolverIterations(struct BroxOpticalFlowPtr ptr)
{
    return ptr->getSolverIterations();
}

// PyrLKOpticalFlow

extern "C"
struct PyrLKOpticalFlowPtr createOptFlow_PyrLK_CUDA()
{
    return rescueObjectFromPtr(superres::createOptFlow_PyrLK_CUDA());
}

extern "C"
void PyrLKOpticalFlow_setWindowSize(struct PyrLKOpticalFlowPtr ptr, int val)
{
    ptr->setWindowSize(val);
}

extern "C"
int PyrLKOpticalFlow_getWindowSize(struct PyrLKOpticalFlowPtr ptr)
{
    return ptr->getWindowSize();
}

extern "C"
void PyrLKOpticalFlow_setMaxLevel(struct PyrLKOpticalFlowPtr ptr, int val)
{
    ptr->setMaxLevel(val);
}

extern "C"
int PyrLKOpticalFlow_getMaxLevel(struct PyrLKOpticalFlowPtr ptr)
{
    return ptr->getMaxLevel();
}

extern "C"
void PyrLKOpticalFlow_setIterations(struct PyrLKOpticalFlowPtr ptr, int val)
{
    ptr->setIterations(val);
}

extern "C"
int PyrLKOpticalFlow_getIterations(struct PyrLKOpticalFlowPtr ptr)
{
    return ptr->getIterations();
}