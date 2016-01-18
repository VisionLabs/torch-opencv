#include <cudabgsegm.hpp>

extern "C"
struct BackgroundSubtractorMOGPtr BackgroundSubtractorMOG_ctor(
        int History, int NMixtures, double BackgroundRatio, double NoiseSigma)
{
    return rescueObjectFromPtr(cuda::createBackgroundSubtractorMOG(
            History, NMixtures, BackgroundRatio, NoiseSigma));
}

extern "C"
struct TensorWrapper BackgroundSubtractorMOG_apply(THCState *state,
                                                    struct BackgroundSubtractorMOGPtr ptr, struct TensorWrapper image,
                                                    struct TensorWrapper fgmask, double learningRate)
{
    cuda::GpuMat retval;
    if (!fgmask.isNull()) retval = fgmask.toGpuMat();
    ptr->apply(image.toGpuMat(), retval, learningRate);
    return TensorWrapper(retval, state);
}

extern "C"
struct TensorWrapper BackgroundSubtractorMOG_getBackgroundImage(
        THCState *state, struct BackgroundSubtractorMOGPtr ptr,
        struct TensorWrapper backgroundImage)
{
    cuda::GpuMat retval;
    if (!backgroundImage.isNull()) retval = backgroundImage.toGpuMat();
    ptr->getBackgroundImage(retval);
    return TensorWrapper(retval, state);
}

extern "C"
void BackgroundSubtractorMOG_setHistory(struct BackgroundSubtractorMOGPtr ptr, int val)
{
    ptr->setHistory(val);
}

extern "C"
int BackgroundSubtractorMOG_getHistory(struct BackgroundSubtractorMOGPtr ptr)
{
    return ptr->getHistory();
}

extern "C"
void BackgroundSubtractorMOG_setNMixtures(struct BackgroundSubtractorMOGPtr ptr, int val)
{
    ptr->setNMixtures(val);
}

extern "C"
int BackgroundSubtractorMOG_getNMixtures(struct BackgroundSubtractorMOGPtr ptr)
{
    return ptr->getNMixtures();
}

extern "C"
void BackgroundSubtractorMOG_setBackgroundRatio(struct BackgroundSubtractorMOGPtr ptr, double val)
{
    ptr->setBackgroundRatio(val);
}

extern "C"
double BackgroundSubtractorMOG_getBackgroundRatio(struct BackgroundSubtractorMOGPtr ptr)
{
    return ptr->getBackgroundRatio();
}

extern "C"
void BackgroundSubtractorMOG_setNoiseSigma(struct BackgroundSubtractorMOGPtr ptr, double val)
{
    ptr->setNoiseSigma(val);
}

extern "C"
double BackgroundSubtractorMOG_getNoiseSigma(struct BackgroundSubtractorMOGPtr ptr)
{
    return ptr->getNoiseSigma();
}

extern "C"
struct BackgroundSubtractorMOG2Ptr BackgroundSubtractorMOG2_ctor(
        int history, double varThreshold, bool detectShadows)
{
    return rescueObjectFromPtr(cuda::createBackgroundSubtractorMOG2(
            history, varThreshold, detectShadows));
}

extern "C"
struct TensorWrapper BackgroundSubtractorMOG2_apply(THCState *state,
        struct BackgroundSubtractorMOG2Ptr ptr, struct TensorWrapper image,
        struct TensorWrapper fgmask, double learningRate)
{
    cuda::GpuMat retval;
    if (!fgmask.isNull()) retval = fgmask.toGpuMat();
    ptr->apply(image.toGpuMat(), retval, learningRate);
    return TensorWrapper(retval, state);
}

extern "C"
struct TensorWrapper BackgroundSubtractorMOG2_getBackgroundImage(
        THCState *state, struct BackgroundSubtractorMOG2Ptr ptr,
        struct TensorWrapper backgroundImage)
{
    cuda::GpuMat retval;
    if (!backgroundImage.isNull()) retval = backgroundImage.toGpuMat();
    ptr->getBackgroundImage(retval);
    return TensorWrapper(retval, state);
}

extern "C"
int BackgroundSubtractorMOG2_getHistory(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getHistory();
}

extern "C"
void BackgroundSubtractorMOG2_setHistory(struct BackgroundSubtractorMOG2Ptr ptr, int history)
{
    ptr->setHistory(history);
}

extern "C"
int BackgroundSubtractorMOG2_getNMixtures(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getNMixtures();
}

extern "C"
void BackgroundSubtractorMOG2_setNMixtures(struct BackgroundSubtractorMOG2Ptr ptr, int nmixtures)
{
    ptr->setNMixtures(nmixtures);
}

extern "C"
int BackgroundSubtractorMOG2_getShadowValue(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getShadowValue();
}

extern "C"
void BackgroundSubtractorMOG2_setShadowValue(struct BackgroundSubtractorMOG2Ptr ptr, int shadow_value)
{
    ptr->setShadowValue(shadow_value);
}

extern "C"
double BackgroundSubtractorMOG2_getBackgroundRatio(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getBackgroundRatio();
}

extern "C"
void BackgroundSubtractorMOG2_setBackgroundRatio(struct BackgroundSubtractorMOG2Ptr ptr, double ratio)
{
    ptr->setBackgroundRatio(ratio);
}

extern "C"
double BackgroundSubtractorMOG2_getVarThreshold(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getVarThreshold();
}

extern "C"
void BackgroundSubtractorMOG2_setVarThreshold(struct BackgroundSubtractorMOG2Ptr ptr, double varThreshold)
{
    ptr->setVarThreshold(varThreshold);
}

extern "C"
double BackgroundSubtractorMOG2_getVarThresholdGen(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getVarThresholdGen();
}

extern "C"
void BackgroundSubtractorMOG2_setVarThresholdGen(struct BackgroundSubtractorMOG2Ptr ptr, double varThresholdGen)
{
    ptr->setVarThresholdGen(varThresholdGen);
}

extern "C"
double BackgroundSubtractorMOG2_getVarInit(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getVarInit();
}

extern "C"
void BackgroundSubtractorMOG2_setVarInit(struct BackgroundSubtractorMOG2Ptr ptr, double varInit)
{
    ptr->setVarInit(varInit);
}

extern "C"
double BackgroundSubtractorMOG2_getVarMin(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getVarMin();
}

extern "C"
void BackgroundSubtractorMOG2_setVarMin(struct BackgroundSubtractorMOG2Ptr ptr, double varMin)
{
    ptr->setVarMin(varMin);
}

extern "C"
double BackgroundSubtractorMOG2_getVarMax(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getVarMax();
}

extern "C"
void BackgroundSubtractorMOG2_setVarMax(struct BackgroundSubtractorMOG2Ptr ptr, double varMax)
{
    ptr->setVarMax(varMax);
}

extern "C"
bool BackgroundSubtractorMOG2_getDetectShadows(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getDetectShadows();
}

extern "C"
void BackgroundSubtractorMOG2_setDetectShadows(struct BackgroundSubtractorMOG2Ptr ptr, bool detectShadows)
{
    ptr->setDetectShadows(detectShadows);
}

extern "C"
double BackgroundSubtractorMOG2_getComplexityReductionThreshold(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getComplexityReductionThreshold();
}

extern "C"
void BackgroundSubtractorMOG2_setComplexityReductionThreshold(struct BackgroundSubtractorMOG2Ptr ptr, double ct)
{
    ptr->setComplexityReductionThreshold(ct);
}

extern "C"
double BackgroundSubtractorMOG2_getShadowThreshold(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getShadowThreshold();
}

extern "C"
void BackgroundSubtractorMOG2_setShadowThreshold(struct BackgroundSubtractorMOG2Ptr ptr, double shadowThreshold)
{
    ptr->setShadowThreshold(shadowThreshold);
}