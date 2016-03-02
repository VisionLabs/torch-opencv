#include <cudabgsegm.hpp>

extern "C"
struct BackgroundSubtractorMOGPtr BackgroundSubtractorMOG_ctorCuda(
        int History, int NMixtures, double BackgroundRatio, double NoiseSigma)
{
    return rescueObjectFromPtr(cuda::createBackgroundSubtractorMOG(
            History, NMixtures, BackgroundRatio, NoiseSigma));
}

extern "C"
struct TensorWrapper BackgroundSubtractorMOG_applyCuda(struct cutorchInfo info,
                                                    struct BackgroundSubtractorMOGPtr ptr, struct TensorWrapper image,
                                                    struct TensorWrapper fgmask, double learningRate)
{
    cuda::GpuMat retval;
    if (!fgmask.isNull()) retval = fgmask.toGpuMat();
    ptr->apply(image.toGpuMat(), retval, learningRate, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper BackgroundSubtractorMOG_getBackgroundImageCuda(
        struct cutorchInfo info, struct BackgroundSubtractorMOGPtr ptr,
        struct TensorWrapper backgroundImage)
{
    cuda::GpuMat retval;
    if (!backgroundImage.isNull()) retval = backgroundImage.toGpuMat();
    ptr->getBackgroundImage(retval, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
void BackgroundSubtractorMOG_setHistoryCuda(struct BackgroundSubtractorMOGPtr ptr, int val)
{
    ptr->setHistory(val);
}

extern "C"
int BackgroundSubtractorMOG_getHistoryCuda(struct BackgroundSubtractorMOGPtr ptr)
{
    return ptr->getHistory();
}

extern "C"
void BackgroundSubtractorMOG_setNMixturesCuda(struct BackgroundSubtractorMOGPtr ptr, int val)
{
    ptr->setNMixtures(val);
}

extern "C"
int BackgroundSubtractorMOG_getNMixturesCuda(struct BackgroundSubtractorMOGPtr ptr)
{
    return ptr->getNMixtures();
}

extern "C"
void BackgroundSubtractorMOG_setBackgroundRatioCuda(struct BackgroundSubtractorMOGPtr ptr, double val)
{
    ptr->setBackgroundRatio(val);
}

extern "C"
double BackgroundSubtractorMOG_getBackgroundRatioCuda(struct BackgroundSubtractorMOGPtr ptr)
{
    return ptr->getBackgroundRatio();
}

extern "C"
void BackgroundSubtractorMOG_setNoiseSigmaCuda(struct BackgroundSubtractorMOGPtr ptr, double val)
{
    ptr->setNoiseSigma(val);
}

extern "C"
double BackgroundSubtractorMOG_getNoiseSigmaCuda(struct BackgroundSubtractorMOGPtr ptr)
{
    return ptr->getNoiseSigma();
}

extern "C"
struct BackgroundSubtractorMOG2Ptr BackgroundSubtractorMOG2_ctorCuda(
        int history, double varThreshold, bool detectShadows)
{
    return rescueObjectFromPtr(cuda::createBackgroundSubtractorMOG2(
            history, varThreshold, detectShadows));
}

extern "C"
struct TensorWrapper BackgroundSubtractorMOG2_applyCuda(struct cutorchInfo info,
        struct BackgroundSubtractorMOG2Ptr ptr, struct TensorWrapper image,
        struct TensorWrapper fgmask, double learningRate)
{
    cuda::GpuMat retval;
    if (!fgmask.isNull()) retval = fgmask.toGpuMat();
    ptr->apply(image.toGpuMat(), retval, learningRate, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper BackgroundSubtractorMOG2_getBackgroundImageCuda(
        struct cutorchInfo info, struct BackgroundSubtractorMOG2Ptr ptr,
        struct TensorWrapper backgroundImage)
{
    cuda::GpuMat retval;
    if (!backgroundImage.isNull()) retval = backgroundImage.toGpuMat();
    ptr->getBackgroundImage(retval, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
int BackgroundSubtractorMOG2_getHistoryCuda(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getHistory();
}

extern "C"
void BackgroundSubtractorMOG2_setHistoryCuda(struct BackgroundSubtractorMOG2Ptr ptr, int history)
{
    ptr->setHistory(history);
}

extern "C"
int BackgroundSubtractorMOG2_getNMixturesCuda(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getNMixtures();
}

extern "C"
void BackgroundSubtractorMOG2_setNMixturesCuda(struct BackgroundSubtractorMOG2Ptr ptr, int nmixtures)
{
    ptr->setNMixtures(nmixtures);
}

extern "C"
int BackgroundSubtractorMOG2_getShadowValueCuda(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getShadowValue();
}

extern "C"
void BackgroundSubtractorMOG2_setShadowValueCuda(struct BackgroundSubtractorMOG2Ptr ptr, int shadow_value)
{
    ptr->setShadowValue(shadow_value);
}

extern "C"
double BackgroundSubtractorMOG2_getBackgroundRatioCuda(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getBackgroundRatio();
}

extern "C"
void BackgroundSubtractorMOG2_setBackgroundRatioCuda(struct BackgroundSubtractorMOG2Ptr ptr, double ratio)
{
    ptr->setBackgroundRatio(ratio);
}

extern "C"
double BackgroundSubtractorMOG2_getVarThresholdCuda(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getVarThreshold();
}

extern "C"
void BackgroundSubtractorMOG2_setVarThresholdCuda(struct BackgroundSubtractorMOG2Ptr ptr, double varThreshold)
{
    ptr->setVarThreshold(varThreshold);
}

extern "C"
double BackgroundSubtractorMOG2_getVarThresholdGenCuda(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getVarThresholdGen();
}

extern "C"
void BackgroundSubtractorMOG2_setVarThresholdGenCuda(struct BackgroundSubtractorMOG2Ptr ptr, double varThresholdGen)
{
    ptr->setVarThresholdGen(varThresholdGen);
}

extern "C"
double BackgroundSubtractorMOG2_getVarInitCuda(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getVarInit();
}

extern "C"
void BackgroundSubtractorMOG2_setVarInitCuda(struct BackgroundSubtractorMOG2Ptr ptr, double varInit)
{
    ptr->setVarInit(varInit);
}

extern "C"
double BackgroundSubtractorMOG2_getVarMinCuda(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getVarMin();
}

extern "C"
void BackgroundSubtractorMOG2_setVarMinCuda(struct BackgroundSubtractorMOG2Ptr ptr, double varMin)
{
    ptr->setVarMin(varMin);
}

extern "C"
double BackgroundSubtractorMOG2_getVarMaxCuda(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getVarMax();
}

extern "C"
void BackgroundSubtractorMOG2_setVarMaxCuda(struct BackgroundSubtractorMOG2Ptr ptr, double varMax)
{
    ptr->setVarMax(varMax);
}

extern "C"
bool BackgroundSubtractorMOG2_getDetectShadowsCuda(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getDetectShadows();
}

extern "C"
void BackgroundSubtractorMOG2_setDetectShadowsCuda(struct BackgroundSubtractorMOG2Ptr ptr, bool detectShadows)
{
    ptr->setDetectShadows(detectShadows);
}

extern "C"
double BackgroundSubtractorMOG2_getComplexityReductionThresholdCuda(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getComplexityReductionThreshold();
}

extern "C"
void BackgroundSubtractorMOG2_setComplexityReductionThresholdCuda(struct BackgroundSubtractorMOG2Ptr ptr, double ct)
{
    ptr->setComplexityReductionThreshold(ct);
}

extern "C"
double BackgroundSubtractorMOG2_getShadowThresholdCuda(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getShadowThreshold();
}

extern "C"
void BackgroundSubtractorMOG2_setShadowThresholdCuda(struct BackgroundSubtractorMOG2Ptr ptr, double shadowThreshold)
{
    ptr->setShadowThreshold(shadowThreshold);
}