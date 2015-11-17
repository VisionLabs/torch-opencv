#include <video.hpp>

// BackgroundSubtractor

extern "C" struct TensorWrapper BackgroundSubtractor_apply(struct BackgroundSubtractorPtr ptr, struct TensorWrapper image,
                struct TensorWrapper fgmast, double learningRate)
{
    if (fgmast.isNull()) {
        cv::Mat retval;
        ptr->apply(image.toMat(), retval, learningRate);
        return TensorWrapper(retval);
    } else {
        ptr->apply(image.toMat(), fgmast.toMat(), learningRate);
    }
    return fgmast;
}

extern "C" struct TensorWrapper BackgroundSubtractor_getBackgroundImage(struct BackgroundSubtractorPtr ptr,
                                    struct TensorWrapper backgroundImage)
{
    ptr->getBackgroundImage(backgroundImage.toMat());
    return backgroundImage;
}

extern "C" struct BackgroundSubtractorMOG2Ptr BackgroundSubtractorMOG2_ctor(int history, double varThreshold, bool detectShadows)
{
        return rescueObjectFromPtr(cv::createBackgroundSubtractorMOG2(history, varThreshold, detectShadows));

}

extern "C" int BackgroundSubtractorMOG2_getHistory(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getHistory();
}

extern "C" void BackgroundSubtractorMOG2_setHistory(struct BackgroundSubtractorMOG2Ptr ptr, int history)
{
    ptr->setHistory(history);
}
extern "C" int BackgroundSubtractorMOG2_getNMixtures(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getNMixtures();
}

extern "C" void BackgroundSubtractorMOG2_setNMixtures(struct BackgroundSubtractorMOG2Ptr ptr, int nmixtures)
{
    ptr->setNMixtures(nmixtures);
}
extern "C" int BackgroundSubtractorMOG2_getShadowValue(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getShadowValue();
}

extern "C" void BackgroundSubtractorMOG2_setShadowValue(struct BackgroundSubtractorMOG2Ptr ptr, int shadow_value)
{
    ptr->setShadowValue(shadow_value);
}

extern "C" double BackgroundSubtractorMOG2_getBackgroundRatio(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getBackgroundRatio();
}

extern "C" void BackgroundSubtractorMOG2_setBackgroundRatio(struct BackgroundSubtractorMOG2Ptr ptr, double ratio)
{
    ptr->setBackgroundRatio(ratio);
}
extern "C" double BackgroundSubtractorMOG2_getVarThreshold(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getVarThreshold();
}

extern "C" void BackgroundSubtractorMOG2_setVarThreshold(struct BackgroundSubtractorMOG2Ptr ptr, double varThreshold)
{
    ptr->setVarThreshold(varThreshold);
}
extern "C" double BackgroundSubtractorMOG2_getVarThresholdGen(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getVarThresholdGen();
}

extern "C" void BackgroundSubtractorMOG2_setVarThresholdGen(struct BackgroundSubtractorMOG2Ptr ptr, double varThresholdGen)
{
    ptr->setVarThresholdGen(varThresholdGen);
}
extern "C" double BackgroundSubtractorMOG2_getVarInit(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getVarInit();
}

extern "C" void BackgroundSubtractorMOG2_setVarInit(struct BackgroundSubtractorMOG2Ptr ptr, double varInit)
{
    ptr->setVarInit(varInit);
}
extern "C" double BackgroundSubtractorMOG2_getVarMin(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getVarMin();
}

extern "C" void BackgroundSubtractorMOG2_setVarMin(struct BackgroundSubtractorMOG2Ptr ptr, double varMin)
{
    ptr->setVarMin(varMin);
}
extern "C" double BackgroundSubtractorMOG2_getVarMax(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getVarMax();
}

extern "C" void BackgroundSubtractorMOG2_setVarMax(struct BackgroundSubtractorMOG2Ptr ptr, double varMax)
{
    ptr->setVarMax(varMax);
}

extern "C" bool BackgroundSubtractorMOG2_getDetectShadows(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getDetectShadows();
}

extern "C" void BackgroundSubtractorMOG2_setDetectShadows(struct BackgroundSubtractorMOG2Ptr ptr, bool detectShadows)
{
    ptr->setDetectShadows(detectShadows);
}

extern "C" double BackgroundSubtractorMOG2_getComplexityReductionThreshold(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getComplexityReductionThreshold();
}

extern "C" void BackgroundSubtractorMOG2_setComplexityReductionThreshold(struct BackgroundSubtractorMOG2Ptr ptr, double ct)
{
    ptr->setComplexityReductionThreshold(ct);
}
extern "C" double BackgroundSubtractorMOG2_getShadowThreshold(struct BackgroundSubtractorMOG2Ptr ptr)
{
    return ptr->getShadowThreshold();
}

extern "C" void BackgroundSubtractorMOG2_setShadowThreshold(struct BackgroundSubtractorMOG2Ptr ptr, double shadowThreshold)
{
    ptr->setShadowThreshold(shadowThreshold);
}

// BackgroundSubtractorKNN

extern "C" struct BackgroundSubtractorKNNPtr BackgroundSubtractorKNN_ctor(int history, double dist2Threshold, bool detectShadows)
{
    return rescueObjectFromPtr(cv::createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows));
}

extern "C" int BackgroundSubtractorKNN_getHistory(struct BackgroundSubtractorKNNPtr ptr)
{
    return ptr->getHistory();
}

extern "C" void BackgroundSubtractorKNN_setHistory(struct BackgroundSubtractorKNNPtr ptr, int history)
{
    ptr->setHistory(history);
}
extern "C" int BackgroundSubtractorKNN_getNSamples(struct BackgroundSubtractorKNNPtr ptr)
{
    return ptr->getNSamples();
}

extern "C" void BackgroundSubtractorKNN_setNSamples(struct BackgroundSubtractorKNNPtr ptr, int nSamples)
{
    ptr->setNSamples(nSamples);
}
extern "C" int BackgroundSubtractorKNN_getkNNSamples(struct BackgroundSubtractorKNNPtr ptr)
{
    return ptr->getkNNSamples();
}

extern "C" void BackgroundSubtractorKNN_setkNNSamples(struct BackgroundSubtractorKNNPtr ptr, int kNNSamples)
{
    ptr->setkNNSamples(kNNSamples);
}
extern "C" int BackgroundSubtractorKNN_getShadowValue(struct BackgroundSubtractorKNNPtr ptr)
{
    return ptr->getShadowValue();
}

extern "C" void BackgroundSubtractorKNN_setShadowValue(struct BackgroundSubtractorKNNPtr ptr, int shadowValue)
{
    ptr->setShadowValue(shadowValue);
}

extern "C" double BackgroundSubtractorKNN_getDist2Threshold(struct BackgroundSubtractorKNNPtr ptr)
{
    return ptr->getDist2Threshold();
}

extern "C" void BackgroundSubtractorKNN_setDist2Threshold(struct BackgroundSubtractorKNNPtr ptr, double dist2Threshold)
{
    ptr->setDist2Threshold(dist2Threshold);
}
extern "C" double BackgroundSubtractorKNN_getShadowThreshold(struct BackgroundSubtractorKNNPtr ptr)
{
    return ptr->getShadowThreshold();
}

extern "C" void BackgroundSubtractorKNN_setShadowThreshold(struct BackgroundSubtractorKNNPtr ptr, double shadowThreshold)
{
    ptr->setShadowThreshold(shadowThreshold);
}

extern "C" bool BackgroundSubtractorKNN_getDetectShadows(struct BackgroundSubtractorKNNPtr ptr)
{
    return ptr->getDetectShadows();
}

extern "C" void BackgroundSubtractorKNN_setDetectShadows(struct BackgroundSubtractorKNNPtr ptr, bool detectShadows)
{
    ptr->setDetectShadows(detectShadows);
}