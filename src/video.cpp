#include <video.hpp>

extern "C" struct RotatedRectPlusRect CamShift(struct TensorWrapper probImage, struct RectWrapper window, struct TermCriteriaWrapper criteria)
{
    struct RotatedRectPlusRect retval;
    cv::Rect windowRect = window;
    retval.rotrect = cv::CamShift(probImage.toMat(), windowRect, criteria);
    new (&retval.rect) RectWrapper(windowRect);
    return retval;
}

extern "C" struct RectPlusInt meanShift(struct TensorWrapper probImage, struct RectWrapper window,
                        struct TermCriteriaWrapper criteria)
{
    RectPlusInt retval;
    cv::Rect rect = window;
    retval.val = cv::meanShift(probImage.toMat(), rect, criteria);
    new (&retval.rect) RectWrapper(rect);
    return retval;
}

extern "C" struct TensorArrayPlusInt buildOpticalFlowPyramid(struct TensorWrapper img, struct TensorArray pyramid,
                        struct SizeWrapper winSize, int maxLevel, bool withDerivatives, int pyrBorder,
                        int derivBorder, bool tryReuseInputImage)
{
    std::vector<cv::Mat> output(pyramid);
    TensorArrayPlusInt retval;
    retval.val = cv::buildOpticalFlowPyramid(img.toMat(), output, winSize, maxLevel,
                    withDerivatives, pyrBorder, derivBorder, tryReuseInputImage);
    retval.tensors = TensorArray(output);
    return retval;
}

extern "C" struct TensorPlusTensorPlusTensor calcOpticalFlowPyrLK(struct TensorWrapper prevImg,
                        struct TensorWrapper nextImg, struct TensorWrapper prevPts,
                        struct TensorWrapper nextPts, struct TensorWrapper status,
                        struct TensorWrapper err, struct SizeWrapper winSize, int maxLevel,
                        struct TermCriteriaWrapper criteria, int flags, double minEigThreshold)
{
    cv::Mat statusMat;
    if (!status.isNull())
        statusMat = status.toMat();
    cv::Mat errMat;
    if (!err.isNull())
        errMat = err.toMat();

    cv::calcOpticalFlowPyrLK(prevImg.toMat(), nextImg.toMat(), prevPts.toMat(), nextPts.toMat(),
                    statusMat, errMat, winSize, maxLevel,
                    criteria.orDefault(cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01)),
                    flags, minEigThreshold);
    
    struct TensorPlusTensorPlusTensor retval;
    retval.tensor = nextPts;
    retval.status = TensorWrapper(statusMat);
    retval.err = TensorWrapper(errMat);
    return retval;
}

extern "C" struct TensorWrapper calcOpticalFlowFarneback(struct TensorWrapper prev, struct TensorWrapper next,
                        struct TensorWrapper flow, double pyr_scale, int levels, int winsize,
                        int iterations, int poly_n, double poly_sigma, int flags)
{
    cv::calcOpticalFlowFarneback(prev.toMat(), next.toMat(), flow.toMat(), pyr_scale,
                    levels, winsize, iterations, poly_n, poly_sigma, flags);
    return flow;
}

extern "C" struct TensorWrapper estimateRigidTransform(struct TensorWrapper src, struct TensorWrapper dst, bool fullAffine)
{
    cv::Mat retval;
    retval = cv::estimateRigidTransform(src.toMat(), dst.toMat(), fullAffine);
    return TensorWrapper(retval);
}

extern "C" struct TensorPlusDouble findTransformECC(struct TensorWrapper templateImage, struct TensorWrapper inputImage,
                        struct TensorWrapper warpMatrix, int motionType, struct TermCriteriaWrapper criteria,
                        struct TensorWrapper inputMask)
{
    struct TensorPlusDouble retval;
    retval.val = cv::findTransformECC(templateImage.toMat(), inputImage.toMat(), warpMatrix.toMat(),
                    motionType, criteria.orDefault(cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 50, 0.001)),
                    TO_MAT_OR_NOARRAY(inputMask));
    retval.tensor = TensorWrapper(warpMatrix);
    return retval;
}

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
    if (backgroundImage.isNull()) {
        cv::Mat retval;
        ptr->getBackgroundImage(retval);
        return TensorWrapper(retval);
    }
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

// KalmanFilter

extern "C" struct KalmanFilterPtr KalmanFilter_ctor_default()
{
    return new cv::KalmanFilter();
}

extern "C" struct KalmanFilterPtr KalmanFilter_ctor(int dynamParams, int measureParams, int controlParams, int type)
{
    return new cv::KalmanFilter(dynamParams, measureParams, controlParams, type);
}

extern "C" void KalmanFilter_dtor(struct KalmanFilterPtr ptr)
{
    delete static_cast<cv::KalmanFilter *>(ptr.ptr);
}

extern "C" void KalmanFilter_init(struct KalmanFilterPtr ptr, int dynamParams, int measureParams, int controlParams, int type)
{
    ptr->init(dynamParams, measureParams, controlParams, type);
}

extern "C" struct TensorWrapper KalmanFilter_predict(struct KalmanFilterPtr ptr, struct TensorWrapper control)
{
    cv::Mat retval = cv::Mat();
    if (!control.isNull())
        retval = control.toMat();
    cv::Mat res = ptr->predict(retval);
    return TensorWrapper(res);
}

extern "C" struct TensorWrapper KalmanFilter_correct(struct KalmanFilterPtr ptr, struct TensorWrapper measurement)
{
    cv::Mat res = ptr->correct(measurement.toMat());
    return TensorWrapper(res);
}

extern "C" struct TensorWrapper DenseOpticalFlow_calc(struct DenseOpticalFlowPtr ptr, struct TensorWrapper I0,
                        struct TensorWrapper I1, struct TensorWrapper flow)
{
    if (flow.isNull()) {
        cv::Mat retval;
        ptr->calc(I0.toMat(), I1.toMat(), retval);
        return TensorWrapper(retval);
    }
    ptr->calc(I0.toMat(), I1.toMat(), flow.toMat());
    return flow;
}

extern "C" void DenseOpticalFlow_collectGarbage(struct DenseOpticalFlowPtr ptr)
{
    ptr->collectGarbage();
}

// DualTVL1OpticalFlow

extern "C" struct DualTVL1OpticalFlowPtr DualTVL1OpticalFlow_ctor()
{
    return rescueObjectFromPtr(cv::createOptFlow_DualTVL1());
}

extern "C" void DualTVL1OpticalFlow_setTau(struct DualTVL1OpticalFlowPtr ptr, double val)
{
    ptr->setTau(val);
}

extern "C" double DualTVL1OpticalFlow_getTau(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getTau();
}

extern "C" void DualTVL1OpticalFlow_setLambda(struct DualTVL1OpticalFlowPtr ptr, double val)
{
    ptr->setLambda(val);
}

extern "C" double DualTVL1OpticalFlow_getLambda(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getLambda();
}

extern "C" void DualTVL1OpticalFlow_setTheta(struct DualTVL1OpticalFlowPtr ptr, double val)
{
    ptr->setTheta(val);
}

extern "C" double DualTVL1OpticalFlow_getTheta(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getTheta();
}

extern "C" void DualTVL1OpticalFlow_setGamma(struct DualTVL1OpticalFlowPtr ptr, double val)
{
    ptr->setGamma(val);
}

extern "C" double DualTVL1OpticalFlow_getGamma(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getGamma();
}

extern "C" void DualTVL1OpticalFlow_setEpsilon(struct DualTVL1OpticalFlowPtr ptr, double val)
{
    ptr->setEpsilon(val);
}

extern "C" double DualTVL1OpticalFlow_getEpsilon(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getEpsilon();
}

extern "C" void DualTVL1OpticalFlow_setScaleStep(struct DualTVL1OpticalFlowPtr ptr, double val)
{
    ptr->setScaleStep(val);
}

extern "C" double DualTVL1OpticalFlow_getScaleStep(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getScaleStep();
}

extern "C" void DualTVL1OpticalFlow_setScalesNumber(struct DualTVL1OpticalFlowPtr ptr, int val)
{
    ptr->setScalesNumber(val);
}

extern "C" int DualTVL1OpticalFlow_getScalesNumber(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getScalesNumber();
}

extern "C" void DualTVL1OpticalFlow_setWarpingsNumber(struct DualTVL1OpticalFlowPtr ptr, int val)
{
    ptr->setWarpingsNumber(val);
}

extern "C" int DualTVL1OpticalFlow_getWarpingsNumber(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getWarpingsNumber();
}

extern "C" void DualTVL1OpticalFlow_setInnerIterations(struct DualTVL1OpticalFlowPtr ptr, int val)
{
    ptr->setInnerIterations(val);
}

extern "C" int DualTVL1OpticalFlow_getInnerIterations(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getInnerIterations();
}

extern "C" void DualTVL1OpticalFlow_setOuterIterations(struct DualTVL1OpticalFlowPtr ptr, int val)
{
    ptr->setOuterIterations(val);
}

extern "C" int DualTVL1OpticalFlow_getOuterIterations(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getOuterIterations();
}

extern "C" void DualTVL1OpticalFlow_setMedianFiltering(struct DualTVL1OpticalFlowPtr ptr, int val)
{
    ptr->setMedianFiltering(val);
}

extern "C" int DualTVL1OpticalFlow_getMedianFiltering(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getMedianFiltering();
}

extern "C" void DualTVL1OpticalFlow_setUseInitialFlow(struct DualTVL1OpticalFlowPtr ptr, bool val)
{
    ptr->setUseInitialFlow(val);
}

extern "C" bool DualTVL1OpticalFlow_getUseInitialFlow(struct DualTVL1OpticalFlowPtr ptr)
{
    return ptr->getUseInitialFlow();
}

// BackgroundSubtractorMOG

extern "C" struct BackgroundSubtractorMOGPtr BackgroundSubtractorMOG_ctor(int history, int nmixtures,
                        double backgroundRatio, double noiseSigma)
{
    return rescueObjectFromPtr(createBackgroundSubtractorMOG(history, nmixtures, backgroundRatio, noiseSigma));
}

extern "C" void BackgroundSubtractorMOG_setHistory(struct BackgroundSubtractorMOGPtr ptr, int val)
{
    ptr->setHistory(val);
}

extern "C" int BackgroundSubtractorMOG_getHistory(struct BackgroundSubtractorMOGPtr ptr)
{
    return ptr->getHistory();
}

extern "C" void BackgroundSubtractorMOG_setNMixtures(struct BackgroundSubtractorMOGPtr ptr, int val)
{
    ptr->setNMixtures(val);
}

extern "C" int BackgroundSubtractorMOG_getNMixtures(struct BackgroundSubtractorMOGPtr ptr)
{
    return ptr->getNMixtures();
}

extern "C" void BackgroundSubtractorMOG_setBackgroundRatio(struct BackgroundSubtractorMOGPtr ptr, double backgroundRatio)
{
    ptr->setBackgroundRatio(backgroundRatio);
}

extern "C" double BackgroundSubtractorMOG_getBackgroundRatio(struct BackgroundSubtractorMOGPtr ptr)
{
    return ptr->getBackgroundRatio();
}

extern "C" void BackgroundSubtractorMOG_setNoiseSigma(struct BackgroundSubtractorMOGPtr ptr, double noiseSigma)
{
    ptr->setNoiseSigma(noiseSigma);
}

extern "C" double BackgroundSubtractorMOG_getNoiseSigma(struct BackgroundSubtractorMOGPtr ptr)
{
    return ptr->getNoiseSigma();
}

// BackgroundSubtractorGMG

extern "C" struct BackgroundSubtractorGMGPtr BackgroundSubtractorGMG_ctor(int initializationFrames, double decisionThreshold)
{
    return rescueObjectFromPtr(createBackgroundSubtractorGMG(initializationFrames, decisionThreshold));
}

extern "C" void BackgroundSubtractorGMG_setMaxFeatures(struct BackgroundSubtractorGMGPtr ptr, int maxFeatures)
{
    ptr->setMaxFeatures(maxFeatures);
}

extern "C" int BackgroundSubtractorGMG_getMaxFeatures(struct BackgroundSubtractorGMGPtr ptr)
{
    return ptr->getMaxFeatures();
}

extern "C" void BackgroundSubtractorGMG_setNumFrames(struct BackgroundSubtractorGMGPtr ptr, int numFrames)
{
    ptr->setNumFrames(numFrames);
}

extern "C" int BackgroundSubtractorGMG_getNumFrames(struct BackgroundSubtractorGMGPtr ptr)
{
    return ptr->getNumFrames();
}

extern "C" void BackgroundSubtractorGMG_setQuantizationLevels(struct BackgroundSubtractorGMGPtr ptr, int quantizationLevels)
{
    ptr->setQuantizationLevels(quantizationLevels);
}

extern "C" int BackgroundSubtractorGMG_getQuantizationLevels(struct BackgroundSubtractorGMGPtr ptr)
{
    return ptr->getQuantizationLevels();
}

extern "C" void BackgroundSubtractorGMG_setSmoothingRadius(struct BackgroundSubtractorGMGPtr ptr, int smoothingRadius)
{
    ptr->setSmoothingRadius(smoothingRadius);
}

extern "C" int BackgroundSubtractorGMG_getSmoothingRadius(struct BackgroundSubtractorGMGPtr ptr)
{
    return ptr->getSmoothingRadius();
}

extern "C" void BackgroundSubtractorGMG_setDefaultLearningRate(struct BackgroundSubtractorGMGPtr ptr, double defaultLearningRate)
{
    ptr->setDefaultLearningRate(defaultLearningRate);
}

extern "C" double BackgroundSubtractorGMG_getDefaultLearningRate(struct BackgroundSubtractorGMGPtr ptr)
{
    return ptr->getDefaultLearningRate();
}

extern "C" void BackgroundSubtractorGMG_setBackgroundPrior(struct BackgroundSubtractorGMGPtr ptr, double backgroundPrior)
{
    ptr->setBackgroundPrior(backgroundPrior);
}

extern "C" double BackgroundSubtractorGMG_getBackgroundPrior(struct BackgroundSubtractorGMGPtr ptr)
{
    return ptr->getBackgroundPrior();
}

extern "C" void BackgroundSubtractorGMG_setDecisionThreshold(struct BackgroundSubtractorGMGPtr ptr, double decisionThreshold)
{
    ptr->setDecisionThreshold(decisionThreshold);
}

extern "C" double BackgroundSubtractorGMG_getDecisionThreshold(struct BackgroundSubtractorGMGPtr ptr)
{
    return ptr->getDecisionThreshold();
}

extern "C" void BackgroundSubtractorGMG_setMinVal(struct BackgroundSubtractorGMGPtr ptr, double minVal)
{
    ptr->setMinVal(minVal);
}

extern "C" double BackgroundSubtractorGMG_getMinVal(struct BackgroundSubtractorGMGPtr ptr)
{
    return ptr->getMinVal();
}

extern "C" void BackgroundSubtractorGMG_setMaxVal(struct BackgroundSubtractorGMGPtr ptr, double maxVal)
{
    ptr->setMaxVal(maxVal);
}

extern "C" double BackgroundSubtractorGMG_getMaxVal(struct BackgroundSubtractorGMGPtr ptr)
{
    return ptr->getMaxVal();
}

extern "C" void BackgroundSubtractorGMG_setUpdateBackgroundModel(struct BackgroundSubtractorGMGPtr ptr, bool updateBackgroundModel)
{
    ptr->setUpdateBackgroundModel(updateBackgroundModel);
}

extern "C" bool BackgroundSubtractorGMG_getUpdateBackgroundModel(struct BackgroundSubtractorGMGPtr ptr)
{
    return ptr->getUpdateBackgroundModel();
}