#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/video.hpp>

extern "C" struct RotatedRectWrapper CamShift(struct TensorWrapper probImage, struct RectWrapper window,
                        struct TermCriteriaWrapper criteria);

extern "C" int meanShift(struct TensorWrapper probImage, struct RectWrapper window,
                        struct TermCriteriaWrapper criteria);

extern "C" struct TensorArray buildOpticalFlowPyramid(struct TensorWrapper img, struct TensorArray pyramid,
                        struct SizeWrapper winSize, int maxLevel, bool withDerivatives, int pyrBorder,
                        int derivBorder, bool tryReuseInputImage);

extern "C" struct TensorWrapper calcOpticalFlowPyrLK(struct TensorWrapper prevImg,
                        struct TensorWrapper nextImg, struct TensorWrapper prevPts,
                        struct TensorWrapper nextPts, struct TensorWrapper status,
                        struct TensorWrapper err, struct SizeWrapper winSize, int maxLevel,
                        struct TermCriteriaWrapper criteria, int flags, double minEigThreshold);

extern "C" struct TensorWrapper calcOpticalFlowFarneback(struct TensorWrapper prev, struct TensorWrapper next,
                        struct TensorWrapper flow, double pyr_scale, int levels, int winsize,
                        int iterations, int poly_n, double poly_sigma, int flags);

extern "C" struct TensorWrapper estimateRigidTransform(struct TensorWrapper src, struct TensorWrapper dst, bool fullAffine);

extern "C" double findTransformECC(struct TensorWrapper templateImage, struct TensorWrapper inputImage,
                        struct TensorWrapper warpMatrix, int motionType, struct TermCriteriaWrapper criteria,
                        struct TensorWrapper inputMask);

// BackgroundSubtractor

struct BackgroundSubtractorPtr {
    void *ptr;
    inline cv::BackgroundSubtractor * operator->() { return static_cast<cv::BackgroundSubtractor *>(ptr); }
    inline BackgroundSubtractorPtr(cv::BackgroundSubtractor *ptr) { this->ptr = ptr; }
};

extern "C" struct TensorWrapper BackgroundSubtractor_apply(struct BackgroundSubtractorPtr ptr, struct TensorWrapper image,
                        struct TensorWrapper fgmast, double learningRate);

extern "C" struct TensorWrapper BackgroundSubtractor_getBackgroundImage(struct BackgroundSubtractorPtr ptr,
                        struct TensorWrapper backgroundImage);

// BackgroundSubtractorMOG2

struct BackgroundSubtractorMOG2Ptr {
    void *ptr;
    inline cv::BackgroundSubtractorMOG2 * operator->() { return static_cast<cv::BackgroundSubtractorMOG2 *>(ptr); }
    inline BackgroundSubtractorMOG2Ptr(cv::BackgroundSubtractorMOG2 *ptr) { this->ptr = ptr; }
};

extern "C" struct BackgroundSubtractorMOG2Ptr BackgroundSubtractorMOG2_ctor(int history, double varThreshold, bool detectShadows);

extern "C" int BackgroundSubtractorMOG2_getHistory(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C" void BackgroundSubtractorMOG2_setHistory(struct BackgroundSubtractorMOG2Ptr ptr, int history);

extern "C" int BackgroundSubtractorMOG2_getNMixtures(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C" void BackgroundSubtractorMOG2_setNMixtures(struct BackgroundSubtractorMOG2Ptr ptr, int nmixtures);

extern "C" int BackgroundSubtractorMOG2_getShadowValue(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C" void BackgroundSubtractorMOG2_setShadowValue(struct BackgroundSubtractorMOG2Ptr ptr, int shadow_value);

extern "C" double BackgroundSubtractorMOG2_getBackgroundRatio(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C" void BackgroundSubtractorMOG2_setBackgroundRatio(struct BackgroundSubtractorMOG2Ptr ptr, double ratio);

extern "C" double BackgroundSubtractorMOG2_getVarThreshold(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C" void BackgroundSubtractorMOG2_setVarThreshold(struct BackgroundSubtractorMOG2Ptr ptr, double varThreshold);

extern "C" double BackgroundSubtractorMOG2_getVarThresholdGen(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C" void BackgroundSubtractorMOG2_setVarThresholdGen(struct BackgroundSubtractorMOG2Ptr ptr, double varThresholdGen);

extern "C" double BackgroundSubtractorMOG2_getVarInit(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C" void BackgroundSubtractorMOG2_setVarInit(struct BackgroundSubtractorMOG2Ptr ptr, double varInit);

extern "C" double BackgroundSubtractorMOG2_getVarMin(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C" void BackgroundSubtractorMOG2_setVarMin(struct BackgroundSubtractorMOG2Ptr ptr, double varMin);

extern "C" double BackgroundSubtractorMOG2_getVarMax(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C" void BackgroundSubtractorMOG2_setVarMax(struct BackgroundSubtractorMOG2Ptr ptr, double varMax);

extern "C" bool BackgroundSubtractorMOG2_getDetectShadows(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C" void BackgroundSubtractorMOG2_setDetectShadows(struct BackgroundSubtractorMOG2Ptr ptr, bool detectShadows);

extern "C" double BackgroundSubtractorMOG2_getComplexityReductionThreshold(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C" void BackgroundSubtractorMOG2_setComplexityReductionThreshold(struct BackgroundSubtractorMOG2Ptr ptr, double ct);

extern "C" double BackgroundSubtractorMOG2_getShadowThreshold(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C" void BackgroundSubtractorMOG2_setShadowThreshold(struct BackgroundSubtractorMOG2Ptr ptr, double shadowThreshold);

// BackgroundSubtractorKNN

struct BackgroundSubtractorKNNPtr {
    void *ptr;
    inline cv::BackgroundSubtractorKNN * operator->() { return static_cast<cv::BackgroundSubtractorKNN *>(ptr); }
    inline BackgroundSubtractorKNNPtr(cv::BackgroundSubtractorKNN *ptr) { this->ptr = ptr; }
};

extern "C" struct BackgroundSubtractorKNNPtr BackgroundSubtractorKNN_ctor(int history, double dist2Threshold, bool detectShadows);

extern "C" int BackgroundSubtractorKNN_getHistory(struct BackgroundSubtractorKNNPtr ptr);

extern "C" void BackgroundSubtractorKNN_setHistory(struct BackgroundSubtractorKNNPtr ptr, int history);

extern "C" int BackgroundSubtractorKNN_getNSamples(struct BackgroundSubtractorKNNPtr ptr);

extern "C" void BackgroundSubtractorKNN_setNSamples(struct BackgroundSubtractorKNNPtr ptr, int nSamples);

extern "C" int BackgroundSubtractorKNN_getkNNSamples(struct BackgroundSubtractorKNNPtr ptr);

extern "C" void BackgroundSubtractorKNN_setkNNSamples(struct BackgroundSubtractorKNNPtr ptr, int kNNSamples);

extern "C" int BackgroundSubtractorKNN_getShadowValue(struct BackgroundSubtractorKNNPtr ptr);

extern "C" void BackgroundSubtractorKNN_setShadowValue(struct BackgroundSubtractorKNNPtr ptr, int shadowValue);

extern "C" double BackgroundSubtractorKNN_getDist2Threshold(struct BackgroundSubtractorKNNPtr ptr);

extern "C" void BackgroundSubtractorKNN_setDist2Threshold(struct BackgroundSubtractorKNNPtr ptr, double dist2Threshold);

extern "C" double BackgroundSubtractorKNN_getShadowThreshold(struct BackgroundSubtractorKNNPtr ptr);

extern "C" void BackgroundSubtractorKNN_setShadowThreshold(struct BackgroundSubtractorKNNPtr ptr, double shadowThreshold);

extern "C" bool BackgroundSubtractorKNN_getDetectShadows(struct BackgroundSubtractorKNNPtr ptr);

extern "C" void BackgroundSubtractorKNN_setDetectShadows(struct BackgroundSubtractorKNNPtr ptr, bool detectShadows);

// KalmanFilter

struct KalmanFilterPtr {
    void *ptr;
    inline cv::KalmanFilter * operator->() { return static_cast<cv::KalmanFilter *>(ptr); }
    inline KalmanFilterPtr(cv::KalmanFilter *ptr) { this->ptr = ptr; }
    inline cv::KalmanFilter & operator*() { return *static_cast<cv::KalmanFilter *>(this->ptr); }
};

extern "C" struct KalmanFilterPtr KalmanFilter_ctor_default();

extern "C" struct KalmanFilterPtr KalmanFilter_ctor(int dynamParams, int measureParams, int controlParams, int type);

extern "C" void KalmanFilter_dtor(struct KalmanFilterPtr ptr);

extern "C" void KalmanFilter_init(struct KalmanFilterPtr ptr, int dynamParams, int measureParams, int controlParams, int type);

extern "C" struct TensorWrapper KalmanFilter_predict(struct KalmanFilterPtr ptr, struct TensorWrapper control);

extern "C" struct TensorWrapper KalmanFilter_correct(struct KalmanFilterPtr ptr, struct TensorWrapper measurement);

// DenseOpticalFlow

struct DenseOpticalFlowPtr {
    void *ptr;
    inline cv::DenseOpticalFlow * operator->() { return static_cast<cv::DenseOpticalFlow *>(ptr); }
    inline DenseOpticalFlowPtr(cv::DenseOpticalFlow *ptr) { this->ptr = ptr; }
    inline cv::DenseOpticalFlow & operator*() { return *static_cast<cv::DenseOpticalFlow *>(this->ptr); }
};

extern "C" struct TensorWrapper DenseOpticalFlow_calc(struct DenseOpticalFlowPtr ptr, struct TensorWrapper I0,
                        struct TensorWrapper I1, struct TensorWrapper flow);

extern "C" void DenseOpticalFlow_collectGarbage(struct DenseOpticalFlowPtr ptr);

// DualTVL1OpticalFlow

struct DualTVL1OpticalFlowPtr {
    void *ptr;
    inline cv::DualTVL1OpticalFlow * operator->() { return static_cast<cv::DualTVL1OpticalFlow *>(ptr); }
    inline DualTVL1OpticalFlowPtr(cv::DualTVL1OpticalFlow *ptr) { this->ptr = ptr; }
    inline cv::DualTVL1OpticalFlow & operator*() { return *static_cast<cv::DualTVL1OpticalFlow *>(this->ptr); }
};

extern "C" struct DualTVL1OpticalFlowPtr DualTVL1OpticalFlow_ctor();

extern "C" void DualTVL1OpticalFlow_setTau(struct DualTVL1OpticalFlowPtr ptr, double val);

extern "C" double DualTVL1OpticalFlow_getTau(struct DualTVL1OpticalFlowPtr ptr);

extern "C" void DualTVL1OpticalFlow_setLambda(struct DualTVL1OpticalFlowPtr ptr, double val);

extern "C" double DualTVL1OpticalFlow_getLambda(struct DualTVL1OpticalFlowPtr ptr);

extern "C" void DualTVL1OpticalFlow_setTheta(struct DualTVL1OpticalFlowPtr ptr, double val);

extern "C" double DualTVL1OpticalFlow_getTheta(struct DualTVL1OpticalFlowPtr ptr);

extern "C" void DualTVL1OpticalFlow_setGamma(struct DualTVL1OpticalFlowPtr ptr, double val);

extern "C" double DualTVL1OpticalFlow_getGamma(struct DualTVL1OpticalFlowPtr ptr);

extern "C" void DualTVL1OpticalFlow_setEpsilon(struct DualTVL1OpticalFlowPtr ptr, double val);

extern "C" double DualTVL1OpticalFlow_getEpsilon(struct DualTVL1OpticalFlowPtr ptr);

extern "C" void DualTVL1OpticalFlow_setScaleStep(struct DualTVL1OpticalFlowPtr ptr, double val);

extern "C" double DualTVL1OpticalFlow_getScaleStep(struct DualTVL1OpticalFlowPtr ptr);

extern "C" void DualTVL1OpticalFlow_setScalesNumber(struct DualTVL1OpticalFlowPtr ptr, int val);

extern "C" int DualTVL1OpticalFlow_getScalesNumber(struct DualTVL1OpticalFlowPtr ptr);

extern "C" void DualTVL1OpticalFlow_setWarpingsNumber(struct DualTVL1OpticalFlowPtr ptr, int val);

extern "C" int DualTVL1OpticalFlow_getWarpingsNumber(struct DualTVL1OpticalFlowPtr ptr);

extern "C" void DualTVL1OpticalFlow_setInnerIterations(struct DualTVL1OpticalFlowPtr ptr, int val);

extern "C" int DualTVL1OpticalFlow_getInnerIterations(struct DualTVL1OpticalFlowPtr ptr);

extern "C" void DualTVL1OpticalFlow_setOuterIterations(struct DualTVL1OpticalFlowPtr ptr, int val);

extern "C" int DualTVL1OpticalFlow_getOuterIterations(struct DualTVL1OpticalFlowPtr ptr);

extern "C" void DualTVL1OpticalFlow_setMedianFiltering(struct DualTVL1OpticalFlowPtr ptr, int val);

extern "C" int DualTVL1OpticalFlow_getMedianFiltering(struct DualTVL1OpticalFlowPtr ptr);

extern "C" void DualTVL1OpticalFlow_setUseInitialFlow(struct DualTVL1OpticalFlowPtr ptr, bool val);

extern "C" bool DualTVL1OpticalFlow_getUseInitialFlow(struct DualTVL1OpticalFlowPtr ptr);