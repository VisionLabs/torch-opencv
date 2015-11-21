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

// A hack for accessing private fields
// To be removed when https://github.com/VisionLabs/torch-opencv/issues/29 is closed

class CV_EXPORTS SRes : public cv::Algorithm, public superres::FrameSource
{
public:
    /** @brief Set input frame source for Super Resolution algorithm.

    @param frameSource Input frame source
     */
    void setInput(const cv::Ptr<superres::FrameSource>& frameSource);

    /** @brief Process next frame from input and return output result.

    @param frame Output result
     */
    void nextFrame(cv::OutputArray frame);
    void reset();

    /** @brief Clear all inner buffers.
    */
    virtual void collectGarbage();

    //! @brief Scale factor
    /** @see setScale */
    virtual int getScale() const = 0;
    /** @copybrief getScale @see getScale */
    virtual void setScale(int val) = 0;

    //! @brief Iterations count
    /** @see setIterations */
    virtual int getIterations() const = 0;
    /** @copybrief getIterations @see getIterations */
    virtual void setIterations(int val) = 0;

    //! @brief Asymptotic value of steepest descent method
    /** @see setTau */
    virtual double getTau() const = 0;
    /** @copybrief getTau @see getTau */
    virtual void setTau(double val) = 0;

    //! @brief Weight parameter to balance data term and smoothness term
    /** @see setLabmda */
    virtual double getLabmda() const = 0;
    /** @copybrief getLabmda @see getLabmda */
    virtual void setLabmda(double val) = 0;

    //! @brief Parameter of spacial distribution in Bilateral-TV
    /** @see setAlpha */
    virtual double getAlpha() const = 0;
    /** @copybrief getAlpha @see getAlpha */
    virtual void setAlpha(double val) = 0;

    //! @brief Kernel size of Bilateral-TV filter
    /** @see setKernelSize */
    virtual int getKernelSize() const = 0;
    /** @copybrief getKernelSize @see getKernelSize */
    virtual void setKernelSize(int val) = 0;

    //! @brief Gaussian blur kernel size
    /** @see setBlurKernelSize */
    virtual int getBlurKernelSize() const = 0;
    /** @copybrief getBlurKernelSize @see getBlurKernelSize */
    virtual void setBlurKernelSize(int val) = 0;

    //! @brief Gaussian blur sigma
    /** @see setBlurSigma */
    virtual double getBlurSigma() const = 0;
    /** @copybrief getBlurSigma @see getBlurSigma */
    virtual void setBlurSigma(double val) = 0;

    //! @brief Radius of the temporal search area
    /** @see setTemporalAreaRadius */
    virtual int getTemporalAreaRadius() const = 0;
    /** @copybrief getTemporalAreaRadius @see getTemporalAreaRadius */
    virtual void setTemporalAreaRadius(int val) = 0;

    //! @brief Dense optical flow algorithm
    /** @see setOpticalFlow */
    virtual cv::Ptr<cv::superres::DenseOpticalFlowExt> getOpticalFlow() const = 0;
    /** @copybrief getOpticalFlow @see getOpticalFlow */
    virtual void setOpticalFlow(const cv::Ptr<cv::superres::DenseOpticalFlowExt> &val) = 0;

//protected:
    SRes();

    virtual void initImpl(cv::Ptr<superres::FrameSource>& frameSource) = 0;
    virtual void processImpl(cv::Ptr<superres::FrameSource>& frameSource, cv::OutputArray output) = 0;

    bool isUmat_;

//private:
    cv::Ptr<superres::FrameSource> frameSource_;
    bool firstCall_;
};

class BTVL1_B : public cv::superres::SuperResolution {
public:
    BTVL1_B();

    int scale_;
    int iterations_;
    double tau_;
    double lambda_;
    double alpha_;
    int btvKernelSize_;
    int blurKernelSize_;
    double blurSigma_;
    int temporalAreaRadius_; // not used in some implementations
    cv::Ptr <cv::superres::DenseOpticalFlowExt> opticalFlow_;

    bool ocl_process(cv::InputArrayOfArrays src, cv::OutputArray dst, cv::InputArrayOfArrays forwardMotions,
                     cv::InputArrayOfArrays backwardMotions, int baseIdx);

    //Ptr<FilterEngine> filter_;
    int curBlurKernelSize_;
    double curBlurSigma_;
    int curSrcType_;

    std::vector<float> btvWeights_;
    cv::UMat ubtvWeights_;

    int curBtvKernelSize_;
    double curAlpha_;

    // Mat
    std::vector<cv::Mat> lowResForwardMotions_;
    std::vector<cv::Mat> lowResBackwardMotions_;

    std::vector<cv::Mat> highResForwardMotions_;
    std::vector<cv::Mat> highResBackwardMotions_;

    std::vector<cv::Mat> forwardMaps_;
    std::vector<cv::Mat> backwardMaps_;

    cv::Mat highRes_;

    cv::Mat diffTerm_, regTerm_;
    cv::Mat a_, b_, c_;
};

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

extern "C"
void SuperResolution_setOpticalFlow(struct SuperResolutionPtr ptr, struct DenseOpticalFlowExtPtr val)
{
    cv::Ptr<superres::DenseOpticalFlowExt> tempPtr(
            static_cast<superres::DenseOpticalFlowExt *>(val.ptr));
    rescueObjectFromPtr(tempPtr);

    ptr->setOpticalFlow(tempPtr);

    std::cout << "Setting this optical flow: " << tempPtr.get() << std::endl;
    BTVL1_B * b = static_cast<BTVL1_B *>(ptr.ptr);
    std::cout << "BTVL1_Base haz " << b->opticalFlow_.get() << std::endl;
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
    std::vector<cv::Mat> retval(2);
    if (!flow1.isNull()) retval[0] = flow1;
    if (!flow2.isNull()) retval[1] = flow2;
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