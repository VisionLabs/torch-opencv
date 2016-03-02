#include <cudaoptflow.hpp>

struct TensorWrapper DenseOpticalFlow_calc(struct cutorchInfo info,
    struct DenseOpticalFlowPtr ptr, struct TensorWrapper I0, struct TensorWrapper I1,
    struct TensorWrapper flow)
{
    cuda::GpuMat retval;
    if (!flow.isNull()) retval = flow.toGpuMat();
    ptr->calc(I0.toGpuMat(), I1.toGpuMat(), retval, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

struct TensorArray SparseOpticalFlow_calc(struct cutorchInfo info,
    struct SparseOpticalFlowPtr ptr, struct TensorWrapper prevImg, struct TensorWrapper nextImg,
    struct TensorWrapper prevPts, struct TensorWrapper nextPts, struct TensorWrapper status,
    bool outputErr, struct TensorWrapper err)
{
    std::vector<cuda::GpuMat> retval(3);
    if (!nextPts.isNull()) retval[0] = nextPts.toGpuMat();
    if (!status.isNull())  retval[1] = status.toGpuMat();
    if (!err.isNull())     retval[2] = err.toGpuMat();

    ptr->calc(prevImg.toGpuMat(), nextImg.toGpuMat(), prevPts.toGpuMat(),
            retval[0], retval[1], outputErr ? retval[2] : cv::noArray(), prepareStream(info));

    return TensorArray(retval, info.state);
}

extern "C"
struct BroxOpticalFlowPtr BroxOpticalFlow_ctorCuda(
        double alpha, double gamma, double scale_factor, int inner_iterations,
        int outer_iterations, int solver_iterations)
{
    return rescueObjectFromPtr(cuda::BroxOpticalFlow::create(
            alpha, gamma, scale_factor, inner_iterations, outer_iterations, solver_iterations));
}

extern "C"
void BroxOpticalFlow_setFlowSmoothnessCuda(struct BroxOpticalFlowPtr ptr, double val)
{
    ptr->setFlowSmoothness(val);
}

extern "C"
double BroxOpticalFlow_getFlowSmoothnessCuda(struct BroxOpticalFlowPtr ptr)
{
    return ptr->getFlowSmoothness();
}

extern "C"
void BroxOpticalFlow_setGradientConstancyImportanceCuda(struct BroxOpticalFlowPtr ptr, double val)
{
    ptr->setGradientConstancyImportance(val);
}

extern "C"
double BroxOpticalFlow_getGradientConstancyImportanceCuda(struct BroxOpticalFlowPtr ptr)
{
    return ptr->getGradientConstancyImportance();
}

extern "C"
void BroxOpticalFlow_setPyramidScaleFactorCuda(struct BroxOpticalFlowPtr ptr, double val)
{
    ptr->setPyramidScaleFactor(val);
}

extern "C"
double BroxOpticalFlow_getPyramidScaleFactorCuda(struct BroxOpticalFlowPtr ptr)
{
    return ptr->getPyramidScaleFactor();
}

extern "C"
void BroxOpticalFlow_setInnerIterationsCuda(struct BroxOpticalFlowPtr ptr, int val)
{
    ptr->setInnerIterations(val);
}

extern "C"
int BroxOpticalFlow_getInnerIterationsCuda(struct BroxOpticalFlowPtr ptr)
{
    return ptr->getInnerIterations();
}

extern "C"
void BroxOpticalFlow_setOuterIterationsCuda(struct BroxOpticalFlowPtr ptr, int val)
{
    ptr->setOuterIterations(val);
}

extern "C"
int BroxOpticalFlow_getOuterIterationsCuda(struct BroxOpticalFlowPtr ptr)
{
    return ptr->getOuterIterations();
}

extern "C"
void BroxOpticalFlow_setSolverIterationsCuda(struct BroxOpticalFlowPtr ptr, int val)
{
    ptr->setSolverIterations(val);
}

extern "C"
int BroxOpticalFlow_getSolverIterationsCuda(struct BroxOpticalFlowPtr ptr)
{
    return ptr->getSolverIterations();
}

extern "C"
struct SparsePyrLKOpticalFlowPtr SparsePyrLKOpticalFlow_ctorCuda(
        struct SizeWrapper winSize, int maxLevel, int iters, bool useInitialFlow)
{
    return rescueObjectFromPtr(cuda::SparsePyrLKOpticalFlow::create(
            winSize, maxLevel, iters, useInitialFlow));
}

extern "C"
void SparsePyrLKOpticalFlow_setWinSizeCuda(struct SparsePyrLKOpticalFlowPtr ptr, struct SizeWrapper val)
{
    ptr->setWinSize(val);
}

extern "C"
struct SizeWrapper SparsePyrLKOpticalFlow_getWinSizeCuda(struct SparsePyrLKOpticalFlowPtr ptr)
{
    return ptr->getWinSize();
}

extern "C"
void SparsePyrLKOpticalFlow_setMaxLevelCuda(struct SparsePyrLKOpticalFlowPtr ptr, int val)
{
    ptr->setMaxLevel(val);
}

extern "C"
int SparsePyrLKOpticalFlow_getMaxLevelCuda(struct SparsePyrLKOpticalFlowPtr ptr)
{
    return ptr->getMaxLevel();
}

extern "C"
void SparsePyrLKOpticalFlow_setNumItersCuda(struct SparsePyrLKOpticalFlowPtr ptr, int val)
{
    ptr->setNumIters(val);
}

extern "C"
int SparsePyrLKOpticalFlow_getNumItersCuda(struct SparsePyrLKOpticalFlowPtr ptr)
{
    return ptr->getNumIters();
}

extern "C"
void SparsePyrLKOpticalFlow_setUseInitialFlowCuda(struct SparsePyrLKOpticalFlowPtr ptr, bool val)
{
    ptr->setUseInitialFlow(val);
}

extern "C"
bool SparsePyrLKOpticalFlow_getUseInitialFlowCuda(struct SparsePyrLKOpticalFlowPtr ptr)
{
    return ptr->getUseInitialFlow();
}

extern "C"
struct DensePyrLKOpticalFlowPtr DensePyrLKOpticalFlow_ctorCuda(
        struct SizeWrapper winSize, int maxLevel, int iters, bool useInitialFlow)
{
    return rescueObjectFromPtr(cuda::DensePyrLKOpticalFlow::create(
            winSize, maxLevel, iters, useInitialFlow));
}

extern "C"
void DensePyrLKOpticalFlow_setWinSizeCuda(struct DensePyrLKOpticalFlowPtr ptr, struct SizeWrapper val)
{
    ptr->setWinSize(val);
}

extern "C"
struct SizeWrapper DensePyrLKOpticalFlow_getWinSizeCuda(struct DensePyrLKOpticalFlowPtr ptr)
{
    return ptr->getWinSize();
}

extern "C"
void DensePyrLKOpticalFlow_setMaxLevelCuda(struct DensePyrLKOpticalFlowPtr ptr, int val)
{
    ptr->setMaxLevel(val);
}

extern "C"
int DensePyrLKOpticalFlow_getMaxLevelCuda(struct DensePyrLKOpticalFlowPtr ptr)
{
    return ptr->getMaxLevel();
}

extern "C"
void DensePyrLKOpticalFlow_setNumItersCuda(struct DensePyrLKOpticalFlowPtr ptr, int val)
{
    ptr->setNumIters(val);
}

extern "C"
int DensePyrLKOpticalFlow_getNumItersCuda(struct DensePyrLKOpticalFlowPtr ptr)
{
    return ptr->getNumIters();
}

extern "C"
void DensePyrLKOpticalFlow_setUseInitialFlowCuda(struct DensePyrLKOpticalFlowPtr ptr, bool val)
{
    ptr->setUseInitialFlow(val);
}

extern "C"
bool DensePyrLKOpticalFlow_getUseInitialFlowCuda(struct DensePyrLKOpticalFlowPtr ptr)
{
    return ptr->getUseInitialFlow();
}

extern "C"
struct FarnebackOpticalFlowPtr FarnebackOpticalFlow_ctorCuda(
        int NumLevels, double PyrScale, bool FastPyramids, int WinSize,
        int NumIters, int PolyN, double PolySigma, int Flags)
{
    return rescueObjectFromPtr(cuda::FarnebackOpticalFlow::create(
            NumLevels, PyrScale, FastPyramids, WinSize, NumIters,
            PolyN, PolySigma, Flags));
}

extern "C"
void FarnebackOpticalFlow_setNumLevelsCuda(struct FarnebackOpticalFlowPtr ptr, int val)
{
    ptr->setNumLevels(val);
}

extern "C"
int FarnebackOpticalFlow_getNumLevelsCuda(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getNumLevels();
}

extern "C"
void FarnebackOpticalFlow_setPyrScaleCuda(struct FarnebackOpticalFlowPtr ptr, double val)
{
    ptr->setPyrScale(val);
}

extern "C"
double FarnebackOpticalFlow_getPyrScaleCuda(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getPyrScale();
}

extern "C"
void FarnebackOpticalFlow_setFastPyramidsCuda(struct FarnebackOpticalFlowPtr ptr, bool val)
{
    ptr->setFastPyramids(val);
}

extern "C"
bool FarnebackOpticalFlow_getFastPyramidsCuda(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getFastPyramids();
}

extern "C"
void FarnebackOpticalFlow_setWinSizeCuda(struct FarnebackOpticalFlowPtr ptr, int val)
{
    ptr->setWinSize(val);
}

extern "C"
int FarnebackOpticalFlow_getWinSizeCuda(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getWinSize();
}

extern "C"
void FarnebackOpticalFlow_setNumItersCuda(struct FarnebackOpticalFlowPtr ptr, int val)
{
    ptr->setNumIters(val);
}

extern "C"
int FarnebackOpticalFlow_getNumItersCuda(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getNumIters();
}

extern "C"
void FarnebackOpticalFlow_setPolyNCuda(struct FarnebackOpticalFlowPtr ptr, int val)
{
    ptr->setPolyN(val);
}

extern "C"
int FarnebackOpticalFlow_getPolyNCuda(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getPolyN();
}

extern "C"
void FarnebackOpticalFlow_setPolySigmaCuda(struct FarnebackOpticalFlowPtr ptr, double val)
{
    ptr->setPolySigma(val);
}

extern "C"
double FarnebackOpticalFlow_getPolySigmaCuda(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getPolySigma();
}

extern "C"
void FarnebackOpticalFlow_setFlagsCuda(struct FarnebackOpticalFlowPtr ptr, int val)
{
    ptr->setFlags(val);
}

extern "C"
int FarnebackOpticalFlow_getFlagsCuda(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getFlags();
}

extern "C"
struct OpticalFlowDual_TVL1Ptr OpticalFlowDual_TVL1_ctorCuda(
        double tau, double lambda, double theta, int nscales, int warps, double epsilon, 
        int iterations, double scaleStep, double gamma, bool useInitialFlow)
{
    return rescueObjectFromPtr(cuda::OpticalFlowDual_TVL1::create(
            tau, lambda, theta, nscales, warps, epsilon,
            iterations, scaleStep, gamma, useInitialFlow));
}

extern "C"
void OpticalFlowDual_TVL1_setTauCuda(struct OpticalFlowDual_TVL1Ptr ptr, double val)
{
    ptr->setTau(val);
}

extern "C"
double OpticalFlowDual_TVL1_getTauCuda(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getTau();
}

extern "C"
void OpticalFlowDual_TVL1_setLambdaCuda(struct OpticalFlowDual_TVL1Ptr ptr, double val)
{
    ptr->setLambda(val);
}

extern "C"
double OpticalFlowDual_TVL1_getLambdaCuda(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getLambda();
}

extern "C"
void OpticalFlowDual_TVL1_setGammaCuda(struct OpticalFlowDual_TVL1Ptr ptr, double val)
{
    ptr->setGamma(val);
}

extern "C"
double OpticalFlowDual_TVL1_getGammaCuda(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getGamma();
}

extern "C"
void OpticalFlowDual_TVL1_setThetaCuda(struct OpticalFlowDual_TVL1Ptr ptr, double val)
{
    ptr->setTheta(val);
}

extern "C"
double OpticalFlowDual_TVL1_getThetaCuda(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getTheta();
}

extern "C"
void OpticalFlowDual_TVL1_setNumScalesCuda(struct OpticalFlowDual_TVL1Ptr ptr, int val)
{
    ptr->setNumScales(val);
}

extern "C"
int OpticalFlowDual_TVL1_getNumScalesCuda(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getNumScales();
}

extern "C"
void OpticalFlowDual_TVL1_setNumWarpsCuda(struct OpticalFlowDual_TVL1Ptr ptr, int val)
{
    ptr->setNumWarps(val);
}

extern "C"
int OpticalFlowDual_TVL1_getNumWarpsCuda(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getNumWarps();
}

extern "C"
void OpticalFlowDual_TVL1_setEpsilonCuda(struct OpticalFlowDual_TVL1Ptr ptr, double val)
{
    ptr->setEpsilon(val);
}

extern "C"
double OpticalFlowDual_TVL1_getEpsilonCuda(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getEpsilon();
}

extern "C"
void OpticalFlowDual_TVL1_setNumIterationsCuda(struct OpticalFlowDual_TVL1Ptr ptr, int val)
{
    ptr->setNumIterations(val);
}

extern "C"
int OpticalFlowDual_TVL1_getNumIterationsCuda(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getNumIterations();
}

extern "C"
void OpticalFlowDual_TVL1_setScaleStepCuda(struct OpticalFlowDual_TVL1Ptr ptr, double val)
{
    ptr->setScaleStep(val);
}

extern "C"
double OpticalFlowDual_TVL1_getScaleStepCuda(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getScaleStep();
}

extern "C"
void OpticalFlowDual_TVL1_setUseInitialFlowCuda(struct OpticalFlowDual_TVL1Ptr ptr, bool val)
{
    ptr->setUseInitialFlow(val);
}

extern "C"
bool OpticalFlowDual_TVL1_getUseInitialFlowCuda(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getUseInitialFlow();
}

