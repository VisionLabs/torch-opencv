#include <cudaoptflow.hpp>

struct TensorWrapper DenseOpticalFlow_calc(struct THCState *state,
    struct DenseOpticalFlowPtr ptr, struct TensorWrapper I0, struct TensorWrapper I1,
    struct TensorWrapper flow)
{
    cuda::GpuMat retval;
    if (!flow.isNull()) retval = flow.toGpuMat();
    ptr->calc(I0.toGpuMat(), I1.toGpuMat(), retval);
    return TensorWrapper(retval, state);
}

struct TensorArray SparseOpticalFlow_calc(struct THCState *state,
    struct SparseOpticalFlowPtr ptr, struct TensorWrapper prevImg, struct TensorWrapper nextImg,
    struct TensorWrapper prevPts, struct TensorWrapper nextPts, struct TensorWrapper status,
    bool outputErr, struct TensorWrapper err)
{
    std::vector<cuda::GpuMat> retval(3);
    if (!nextPts.isNull()) retval[0] = nextPts.toGpuMat();
    if (!status.isNull())  retval[1] = status.toGpuMat();
    if (!err.isNull())     retval[2] = err.toGpuMat();

    ptr->calc(prevImg.toGpuMat(), nextImg.toGpuMat(), prevPts.toGpuMat(),
            retval[0], retval[1], outputErr ? retval[2] : cv::noArray());

    return TensorArray(retval, state);
}

extern "C"
struct BroxOpticalFlowPtr BroxOpticalFlow_ctor(
        double alpha, double gamma, double scale_factor, int inner_iterations,
        int outer_iterations, int solver_iterations)
{
    return rescueObjectFromPtr(cuda::BroxOpticalFlow::create(
            alpha, gamma, scale_factor, inner_iterations, outer_iterations, solver_iterations));
}

extern "C"
void BroxOpticalFlow_setFlowSmoothness(struct BroxOpticalFlowPtr ptr, double val)
{
    ptr->setFlowSmoothness(val);
}

extern "C"
double BroxOpticalFlow_getFlowSmoothness(struct BroxOpticalFlowPtr ptr)
{
    return ptr->getFlowSmoothness();
}

extern "C"
void BroxOpticalFlow_setGradientConstancyImportance(struct BroxOpticalFlowPtr ptr, double val)
{
    ptr->setGradientConstancyImportance(val);
}

extern "C"
double BroxOpticalFlow_getGradientConstancyImportance(struct BroxOpticalFlowPtr ptr)
{
    return ptr->getGradientConstancyImportance();
}

extern "C"
void BroxOpticalFlow_setPyramidScaleFactor(struct BroxOpticalFlowPtr ptr, double val)
{
    ptr->setPyramidScaleFactor(val);
}

extern "C"
double BroxOpticalFlow_getPyramidScaleFactor(struct BroxOpticalFlowPtr ptr)
{
    return ptr->getPyramidScaleFactor();
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

extern "C"
struct SparsePyrLKOpticalFlowPtr SparsePyrLKOpticalFlow_ctor(
        struct SizeWrapper winSize, int maxLevel, int iters, bool useInitialFlow)
{
    return rescueObjectFromPtr(cuda::SparsePyrLKOpticalFlow::create(
            winSize, maxLevel, iters, useInitialFlow));
}

extern "C"
void SparsePyrLKOpticalFlow_setWinSize(struct SparsePyrLKOpticalFlowPtr ptr, struct SizeWrapper val)
{
    ptr->setWinSize(val);
}

extern "C"
struct SizeWrapper SparsePyrLKOpticalFlow_getWinSize(struct SparsePyrLKOpticalFlowPtr ptr)
{
    return ptr->getWinSize();
}

extern "C"
void SparsePyrLKOpticalFlow_setMaxLevel(struct SparsePyrLKOpticalFlowPtr ptr, int val)
{
    ptr->setMaxLevel(val);
}

extern "C"
int SparsePyrLKOpticalFlow_getMaxLevel(struct SparsePyrLKOpticalFlowPtr ptr)
{
    return ptr->getMaxLevel();
}

extern "C"
void SparsePyrLKOpticalFlow_setNumIters(struct SparsePyrLKOpticalFlowPtr ptr, int val)
{
    ptr->setNumIters(val);
}

extern "C"
int SparsePyrLKOpticalFlow_getNumIters(struct SparsePyrLKOpticalFlowPtr ptr)
{
    return ptr->getNumIters();
}

extern "C"
void SparsePyrLKOpticalFlow_setUseInitialFlow(struct SparsePyrLKOpticalFlowPtr ptr, bool val)
{
    ptr->setUseInitialFlow(val);
}

extern "C"
bool SparsePyrLKOpticalFlow_getUseInitialFlow(struct SparsePyrLKOpticalFlowPtr ptr)
{
    return ptr->getUseInitialFlow();
}

extern "C"
struct DensePyrLKOpticalFlowPtr DensePyrLKOpticalFlow_ctor(
        struct SizeWrapper winSize, int maxLevel, int iters, bool useInitialFlow)
{
    return rescueObjectFromPtr(cuda::DensePyrLKOpticalFlow::create(
            winSize, maxLevel, iters, useInitialFlow));
}

extern "C"
void DensePyrLKOpticalFlow_setWinSize(struct DensePyrLKOpticalFlowPtr ptr, struct SizeWrapper val)
{
    ptr->setWinSize(val);
}

extern "C"
struct SizeWrapper DensePyrLKOpticalFlow_getWinSize(struct DensePyrLKOpticalFlowPtr ptr)
{
    return ptr->getWinSize();
}

extern "C"
void DensePyrLKOpticalFlow_setMaxLevel(struct DensePyrLKOpticalFlowPtr ptr, int val)
{
    ptr->setMaxLevel(val);
}

extern "C"
int DensePyrLKOpticalFlow_getMaxLevel(struct DensePyrLKOpticalFlowPtr ptr)
{
    return ptr->getMaxLevel();
}

extern "C"
void DensePyrLKOpticalFlow_setNumIters(struct DensePyrLKOpticalFlowPtr ptr, int val)
{
    ptr->setNumIters(val);
}

extern "C"
int DensePyrLKOpticalFlow_getNumIters(struct DensePyrLKOpticalFlowPtr ptr)
{
    return ptr->getNumIters();
}

extern "C"
void DensePyrLKOpticalFlow_setUseInitialFlow(struct DensePyrLKOpticalFlowPtr ptr, bool val)
{
    ptr->setUseInitialFlow(val);
}

extern "C"
bool DensePyrLKOpticalFlow_getUseInitialFlow(struct DensePyrLKOpticalFlowPtr ptr)
{
    return ptr->getUseInitialFlow();
}

extern "C"
struct FarnebackOpticalFlowPtr FarnebackOpticalFlow_ctor(
        int NumLevels, double PyrScale, bool FastPyramids, int WinSize,
        int NumIters, int PolyN, double PolySigma, int Flags)
{
    return rescueObjectFromPtr(cuda::FarnebackOpticalFlow::create(
            NumLevels, PyrScale, FastPyramids, WinSize, NumIters,
            PolyN, PolySigma, Flags));
}

extern "C"
void FarnebackOpticalFlow_setNumLevels(struct FarnebackOpticalFlowPtr ptr, int val)
{
    ptr->setNumLevels(val);
}

extern "C"
int FarnebackOpticalFlow_getNumLevels(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getNumLevels();
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
void FarnebackOpticalFlow_setFastPyramids(struct FarnebackOpticalFlowPtr ptr, bool val)
{
    ptr->setFastPyramids(val);
}

extern "C"
bool FarnebackOpticalFlow_getFastPyramids(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getFastPyramids();
}

extern "C"
void FarnebackOpticalFlow_setWinSize(struct FarnebackOpticalFlowPtr ptr, int val)
{
    ptr->setWinSize(val);
}

extern "C"
int FarnebackOpticalFlow_getWinSize(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getWinSize();
}

extern "C"
void FarnebackOpticalFlow_setNumIters(struct FarnebackOpticalFlowPtr ptr, int val)
{
    ptr->setNumIters(val);
}

extern "C"
int FarnebackOpticalFlow_getNumIters(struct FarnebackOpticalFlowPtr ptr)
{
    return ptr->getNumIters();
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
void FarnebackOpticalFlow_setPolySigma(struct FarnebackOpticalFlowPtr ptr, double val)
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

extern "C"
struct OpticalFlowDual_TVL1Ptr OpticalFlowDual_TVL1_ctor(
        double tau, double lambda, double theta, int nscales, int warps, double epsilon, 
        int iterations, double scaleStep, double gamma, bool useInitialFlow)
{
    return rescueObjectFromPtr(cuda::OpticalFlowDual_TVL1::create(
            tau, lambda, theta, nscales, warps, epsilon,
            iterations, scaleStep, gamma, useInitialFlow));
}

extern "C"
void OpticalFlowDual_TVL1_setTau(struct OpticalFlowDual_TVL1Ptr ptr, double val)
{
    ptr->setTau(val);
}

extern "C"
double OpticalFlowDual_TVL1_getTau(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getTau();
}

extern "C"
void OpticalFlowDual_TVL1_setLambda(struct OpticalFlowDual_TVL1Ptr ptr, double val)
{
    ptr->setLambda(val);
}

extern "C"
double OpticalFlowDual_TVL1_getLambda(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getLambda();
}

extern "C"
void OpticalFlowDual_TVL1_setGamma(struct OpticalFlowDual_TVL1Ptr ptr, double val)
{
    ptr->setGamma(val);
}

extern "C"
double OpticalFlowDual_TVL1_getGamma(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getGamma();
}

extern "C"
void OpticalFlowDual_TVL1_setTheta(struct OpticalFlowDual_TVL1Ptr ptr, double val)
{
    ptr->setTheta(val);
}

extern "C"
double OpticalFlowDual_TVL1_getTheta(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getTheta();
}

extern "C"
void OpticalFlowDual_TVL1_setNumScales(struct OpticalFlowDual_TVL1Ptr ptr, int val)
{
    ptr->setNumScales(val);
}

extern "C"
int OpticalFlowDual_TVL1_getNumScales(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getNumScales();
}

extern "C"
void OpticalFlowDual_TVL1_setNumWarps(struct OpticalFlowDual_TVL1Ptr ptr, int val)
{
    ptr->setNumWarps(val);
}

extern "C"
int OpticalFlowDual_TVL1_getNumWarps(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getNumWarps();
}

extern "C"
void OpticalFlowDual_TVL1_setEpsilon(struct OpticalFlowDual_TVL1Ptr ptr, double val)
{
    ptr->setEpsilon(val);
}

extern "C"
double OpticalFlowDual_TVL1_getEpsilon(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getEpsilon();
}

extern "C"
void OpticalFlowDual_TVL1_setNumIterations(struct OpticalFlowDual_TVL1Ptr ptr, int val)
{
    ptr->setNumIterations(val);
}

extern "C"
int OpticalFlowDual_TVL1_getNumIterations(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getNumIterations();
}

extern "C"
void OpticalFlowDual_TVL1_setScaleStep(struct OpticalFlowDual_TVL1Ptr ptr, double val)
{
    ptr->setScaleStep(val);
}

extern "C"
double OpticalFlowDual_TVL1_getScaleStep(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getScaleStep();
}

extern "C"
void OpticalFlowDual_TVL1_setUseInitialFlow(struct OpticalFlowDual_TVL1Ptr ptr, bool val)
{
    ptr->setUseInitialFlow(val);
}

extern "C"
bool OpticalFlowDual_TVL1_getUseInitialFlow(struct OpticalFlowDual_TVL1Ptr ptr)
{
    return ptr->getUseInitialFlow();
}

