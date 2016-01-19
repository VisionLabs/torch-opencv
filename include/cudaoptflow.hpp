#include <CUDACommon.hpp>
#include <include/Classes.hpp>
#include <opencv2/cudaoptflow.hpp>

// DenseOpticalFlow

struct DenseOpticalFlowPtr {
    void *ptr;
    inline cuda::DenseOpticalFlow * operator->() { return static_cast<cuda::DenseOpticalFlow *>(ptr); }
    inline DenseOpticalFlowPtr(cuda::DenseOpticalFlow *ptr) { this->ptr = ptr; }
    inline cuda::DenseOpticalFlow & operator*() { return *static_cast<cuda::DenseOpticalFlow *>(this->ptr); }
};

// SparseOpticalFlow

struct SparseOpticalFlowPtr {
    void *ptr;
    inline cuda::SparseOpticalFlow * operator->() { return static_cast<cuda::SparseOpticalFlow *>(ptr); }
    inline SparseOpticalFlowPtr(cuda::SparseOpticalFlow *ptr) { this->ptr = ptr; }
    inline cuda::SparseOpticalFlow & operator*() { return *static_cast<cuda::SparseOpticalFlow *>(this->ptr); }
};

// BroxOpticalFlow

struct BroxOpticalFlowPtr {
    void *ptr;
    inline cuda::BroxOpticalFlow * operator->() { return static_cast<cuda::BroxOpticalFlow *>(ptr); }
    inline BroxOpticalFlowPtr(cuda::BroxOpticalFlow *ptr) { this->ptr = ptr; }
    inline cuda::BroxOpticalFlow & operator*() { return *static_cast<cuda::BroxOpticalFlow *>(this->ptr); }
};

// SparsePyrLKOpticalFlow

struct SparsePyrLKOpticalFlowPtr {
    void *ptr;
    inline cuda::SparsePyrLKOpticalFlow * operator->() { return static_cast<cuda::SparsePyrLKOpticalFlow *>(ptr); }
    inline SparsePyrLKOpticalFlowPtr(cuda::SparsePyrLKOpticalFlow *ptr) { this->ptr = ptr; }
    inline cuda::SparsePyrLKOpticalFlow & operator*() { return *static_cast<cuda::SparsePyrLKOpticalFlow *>(this->ptr); }
};

// DensePyrLKOpticalFlow

struct DensePyrLKOpticalFlowPtr {
    void *ptr;
    inline cuda::DensePyrLKOpticalFlow * operator->() { return static_cast<cuda::DensePyrLKOpticalFlow *>(ptr); }
    inline DensePyrLKOpticalFlowPtr(cuda::DensePyrLKOpticalFlow *ptr) { this->ptr = ptr; }
    inline cuda::DensePyrLKOpticalFlow & operator*() { return *static_cast<cuda::DensePyrLKOpticalFlow *>(this->ptr); }
};

// FarnebackOpticalFlow

struct FarnebackOpticalFlowPtr {
    void *ptr;
    inline cuda::FarnebackOpticalFlow * operator->() { return static_cast<cuda::FarnebackOpticalFlow *>(ptr); }
    inline FarnebackOpticalFlowPtr(cuda::FarnebackOpticalFlow *ptr) { this->ptr = ptr; }
    inline cuda::FarnebackOpticalFlow & operator*() { return *static_cast<cuda::FarnebackOpticalFlow *>(this->ptr); }
};

// OpticalFlowDual_TVL1

struct OpticalFlowDual_TVL1Ptr {
    void *ptr;
    inline cuda::OpticalFlowDual_TVL1 * operator->() { return static_cast<cuda::OpticalFlowDual_TVL1 *>(ptr); }
    inline OpticalFlowDual_TVL1Ptr(cuda::OpticalFlowDual_TVL1 *ptr) { this->ptr = ptr; }
    inline cuda::OpticalFlowDual_TVL1 & operator*() { return *static_cast<cuda::OpticalFlowDual_TVL1 *>(this->ptr); }
};

struct TensorWrapper DenseOpticalFlow_calc(struct cutorchInfo info,
                                           struct DenseOpticalFlowPtr ptr, struct TensorWrapper I0, struct TensorWrapper I1,
                                           struct TensorWrapper flow);

extern "C"
struct BroxOpticalFlowPtr BroxOpticalFlow_ctor(
        double alpha, double gamma, double scale_factor, int inner_iterations,
        int outer_iterations, int solver_iterations);

extern "C"
void BroxOpticalFlow_setFlowSmoothness(struct BroxOpticalFlowPtr ptr, double val);

extern "C"
double BroxOpticalFlow_getFlowSmoothness(struct BroxOpticalFlowPtr ptr);

extern "C"
void BroxOpticalFlow_setGradientConstancyImportance(struct BroxOpticalFlowPtr ptr, double val);

extern "C"
double BroxOpticalFlow_getGradientConstancyImportance(struct BroxOpticalFlowPtr ptr);

extern "C"
void BroxOpticalFlow_setPyramidScaleFactor(struct BroxOpticalFlowPtr ptr, double val);

extern "C"
double BroxOpticalFlow_getPyramidScaleFactor(struct BroxOpticalFlowPtr ptr);

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
struct SparsePyrLKOpticalFlowPtr SparsePyrLKOpticalFlow_ctor(
        struct SizeWrapper winSize, int maxLevel, int iters, bool useInitialFlow);

extern "C"
void SparsePyrLKOpticalFlow_setWinSize(struct SparsePyrLKOpticalFlowPtr ptr, struct SizeWrapper val);

extern "C"
struct SizeWrapper SparsePyrLKOpticalFlow_getWinSize(struct SparsePyrLKOpticalFlowPtr ptr);

extern "C"
void SparsePyrLKOpticalFlow_setMaxLevel(struct SparsePyrLKOpticalFlowPtr ptr, int val);

extern "C"
int SparsePyrLKOpticalFlow_getMaxLevel(struct SparsePyrLKOpticalFlowPtr ptr);

extern "C"
void SparsePyrLKOpticalFlow_setNumIters(struct SparsePyrLKOpticalFlowPtr ptr, int val);

extern "C"
int SparsePyrLKOpticalFlow_getNumIters(struct SparsePyrLKOpticalFlowPtr ptr);

extern "C"
void SparsePyrLKOpticalFlow_setUseInitialFlow(struct SparsePyrLKOpticalFlowPtr ptr, bool val);

extern "C"
bool SparsePyrLKOpticalFlow_getUseInitialFlow(struct SparsePyrLKOpticalFlowPtr ptr);

extern "C"
struct DensePyrLKOpticalFlowPtr DensePyrLKOpticalFlow_ctor(
        struct SizeWrapper winSize, int maxLevel, int iters, bool useInitialFlow);

extern "C"
void DensePyrLKOpticalFlow_setWinSize(struct DensePyrLKOpticalFlowPtr ptr, struct SizeWrapper val);

extern "C"
struct SizeWrapper DensePyrLKOpticalFlow_getWinSize(struct DensePyrLKOpticalFlowPtr ptr);

extern "C"
void DensePyrLKOpticalFlow_setMaxLevel(struct DensePyrLKOpticalFlowPtr ptr, int val);

extern "C"
int DensePyrLKOpticalFlow_getMaxLevel(struct DensePyrLKOpticalFlowPtr ptr);

extern "C"
void DensePyrLKOpticalFlow_setNumIters(struct DensePyrLKOpticalFlowPtr ptr, int val);

extern "C"
int DensePyrLKOpticalFlow_getNumIters(struct DensePyrLKOpticalFlowPtr ptr);

extern "C"
void DensePyrLKOpticalFlow_setUseInitialFlow(struct DensePyrLKOpticalFlowPtr ptr, bool val);

extern "C"
bool DensePyrLKOpticalFlow_getUseInitialFlow(struct DensePyrLKOpticalFlowPtr ptr);

extern "C"
struct FarnebackOpticalFlowPtr FarnebackOpticalFlow_ctor(
        int NumLevels, double PyrScale, bool FastPyramids, int WinSize,
        int NumIters, int PolyN, double PolySigma, int Flags);

extern "C"
void FarnebackOpticalFlow_setNumLevels(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C"
int FarnebackOpticalFlow_getNumLevels(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setPyrScale(struct FarnebackOpticalFlowPtr ptr, double val);

extern "C"
double FarnebackOpticalFlow_getPyrScale(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setFastPyramids(struct FarnebackOpticalFlowPtr ptr, bool val);

extern "C"
bool FarnebackOpticalFlow_getFastPyramids(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setWinSize(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C"
int FarnebackOpticalFlow_getWinSize(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setNumIters(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C"
int FarnebackOpticalFlow_getNumIters(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setPolyN(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C"
int FarnebackOpticalFlow_getPolyN(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setPolySigma(struct FarnebackOpticalFlowPtr ptr, double val);

extern "C"
double FarnebackOpticalFlow_getPolySigma(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setFlags(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C"
int FarnebackOpticalFlow_getFlags(struct FarnebackOpticalFlowPtr ptr);

extern "C"
struct OpticalFlowDual_TVL1Ptr OpticalFlowDual_TVL1_ctor(
        double tau, double lambda, double theta, int nscales, int warps, double epsilon,
        int iterations, double scaleStep, double gamma, bool useInitialFlow);

extern "C"
void OpticalFlowDual_TVL1_setTau(struct OpticalFlowDual_TVL1Ptr ptr, double val);

extern "C"
double OpticalFlowDual_TVL1_getTau(struct OpticalFlowDual_TVL1Ptr ptr);

extern "C"
void OpticalFlowDual_TVL1_setLambda(struct OpticalFlowDual_TVL1Ptr ptr, double val);

extern "C"
double OpticalFlowDual_TVL1_getLambda(struct OpticalFlowDual_TVL1Ptr ptr);

extern "C"
void OpticalFlowDual_TVL1_setGamma(struct OpticalFlowDual_TVL1Ptr ptr, double val);

extern "C"
double OpticalFlowDual_TVL1_getGamma(struct OpticalFlowDual_TVL1Ptr ptr);

extern "C"
void OpticalFlowDual_TVL1_setTheta(struct OpticalFlowDual_TVL1Ptr ptr, double val);

extern "C"
double OpticalFlowDual_TVL1_getTheta(struct OpticalFlowDual_TVL1Ptr ptr);

extern "C"
void OpticalFlowDual_TVL1_setNumScales(struct OpticalFlowDual_TVL1Ptr ptr, int val);

extern "C"
int OpticalFlowDual_TVL1_getNumScales(struct OpticalFlowDual_TVL1Ptr ptr);

extern "C"
void OpticalFlowDual_TVL1_setNumWarps(struct OpticalFlowDual_TVL1Ptr ptr, int val);

extern "C"
int OpticalFlowDual_TVL1_getNumWarps(struct OpticalFlowDual_TVL1Ptr ptr);

extern "C"
void OpticalFlowDual_TVL1_setEpsilon(struct OpticalFlowDual_TVL1Ptr ptr, double val);

extern "C"
double OpticalFlowDual_TVL1_getEpsilon(struct OpticalFlowDual_TVL1Ptr ptr);

extern "C"
void OpticalFlowDual_TVL1_setNumIterations(struct OpticalFlowDual_TVL1Ptr ptr, int val);

extern "C"
int OpticalFlowDual_TVL1_getNumIterations(struct OpticalFlowDual_TVL1Ptr ptr);

extern "C"
void OpticalFlowDual_TVL1_setScaleStep(struct OpticalFlowDual_TVL1Ptr ptr, double val);

extern "C"
double OpticalFlowDual_TVL1_getScaleStep(struct OpticalFlowDual_TVL1Ptr ptr);

extern "C"
void OpticalFlowDual_TVL1_setUseInitialFlow(struct OpticalFlowDual_TVL1Ptr ptr, bool val);

extern "C"
bool OpticalFlowDual_TVL1_getUseInitialFlow(struct OpticalFlowDual_TVL1Ptr ptr);
