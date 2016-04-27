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

extern "C"
struct TensorWrapper DenseOpticalFlow_calcCuda(struct cutorchInfo info,
                                           struct DenseOpticalFlowPtr ptr, struct TensorWrapper I0, struct TensorWrapper I1,
                                           struct TensorWrapper flow);

extern "C"
struct BroxOpticalFlowPtr BroxOpticalFlow_ctorCuda(
        double alpha, double gamma, double scale_factor, int inner_iterations,
        int outer_iterations, int solver_iterations);

extern "C"
void BroxOpticalFlow_setFlowSmoothnessCuda(struct BroxOpticalFlowPtr ptr, double val);

extern "C"
double BroxOpticalFlow_getFlowSmoothnessCuda(struct BroxOpticalFlowPtr ptr);

extern "C"
void BroxOpticalFlow_setGradientConstancyImportanceCuda(struct BroxOpticalFlowPtr ptr, double val);

extern "C"
double BroxOpticalFlow_getGradientConstancyImportanceCuda(struct BroxOpticalFlowPtr ptr);

extern "C"
void BroxOpticalFlow_setPyramidScaleFactorCuda(struct BroxOpticalFlowPtr ptr, double val);

extern "C"
double BroxOpticalFlow_getPyramidScaleFactorCuda(struct BroxOpticalFlowPtr ptr);

extern "C"
void BroxOpticalFlow_setInnerIterationsCuda(struct BroxOpticalFlowPtr ptr, int val);

extern "C"
int BroxOpticalFlow_getInnerIterationsCuda(struct BroxOpticalFlowPtr ptr);

extern "C"
void BroxOpticalFlow_setOuterIterationsCuda(struct BroxOpticalFlowPtr ptr, int val);

extern "C"
int BroxOpticalFlow_getOuterIterationsCuda(struct BroxOpticalFlowPtr ptr);

extern "C"
void BroxOpticalFlow_setSolverIterationsCuda(struct BroxOpticalFlowPtr ptr, int val);

extern "C"
int BroxOpticalFlow_getSolverIterationsCuda(struct BroxOpticalFlowPtr ptr);

extern "C"
struct SparsePyrLKOpticalFlowPtr SparsePyrLKOpticalFlow_ctorCuda(
        struct SizeWrapper winSize, int maxLevel, int iters, bool useInitialFlow);

extern "C"
void SparsePyrLKOpticalFlow_setWinSizeCuda(struct SparsePyrLKOpticalFlowPtr ptr, struct SizeWrapper val);

extern "C"
struct SizeWrapper SparsePyrLKOpticalFlow_getWinSizeCuda(struct SparsePyrLKOpticalFlowPtr ptr);

extern "C"
void SparsePyrLKOpticalFlow_setMaxLevelCuda(struct SparsePyrLKOpticalFlowPtr ptr, int val);

extern "C"
int SparsePyrLKOpticalFlow_getMaxLevelCuda(struct SparsePyrLKOpticalFlowPtr ptr);

extern "C"
void SparsePyrLKOpticalFlow_setNumItersCuda(struct SparsePyrLKOpticalFlowPtr ptr, int val);

extern "C"
int SparsePyrLKOpticalFlow_getNumItersCuda(struct SparsePyrLKOpticalFlowPtr ptr);

extern "C"
void SparsePyrLKOpticalFlow_setUseInitialFlowCuda(struct SparsePyrLKOpticalFlowPtr ptr, bool val);

extern "C"
bool SparsePyrLKOpticalFlow_getUseInitialFlowCuda(struct SparsePyrLKOpticalFlowPtr ptr);

extern "C"
struct DensePyrLKOpticalFlowPtr DensePyrLKOpticalFlow_ctorCuda(
        struct SizeWrapper winSize, int maxLevel, int iters, bool useInitialFlow);

extern "C"
void DensePyrLKOpticalFlow_setWinSizeCuda(struct DensePyrLKOpticalFlowPtr ptr, struct SizeWrapper val);

extern "C"
struct SizeWrapper DensePyrLKOpticalFlow_getWinSizeCuda(struct DensePyrLKOpticalFlowPtr ptr);

extern "C"
void DensePyrLKOpticalFlow_setMaxLevelCuda(struct DensePyrLKOpticalFlowPtr ptr, int val);

extern "C"
int DensePyrLKOpticalFlow_getMaxLevelCuda(struct DensePyrLKOpticalFlowPtr ptr);

extern "C"
void DensePyrLKOpticalFlow_setNumItersCuda(struct DensePyrLKOpticalFlowPtr ptr, int val);

extern "C"
int DensePyrLKOpticalFlow_getNumItersCuda(struct DensePyrLKOpticalFlowPtr ptr);

extern "C"
void DensePyrLKOpticalFlow_setUseInitialFlowCuda(struct DensePyrLKOpticalFlowPtr ptr, bool val);

extern "C"
bool DensePyrLKOpticalFlow_getUseInitialFlowCuda(struct DensePyrLKOpticalFlowPtr ptr);

extern "C"
struct FarnebackOpticalFlowPtr FarnebackOpticalFlow_ctorCuda(
        int NumLevels, double PyrScale, bool FastPyramids, int WinSize,
        int NumIters, int PolyN, double PolySigma, int Flags);

extern "C"
void FarnebackOpticalFlow_setNumLevelsCuda(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C"
int FarnebackOpticalFlow_getNumLevelsCuda(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setPyrScaleCuda(struct FarnebackOpticalFlowPtr ptr, double val);

extern "C"
double FarnebackOpticalFlow_getPyrScaleCuda(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setFastPyramidsCuda(struct FarnebackOpticalFlowPtr ptr, bool val);

extern "C"
bool FarnebackOpticalFlow_getFastPyramidsCuda(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setWinSizeCuda(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C"
int FarnebackOpticalFlow_getWinSizeCuda(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setNumItersCuda(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C"
int FarnebackOpticalFlow_getNumItersCuda(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setPolyNCuda(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C"
int FarnebackOpticalFlow_getPolyNCuda(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setPolySigmaCuda(struct FarnebackOpticalFlowPtr ptr, double val);

extern "C"
double FarnebackOpticalFlow_getPolySigmaCuda(struct FarnebackOpticalFlowPtr ptr);

extern "C"
void FarnebackOpticalFlow_setFlagsCuda(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C"
int FarnebackOpticalFlow_getFlagsCuda(struct FarnebackOpticalFlowPtr ptr);

extern "C"
struct OpticalFlowDual_TVL1Ptr OpticalFlowDual_TVL1_ctorCuda(
        double tau, double lambda, double theta, int nscales, int warps, double epsilon,
        int iterations, double scaleStep, double gamma, bool useInitialFlow);

extern "C"
void OpticalFlowDual_TVL1_setTauCuda(struct OpticalFlowDual_TVL1Ptr ptr, double val);

extern "C"
double OpticalFlowDual_TVL1_getTauCuda(struct OpticalFlowDual_TVL1Ptr ptr);

extern "C"
void OpticalFlowDual_TVL1_setLambdaCuda(struct OpticalFlowDual_TVL1Ptr ptr, double val);

extern "C"
double OpticalFlowDual_TVL1_getLambdaCuda(struct OpticalFlowDual_TVL1Ptr ptr);

extern "C"
void OpticalFlowDual_TVL1_setGammaCuda(struct OpticalFlowDual_TVL1Ptr ptr, double val);

extern "C"
double OpticalFlowDual_TVL1_getGammaCuda(struct OpticalFlowDual_TVL1Ptr ptr);

extern "C"
void OpticalFlowDual_TVL1_setThetaCuda(struct OpticalFlowDual_TVL1Ptr ptr, double val);

extern "C"
double OpticalFlowDual_TVL1_getThetaCuda(struct OpticalFlowDual_TVL1Ptr ptr);

extern "C"
void OpticalFlowDual_TVL1_setNumScalesCuda(struct OpticalFlowDual_TVL1Ptr ptr, int val);

extern "C"
int OpticalFlowDual_TVL1_getNumScalesCuda(struct OpticalFlowDual_TVL1Ptr ptr);

extern "C"
void OpticalFlowDual_TVL1_setNumWarpsCuda(struct OpticalFlowDual_TVL1Ptr ptr, int val);

extern "C"
int OpticalFlowDual_TVL1_getNumWarpsCuda(struct OpticalFlowDual_TVL1Ptr ptr);

extern "C"
void OpticalFlowDual_TVL1_setEpsilonCuda(struct OpticalFlowDual_TVL1Ptr ptr, double val);

extern "C"
double OpticalFlowDual_TVL1_getEpsilonCuda(struct OpticalFlowDual_TVL1Ptr ptr);

extern "C"
void OpticalFlowDual_TVL1_setNumIterationsCuda(struct OpticalFlowDual_TVL1Ptr ptr, int val);

extern "C"
int OpticalFlowDual_TVL1_getNumIterationsCuda(struct OpticalFlowDual_TVL1Ptr ptr);

extern "C"
void OpticalFlowDual_TVL1_setScaleStepCuda(struct OpticalFlowDual_TVL1Ptr ptr, double val);

extern "C"
double OpticalFlowDual_TVL1_getScaleStepCuda(struct OpticalFlowDual_TVL1Ptr ptr);

extern "C"
void OpticalFlowDual_TVL1_setUseInitialFlowCuda(struct OpticalFlowDual_TVL1Ptr ptr, bool val);

extern "C"
bool OpticalFlowDual_TVL1_getUseInitialFlowCuda(struct OpticalFlowDual_TVL1Ptr ptr);
