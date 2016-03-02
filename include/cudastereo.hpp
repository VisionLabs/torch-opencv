#include <CUDACommon.hpp>
#include <include/Classes.hpp>
#include <opencv2/cudastereo.hpp>

// StereoBM

struct StereoBMPtr {
    void *ptr;
    inline cuda::StereoBM * operator->() { return static_cast<cuda::StereoBM *>(ptr); }
    inline StereoBMPtr(cuda::StereoBM *ptr) { this->ptr = ptr; }
    inline cuda::StereoBM & operator*() { return *static_cast<cuda::StereoBM *>(this->ptr); }
};

// StereoBeliefPropagation

struct StereoBeliefPropagationPtr {
    void *ptr;
    inline cuda::StereoBeliefPropagation * operator->() { return static_cast<cuda::StereoBeliefPropagation *>(ptr); }
    inline StereoBeliefPropagationPtr(cuda::StereoBeliefPropagation *ptr) { this->ptr = ptr; }
    inline cuda::StereoBeliefPropagation & operator*() { return *static_cast<cuda::StereoBeliefPropagation *>(this->ptr); }
};

// StereoConstantSpaceBP

struct StereoConstantSpaceBPPtr {
    void *ptr;
    inline cuda::StereoConstantSpaceBP * operator->() { return static_cast<cuda::StereoConstantSpaceBP *>(ptr); }
    inline StereoConstantSpaceBPPtr(cuda::StereoConstantSpaceBP *ptr) { this->ptr = ptr; }
    inline cuda::StereoConstantSpaceBP & operator*() { return *static_cast<cuda::StereoConstantSpaceBP *>(this->ptr); }
};

// DisparityBilateralFilter

struct DisparityBilateralFilterPtr {
    void *ptr;
    inline cuda::DisparityBilateralFilter * operator->() { return static_cast<cuda::DisparityBilateralFilter *>(ptr); }
    inline DisparityBilateralFilterPtr(cuda::DisparityBilateralFilter *ptr) { this->ptr = ptr; }
    inline cuda::DisparityBilateralFilter & operator*() { return *static_cast<cuda::DisparityBilateralFilter *>(this->ptr); }
};

extern "C"
struct StereoBMPtr createStereoBMCuda(int numDisparities, int blockSize);

extern "C"
struct TensorWrapper StereoBM_computeCuda(struct cutorchInfo info, struct StereoBMPtr ptr,
                                    struct TensorWrapper left, struct TensorWrapper right, struct TensorWrapper disparity);

extern "C"
struct StereoBeliefPropagationPtr createStereoBeliefPropagationCuda(
        int ndisp, int iters, int levels, int msg_type);

extern "C"
struct TensorWrapper StereoBeliefPropagation_computeCuda(struct cutorchInfo info,
                                                     struct StereoBeliefPropagationPtr ptr, struct TensorWrapper left,
                                                     struct TensorWrapper right, struct TensorWrapper disparity);

extern "C"
struct TensorWrapper StereoBeliefPropagation_compute2Cuda(struct cutorchInfo info,
                                                      struct StereoBeliefPropagationPtr ptr, struct TensorWrapper data,
                                                      struct TensorWrapper disparity);

extern "C"
void StereoBeliefPropagation_setNumItersCuda(struct StereoBeliefPropagationPtr ptr, int val);

extern "C"
int StereoBeliefPropagation_getNumItersCuda(struct StereoBeliefPropagationPtr ptr);

extern "C"
void StereoBeliefPropagation_setNumLevelsCuda(struct StereoBeliefPropagationPtr ptr, int val);

extern "C"
int StereoBeliefPropagation_getNumLevelsCuda(struct StereoBeliefPropagationPtr ptr);

extern "C"
void StereoBeliefPropagation_setMaxDataTermCuda(struct StereoBeliefPropagationPtr ptr, double val);

extern "C"
double StereoBeliefPropagation_getMaxDataTermCuda(struct StereoBeliefPropagationPtr ptr);

extern "C"
void StereoBeliefPropagation_setDataWeightCuda(struct StereoBeliefPropagationPtr ptr, double val);

extern "C"
double StereoBeliefPropagation_getDataWeightCuda(struct StereoBeliefPropagationPtr ptr);

extern "C"
void StereoBeliefPropagation_setMaxDiscTermCuda(struct StereoBeliefPropagationPtr ptr, double val);

extern "C"
double StereoBeliefPropagation_getMaxDiscTermCuda(struct StereoBeliefPropagationPtr ptr);

extern "C"
void StereoBeliefPropagation_setDiscSingleJumpCuda(struct StereoBeliefPropagationPtr ptr, double val);

extern "C"
double StereoBeliefPropagation_getDiscSingleJumpCuda(struct StereoBeliefPropagationPtr ptr);

extern "C"
void StereoBeliefPropagation_setMsgTypeCuda(struct StereoBeliefPropagationPtr ptr, int val);

extern "C"
int StereoBeliefPropagation_getMsgTypeCuda(struct StereoBeliefPropagationPtr ptr);

extern "C"
struct Vec3iWrapper StereoBeliefPropagation_estimateRecommendedParamsCuda(int width, int height);

extern "C"
int StereoConstantSpaceBP_getNrPlaneCuda(struct StereoConstantSpaceBPPtr ptr);

extern "C"
void StereoConstantSpaceBP_setNrPlaneCuda(struct StereoConstantSpaceBPPtr ptr, int val);

extern "C"
bool StereoConstantSpaceBP_getUseLocalInitDataCostCuda(struct StereoConstantSpaceBPPtr ptr);

extern "C"
void StereoConstantSpaceBP_setUseLocalInitDataCostCuda(struct StereoConstantSpaceBPPtr ptr, bool val);

extern "C"
struct Vec4iWrapper StereoConstantSpaceBP_estimateRecommendedParamsCuda(int width, int height);

extern "C"
struct StereoConstantSpaceBPPtr createStereoConstantSpaceBPCuda(
        int ndisp, int iters, int levels, int nr_plane, int msg_type);

extern "C"
struct TensorWrapper reprojectImageTo3DCuda(
        struct cutorchInfo info, struct TensorWrapper disp,
        struct TensorWrapper xyzw, struct TensorWrapper Q, int dst_cn);

extern "C"
struct TensorWrapper drawColorDispCuda(
        struct cutorchInfo info, struct TensorWrapper src_disp,
        struct TensorWrapper dst_disp, int ndisp);
