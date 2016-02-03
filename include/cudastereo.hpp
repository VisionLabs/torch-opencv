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
struct StereoBMPtr createStereoBM(int numDisparities, int blockSize);

extern "C"
struct TensorWrapper StereoBM_compute(struct cutorchInfo info, struct StereoBMPtr ptr,
                                    struct TensorWrapper left, struct TensorWrapper right, struct TensorWrapper disparity);

extern "C"
struct StereoBeliefPropagationPtr createStereoBeliefPropagation(
        int ndisp, int iters, int levels, int msg_type);

extern "C"
struct TensorWrapper StereoBeliefPropagation_compute(struct cutorchInfo info,
                                                     struct StereoBeliefPropagationPtr ptr, struct TensorWrapper left,
                                                     struct TensorWrapper right, struct TensorWrapper disparity);

extern "C"
struct TensorWrapper StereoBeliefPropagation_compute2(struct cutorchInfo info,
                                                      struct StereoBeliefPropagationPtr ptr, struct TensorWrapper data,
                                                      struct TensorWrapper disparity);

extern "C"
void StereoBeliefPropagation_setNumIters(struct StereoBeliefPropagationPtr ptr, int val);

extern "C"
int StereoBeliefPropagation_getNumIters(struct StereoBeliefPropagationPtr ptr);

extern "C"
void StereoBeliefPropagation_setNumLevels(struct StereoBeliefPropagationPtr ptr, int val);

extern "C"
int StereoBeliefPropagation_getNumLevels(struct StereoBeliefPropagationPtr ptr);

extern "C"
void StereoBeliefPropagation_setMaxDataTerm(struct StereoBeliefPropagationPtr ptr, double val);

extern "C"
double StereoBeliefPropagation_getMaxDataTerm(struct StereoBeliefPropagationPtr ptr);

extern "C"
void StereoBeliefPropagation_setDataWeight(struct StereoBeliefPropagationPtr ptr, double val);

extern "C"
double StereoBeliefPropagation_getDataWeight(struct StereoBeliefPropagationPtr ptr);

extern "C"
void StereoBeliefPropagation_setMaxDiscTerm(struct StereoBeliefPropagationPtr ptr, double val);

extern "C"
double StereoBeliefPropagation_getMaxDiscTerm(struct StereoBeliefPropagationPtr ptr);

extern "C"
void StereoBeliefPropagation_setDiscSingleJump(struct StereoBeliefPropagationPtr ptr, double val);

extern "C"
double StereoBeliefPropagation_getDiscSingleJump(struct StereoBeliefPropagationPtr ptr);

extern "C"
void StereoBeliefPropagation_setMsgType(struct StereoBeliefPropagationPtr ptr, int val);

extern "C"
int StereoBeliefPropagation_getMsgType(struct StereoBeliefPropagationPtr ptr);

extern "C"
struct Vec3iWrapper StereoBeliefPropagation_estimateRecommendedParams(int width, int height);

extern "C"
int StereoConstantSpaceBP_getNrPlane(struct StereoConstantSpaceBPPtr ptr);

extern "C"
void StereoConstantSpaceBP_setNrPlane(struct StereoConstantSpaceBPPtr ptr, int val);

extern "C"
bool StereoConstantSpaceBP_getUseLocalInitDataCost(struct StereoConstantSpaceBPPtr ptr);

extern "C"
void StereoConstantSpaceBP_setUseLocalInitDataCost(struct StereoConstantSpaceBPPtr ptr, bool val);

extern "C"
struct Vec4iWrapper StereoConstantSpaceBP_estimateRecommendedParams(int width, int height);

extern "C"
struct StereoConstantSpaceBPPtr createStereoConstantSpaceBP(
        int ndisp, int iters, int levels, int nr_plane, int msg_type);

extern "C"
struct TensorWrapper reprojectImageTo3D(
        struct cutorchInfo info, struct TensorWrapper disp,
        struct TensorWrapper xyzw, struct TensorWrapper Q, int dst_cn);

extern "C"
struct TensorWrapper drawColorDisp(
        struct cutorchInfo info, struct TensorWrapper src_disp,
        struct TensorWrapper dst_disp, int ndisp);
