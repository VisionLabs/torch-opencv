#include <CUDACommon.hpp>
#include <include/Classes.hpp>
#include <opencv2/cudabgsegm.hpp>

// BackgroundSubtractorMOG

struct BackgroundSubtractorMOGPtr {
    void *ptr;
    inline cuda::BackgroundSubtractorMOG * operator->() { return static_cast<cuda::BackgroundSubtractorMOG *>(ptr); }
    inline BackgroundSubtractorMOGPtr(cuda::BackgroundSubtractorMOG *ptr) { this->ptr = ptr; }
    inline cuda::BackgroundSubtractorMOG & operator*() { return *static_cast<cuda::BackgroundSubtractorMOG *>(this->ptr); }
};

// BackgroundSubtractorMOG2

struct BackgroundSubtractorMOG2Ptr {
    void *ptr;
    inline cuda::BackgroundSubtractorMOG2 * operator->() { return static_cast<cuda::BackgroundSubtractorMOG2 *>(ptr); }
    inline BackgroundSubtractorMOG2Ptr(cuda::BackgroundSubtractorMOG2 *ptr) { this->ptr = ptr; }
    inline cuda::BackgroundSubtractorMOG2 & operator*() { return *static_cast<cuda::BackgroundSubtractorMOG2 *>(this->ptr); }
};

extern "C"
struct BackgroundSubtractorMOGPtr BackgroundSubtractorMOG_ctorCuda(
        int History, int NMixtures, double BackgroundRatio, double NoiseSigma);

extern "C"
struct TensorWrapper BackgroundSubtractorMOG_applyCuda(struct cutorchInfo info,
                                                   struct BackgroundSubtractorMOGPtr ptr, struct TensorWrapper image,
                                                   struct TensorWrapper fgmask, double learningRate);

extern "C"
struct TensorWrapper BackgroundSubtractorMOG_getBackgroundImageCuda(
        struct cutorchInfo info, struct BackgroundSubtractorMOGPtr ptr,
        struct TensorWrapper backgroundImage);

extern "C"
void BackgroundSubtractorMOG_setHistoryCuda(struct BackgroundSubtractorMOGPtr ptr, int val);

extern "C"
int BackgroundSubtractorMOG_getHistoryCuda(struct BackgroundSubtractorMOGPtr ptr);

extern "C"
void BackgroundSubtractorMOG_setNMixturesCuda(struct BackgroundSubtractorMOGPtr ptr, int val);

extern "C"
int BackgroundSubtractorMOG_getNMixturesCuda(struct BackgroundSubtractorMOGPtr ptr);

extern "C"
void BackgroundSubtractorMOG_setBackgroundRatioCuda(struct BackgroundSubtractorMOGPtr ptr, double val);

extern "C"
double BackgroundSubtractorMOG_getBackgroundRatioCuda(struct BackgroundSubtractorMOGPtr ptr);

extern "C"
void BackgroundSubtractorMOG_setNoiseSigmaCuda(struct BackgroundSubtractorMOGPtr ptr, double val);

extern "C"
double BackgroundSubtractorMOG_getNoiseSigmaCuda(struct BackgroundSubtractorMOGPtr ptr);

extern "C"
struct BackgroundSubtractorMOG2Ptr BackgroundSubtractorMOG2_ctorCuda(
        int history, double varThreshold, bool detectShadows);

extern "C"
struct TensorWrapper BackgroundSubtractorMOG2_applyCuda(struct cutorchInfo info,
                                                    struct BackgroundSubtractorMOG2Ptr ptr, struct TensorWrapper image,
                                                    struct TensorWrapper fgmask, double learningRate);

extern "C"
struct TensorWrapper BackgroundSubtractorMOG2_getBackgroundImageCuda(
        struct cutorchInfo info, struct BackgroundSubtractorMOG2Ptr ptr,
        struct TensorWrapper backgroundImage);

extern "C"
int BackgroundSubtractorMOG2_getHistoryCuda(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setHistoryCuda(struct BackgroundSubtractorMOG2Ptr ptr, int history);

extern "C"
int BackgroundSubtractorMOG2_getNMixturesCuda(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setNMixturesCuda(struct BackgroundSubtractorMOG2Ptr ptr, int nmixtures);

extern "C"
int BackgroundSubtractorMOG2_getShadowValueCuda(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setShadowValueCuda(struct BackgroundSubtractorMOG2Ptr ptr, int shadow_value);

extern "C"
double BackgroundSubtractorMOG2_getBackgroundRatioCuda(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setBackgroundRatioCuda(struct BackgroundSubtractorMOG2Ptr ptr, double ratio);

extern "C"
double BackgroundSubtractorMOG2_getVarThresholdCuda(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setVarThresholdCuda(struct BackgroundSubtractorMOG2Ptr ptr, double varThreshold);

extern "C"
double BackgroundSubtractorMOG2_getVarThresholdGenCuda(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setVarThresholdGenCuda(struct BackgroundSubtractorMOG2Ptr ptr, double varThresholdGen);

extern "C"
double BackgroundSubtractorMOG2_getVarInitCuda(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setVarInitCuda(struct BackgroundSubtractorMOG2Ptr ptr, double varInit);

extern "C"
double BackgroundSubtractorMOG2_getVarMinCuda(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setVarMinCuda(struct BackgroundSubtractorMOG2Ptr ptr, double varMin);

extern "C"
double BackgroundSubtractorMOG2_getVarMaxCuda(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setVarMaxCuda(struct BackgroundSubtractorMOG2Ptr ptr, double varMax);

extern "C"
bool BackgroundSubtractorMOG2_getDetectShadowsCuda(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setDetectShadowsCuda(struct BackgroundSubtractorMOG2Ptr ptr, bool detectShadows);

extern "C"
double BackgroundSubtractorMOG2_getComplexityReductionThresholdCuda(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setComplexityReductionThresholdCuda(struct BackgroundSubtractorMOG2Ptr ptr, double ct);

extern "C"
double BackgroundSubtractorMOG2_getShadowThresholdCuda(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setShadowThresholdCuda(struct BackgroundSubtractorMOG2Ptr ptr, double shadowThreshold);
