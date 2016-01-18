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
struct BackgroundSubtractorMOGPtr BackgroundSubtractorMOG_ctor(
        int History, int NMixtures, double BackgroundRatio, double NoiseSigma);

extern "C"
struct TensorWrapper BackgroundSubtractorMOG_apply(THCState *state,
                                                   struct BackgroundSubtractorMOGPtr ptr, struct TensorWrapper image,
                                                   struct TensorWrapper fgmask, double learningRate);

extern "C"
struct TensorWrapper BackgroundSubtractorMOG_getBackgroundImage(
        THCState *state, struct BackgroundSubtractorMOGPtr ptr,
        struct TensorWrapper backgroundImage);

extern "C"
void BackgroundSubtractorMOG_setHistory(struct BackgroundSubtractorMOGPtr ptr, int val);

extern "C"
int BackgroundSubtractorMOG_getHistory(struct BackgroundSubtractorMOGPtr ptr);

extern "C"
void BackgroundSubtractorMOG_setNMixtures(struct BackgroundSubtractorMOGPtr ptr, int val);

extern "C"
int BackgroundSubtractorMOG_getNMixtures(struct BackgroundSubtractorMOGPtr ptr);

extern "C"
void BackgroundSubtractorMOG_setBackgroundRatio(struct BackgroundSubtractorMOGPtr ptr, double val);

extern "C"
double BackgroundSubtractorMOG_getBackgroundRatio(struct BackgroundSubtractorMOGPtr ptr);

extern "C"
void BackgroundSubtractorMOG_setNoiseSigma(struct BackgroundSubtractorMOGPtr ptr, double val);

extern "C"
double BackgroundSubtractorMOG_getNoiseSigma(struct BackgroundSubtractorMOGPtr ptr);

extern "C"
struct BackgroundSubtractorMOG2Ptr BackgroundSubtractorMOG2_ctor(
        int history, double varThreshold, bool detectShadows);

extern "C"
struct TensorWrapper BackgroundSubtractorMOG2_apply(THCState *state,
                                                    struct BackgroundSubtractorMOG2Ptr ptr, struct TensorWrapper image,
                                                    struct TensorWrapper fgmask, double learningRate);

extern "C"
struct TensorWrapper BackgroundSubtractorMOG2_getBackgroundImage(
        THCState *state, struct BackgroundSubtractorMOG2Ptr ptr,
        struct TensorWrapper backgroundImage);

extern "C"
int BackgroundSubtractorMOG2_getHistory(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setHistory(struct BackgroundSubtractorMOG2Ptr ptr, int history);

extern "C"
int BackgroundSubtractorMOG2_getNMixtures(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setNMixtures(struct BackgroundSubtractorMOG2Ptr ptr, int nmixtures);

extern "C"
int BackgroundSubtractorMOG2_getShadowValue(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setShadowValue(struct BackgroundSubtractorMOG2Ptr ptr, int shadow_value);

extern "C"
double BackgroundSubtractorMOG2_getBackgroundRatio(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setBackgroundRatio(struct BackgroundSubtractorMOG2Ptr ptr, double ratio);

extern "C"
double BackgroundSubtractorMOG2_getVarThreshold(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setVarThreshold(struct BackgroundSubtractorMOG2Ptr ptr, double varThreshold);

extern "C"
double BackgroundSubtractorMOG2_getVarThresholdGen(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setVarThresholdGen(struct BackgroundSubtractorMOG2Ptr ptr, double varThresholdGen);

extern "C"
double BackgroundSubtractorMOG2_getVarInit(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setVarInit(struct BackgroundSubtractorMOG2Ptr ptr, double varInit);

extern "C"
double BackgroundSubtractorMOG2_getVarMin(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setVarMin(struct BackgroundSubtractorMOG2Ptr ptr, double varMin);

extern "C"
double BackgroundSubtractorMOG2_getVarMax(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setVarMax(struct BackgroundSubtractorMOG2Ptr ptr, double varMax);

extern "C"
bool BackgroundSubtractorMOG2_getDetectShadows(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setDetectShadows(struct BackgroundSubtractorMOG2Ptr ptr, bool detectShadows);

extern "C"
double BackgroundSubtractorMOG2_getComplexityReductionThreshold(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setComplexityReductionThreshold(struct BackgroundSubtractorMOG2Ptr ptr, double ct);

extern "C"
double BackgroundSubtractorMOG2_getShadowThreshold(struct BackgroundSubtractorMOG2Ptr ptr);

extern "C"
void BackgroundSubtractorMOG2_setShadowThreshold(struct BackgroundSubtractorMOG2Ptr ptr, double shadowThreshold);
