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

