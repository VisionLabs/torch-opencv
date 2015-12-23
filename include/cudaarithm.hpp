#include <CUDACommon.hpp>
#include <include/Classes.hpp>
#include <opencv2/cudaarithm.hpp>

struct LookUpTablePtr {
    void *ptr;

    inline cuda::LookUpTable * operator->() { return static_cast<cuda::LookUpTable *>(ptr); }
    inline LookUpTablePtr(cuda::LookUpTable *ptr) { this->ptr = ptr; }
    inline operator const cuda::LookUpTable &() { return *static_cast<cuda::LookUpTable *>(ptr); }
};

struct ConvolutionPtr {
    void *ptr;

    inline cuda::Convolution * operator->() { return static_cast<cuda::Convolution *>(ptr); }
    inline ConvolutionPtr(cuda::Convolution *ptr) { this->ptr = ptr; }
    inline operator const cuda::Convolution &() { return *static_cast<cuda::Convolution *>(ptr); }
};