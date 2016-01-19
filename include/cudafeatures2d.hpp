#include <CUDACommon.hpp>
#include <include/Classes.hpp>
#include <opencv2/cudafeatures2d.hpp>

// DescriptorMatcher

struct DescriptorMatcherPtr {
    void *ptr;
    inline cuda::DescriptorMatcher * operator->() { return static_cast<cuda::DescriptorMatcher *>(ptr); }
    inline DescriptorMatcherPtr(cuda::DescriptorMatcher *ptr) { this->ptr = ptr; }
    inline cuda::DescriptorMatcher & operator*() { return *static_cast<cuda::DescriptorMatcher *>(this->ptr); }
};

// Feature2DAsync

struct Feature2DAsyncPtr {
    void *ptr;
    inline cuda::Feature2DAsync * operator->() { return static_cast<cuda::Feature2DAsync *>(ptr); }
    inline Feature2DAsyncPtr(cuda::Feature2DAsync *ptr) { this->ptr = ptr; }
    inline cuda::Feature2DAsync & operator*() { return *static_cast<cuda::Feature2DAsync *>(this->ptr); }
};

