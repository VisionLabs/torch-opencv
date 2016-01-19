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

