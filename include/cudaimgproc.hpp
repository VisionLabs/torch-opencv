#include <CUDACommon.hpp>
#include <include/Classes.hpp>
#include <opencv2/cudaimgproc.hpp>

// CornernessCriteria

struct CornernessCriteriaPtr {
    void *ptr;
    inline cuda::CornernessCriteria * operator->() { return static_cast<cuda::CornernessCriteria *>(ptr); }
    inline CornernessCriteriaPtr(cuda::CornernessCriteria *ptr) { this->ptr = ptr; }
    inline cuda::CornernessCriteria & operator*() { return *static_cast<cuda::CornernessCriteria *>(this->ptr); }
};

