#include <CUDACommon.hpp>
#include <include/Classes.hpp>
#include <opencv2/cudaobjdetect.hpp>

// HOG

struct HOGPtr {
    void *ptr;
    inline cuda::HOG * operator->() { return static_cast<cuda::HOG *>(ptr); }
    inline HOGPtr(cuda::HOG *ptr) { this->ptr = ptr; }
    inline cuda::HOG & operator*() { return *static_cast<cuda::HOG *>(this->ptr); }
};

// CascadeClassifier

struct CascadeClassifierPtr {
    void *ptr;
    inline cuda::CascadeClassifier * operator->() { return static_cast<cuda::CascadeClassifier *>(ptr); }
    inline CascadeClassifierPtr(cuda::CascadeClassifier *ptr) { this->ptr = ptr; }
    inline cuda::CascadeClassifier & operator*() { return *static_cast<cuda::CascadeClassifier *>(this->ptr); }
};