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

struct HOGPtr HOG_ctor(
        struct SizeWrapper win_size, struct SizeWrapper block_size,
        struct SizeWrapper block_stride, struct SizeWrapper cell_size, int nbins);

extern "C"
void HOG_setWinSigma(struct HOGPtr ptr, double val);

extern "C"
double HOG_getWinSigma(struct HOGPtr ptr);

extern "C"
void HOG_setL2HysThreshold(struct HOGPtr ptr, double val);

extern "C"
double HOG_getL2HysThreshold(struct HOGPtr ptr);

extern "C"
void HOG_setGammaCorrection(struct HOGPtr ptr, bool val);

extern "C"
bool HOG_getGammaCorrection(struct HOGPtr ptr);

extern "C"
void HOG_setNumLevels(struct HOGPtr ptr, int val);

extern "C"
int HOG_getNumLevels(struct HOGPtr ptr);

extern "C"
void HOG_setHitThreshold(struct HOGPtr ptr, double val);

extern "C"
double HOG_getHitThreshold(struct HOGPtr ptr);

extern "C"
void HOG_setWinStride(struct HOGPtr ptr, struct SizeWrapper val);

extern "C"
struct SizeWrapper HOG_getWinStride(struct HOGPtr ptr);

extern "C"
void HOG_setScaleFactor(struct HOGPtr ptr, double val);

extern "C"
double HOG_getScaleFactor(struct HOGPtr ptr);

extern "C"
void HOG_setGroupThreshold(struct HOGPtr ptr, int val);

extern "C"
int HOG_getGroupThreshold(struct HOGPtr ptr);

extern "C"
void HOG_setDescriptorFormat(struct HOGPtr ptr, int val);

extern "C"
int HOG_getDescriptorFormat(struct HOGPtr ptr);

extern "C"
size_t HOG_getDescriptorSize(struct HOGPtr ptr);

extern "C"
size_t HOG_getBlockHistogramSize(struct HOGPtr ptr);

extern "C"
void HOG_setSVMDetector(struct HOGPtr ptr, struct TensorWrapper val);

extern "C"
struct TensorWrapper HOG_getDefaultPeopleDetector(struct HOGPtr ptr);

extern "C"
struct TensorPlusPointArray HOG_detect(
        struct cutorchInfo info, struct HOGPtr ptr, struct TensorWrapper img);

extern "C"
struct TensorPlusRectArray HOG_detectMultiScale(
        struct cutorchInfo info, struct HOGPtr ptr, struct TensorWrapper img);

extern "C"
struct TensorWrapper HOG_compute(
        struct cutorchInfo info, struct HOGPtr ptr, struct TensorWrapper img,
        struct TensorWrapper descriptors);

extern "C"
struct CascadeClassifierPtr CascadeClassifier_ctor_filename(const char *filename);

extern "C"
struct CascadeClassifierPtr CascadeClassifier_ctor_file(struct FileStoragePtr file);

extern "C"
void CascadeClassifier_setMaxObjectSize(struct CascadeClassifierPtr ptr, struct SizeWrapper val);

extern "C"
struct SizeWrapper CascadeClassifier_getMaxObjectSize(struct CascadeClassifierPtr ptr);

extern "C"
void CascadeClassifier_setMinObjectSize(struct CascadeClassifierPtr ptr, struct SizeWrapper val);

extern "C"
struct SizeWrapper CascadeClassifier_getMinObjectSize(struct CascadeClassifierPtr ptr);

extern "C"
void CascadeClassifier_setScaleFactor(struct CascadeClassifierPtr ptr, double val);

extern "C"
double CascadeClassifier_getScaleFactor(struct CascadeClassifierPtr ptr);

extern "C"
void CascadeClassifier_setMinNeighbors(struct CascadeClassifierPtr ptr, int val);

extern "C"
int CascadeClassifier_getMinNeighbors(struct CascadeClassifierPtr ptr);

extern "C"
void CascadeClassifier_setFindLargestObject(struct CascadeClassifierPtr ptr, bool val);

extern "C"
bool CascadeClassifier_getFindLargestObject(struct CascadeClassifierPtr ptr);

extern "C"
void CascadeClassifier_setMaxNumObjects(struct CascadeClassifierPtr ptr, int val);

extern "C"
int CascadeClassifier_getMaxNumObjects(struct CascadeClassifierPtr ptr);

extern "C"
struct SizeWrapper CascadeClassifier_getClassifierSize(struct CascadeClassifierPtr ptr);

extern "C"
struct TensorWrapper CascadeClassifier_detectMultiScale(
        struct cutorchInfo info, struct CascadeClassifierPtr ptr,
        struct TensorWrapper image, struct TensorWrapper objects);

extern "C"
struct RectArray CascadeClassifier_convert(
        struct CascadeClassifierPtr ptr, struct TensorWrapper gpu_objects);
