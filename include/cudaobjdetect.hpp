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

extern "C"
struct HOGPtr HOG_ctorCuda(
        struct SizeWrapper win_size, struct SizeWrapper block_size,
        struct SizeWrapper block_stride, struct SizeWrapper cell_size, int nbins);

extern "C"
void HOG_setWinSigmaCuda(struct HOGPtr ptr, double val);

extern "C"
double HOG_getWinSigmaCuda(struct HOGPtr ptr);

extern "C"
void HOG_setL2HysThresholdCuda(struct HOGPtr ptr, double val);

extern "C"
double HOG_getL2HysThresholdCuda(struct HOGPtr ptr);

extern "C"
void HOG_setGammaCorrectionCuda(struct HOGPtr ptr, bool val);

extern "C"
bool HOG_getGammaCorrectionCuda(struct HOGPtr ptr);

extern "C"
void HOG_setNumLevelsCuda(struct HOGPtr ptr, int val);

extern "C"
int HOG_getNumLevelsCuda(struct HOGPtr ptr);

extern "C"
void HOG_setHitThresholdCuda(struct HOGPtr ptr, double val);

extern "C"
double HOG_getHitThresholdCuda(struct HOGPtr ptr);

extern "C"
void HOG_setWinStrideCuda(struct HOGPtr ptr, struct SizeWrapper val);

extern "C"
struct SizeWrapper HOG_getWinStrideCuda(struct HOGPtr ptr);

extern "C"
void HOG_setScaleFactorCuda(struct HOGPtr ptr, double val);

extern "C"
double HOG_getScaleFactorCuda(struct HOGPtr ptr);

extern "C"
void HOG_setGroupThresholdCuda(struct HOGPtr ptr, int val);

extern "C"
int HOG_getGroupThresholdCuda(struct HOGPtr ptr);

extern "C"
void HOG_setDescriptorFormatCuda(struct HOGPtr ptr, int val);

extern "C"
int HOG_getDescriptorFormatCuda(struct HOGPtr ptr);

extern "C"
size_t HOG_getDescriptorSizeCuda(struct HOGPtr ptr);

extern "C"
size_t HOG_getBlockHistogramSizeCuda(struct HOGPtr ptr);

extern "C"
void HOG_setSVMDetectorCuda(struct HOGPtr ptr, struct TensorWrapper val);

extern "C"
struct TensorWrapper HOG_getDefaultPeopleDetectorCuda(struct HOGPtr ptr);

extern "C"
struct TensorPlusPointArray HOG_detectCuda(
        struct cutorchInfo info, struct HOGPtr ptr, struct TensorWrapper img);

extern "C"
struct TensorPlusRectArray HOG_detectMultiScaleCuda(
        struct cutorchInfo info, struct HOGPtr ptr, struct TensorWrapper img);

extern "C"
struct TensorWrapper HOG_computeCuda(
        struct cutorchInfo info, struct HOGPtr ptr, struct TensorWrapper img,
        struct TensorWrapper descriptors);

extern "C"
struct CascadeClassifierPtr CascadeClassifier_ctor_filenameCuda(const char *filename);

extern "C"
struct CascadeClassifierPtr CascadeClassifier_ctor_fileCuda(struct FileStoragePtr file);

extern "C"
void CascadeClassifier_setMaxObjectSizeCuda(struct CascadeClassifierPtr ptr, struct SizeWrapper val);

extern "C"
struct SizeWrapper CascadeClassifier_getMaxObjectSizeCuda(struct CascadeClassifierPtr ptr);

extern "C"
void CascadeClassifier_setMinObjectSizeCuda(struct CascadeClassifierPtr ptr, struct SizeWrapper val);

extern "C"
struct SizeWrapper CascadeClassifier_getMinObjectSizeCuda(struct CascadeClassifierPtr ptr);

extern "C"
void CascadeClassifier_setScaleFactorCuda(struct CascadeClassifierPtr ptr, double val);

extern "C"
double CascadeClassifier_getScaleFactorCuda(struct CascadeClassifierPtr ptr);

extern "C"
void CascadeClassifier_setMinNeighborsCuda(struct CascadeClassifierPtr ptr, int val);

extern "C"
int CascadeClassifier_getMinNeighborsCuda(struct CascadeClassifierPtr ptr);

extern "C"
void CascadeClassifier_setFindLargestObjectCuda(struct CascadeClassifierPtr ptr, bool val);

extern "C"
bool CascadeClassifier_getFindLargestObjectCuda(struct CascadeClassifierPtr ptr);

extern "C"
void CascadeClassifier_setMaxNumObjectsCuda(struct CascadeClassifierPtr ptr, int val);

extern "C"
int CascadeClassifier_getMaxNumObjectsCuda(struct CascadeClassifierPtr ptr);

extern "C"
struct SizeWrapper CascadeClassifier_getClassifierSizeCuda(struct CascadeClassifierPtr ptr);

extern "C"
struct TensorWrapper CascadeClassifier_detectMultiScaleCuda(
        struct cutorchInfo info, struct CascadeClassifierPtr ptr,
        struct TensorWrapper image, struct TensorWrapper objects);

extern "C"
struct RectArray CascadeClassifier_convertCuda(
        struct CascadeClassifierPtr ptr, struct TensorWrapper gpu_objects);
