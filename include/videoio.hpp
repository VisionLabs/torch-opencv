#include <Common.hpp>
#include <opencv2/videoio.hpp>

struct VideoCapturePtr {
    void *ptr;

    inline cv::VideoCapture * operator->() { return static_cast<cv::VideoCapture *>(ptr); }
    inline VideoCapturePtr(cv::VideoCapture *ptr) { this->ptr = ptr; }
    inline cv::VideoCapture & operator*() { return *static_cast<cv::VideoCapture *>(this->ptr); }
};

extern "C"
struct VideoCapturePtr VideoCapture_ctor_default();

extern "C"
struct VideoCapturePtr VideoCapture_ctor_device(int device);

extern "C"
struct VideoCapturePtr VideoCapture_ctor_filename(const char *filename);

extern "C"
void VideoCapture_dtor(VideoCapturePtr ptr);

extern "C"
bool VideoCapture_open(VideoCapturePtr ptr, int device);

extern "C"
bool VideoCapture_isOpened(VideoCapturePtr ptr);

extern "C"
void VideoCapture_release(VideoCapturePtr ptr);

extern "C"
bool VideoCapture_grab(VideoCapturePtr ptr);

extern "C"
struct TensorPlusBool VideoCapture_retrieve(
        VideoCapturePtr ptr, struct TensorWrapper image, int flag);

extern "C"
struct TensorPlusBool VideoCapture_read(
        VideoCapturePtr ptr, struct TensorWrapper image);

extern "C"
bool VideoCapture_set(VideoCapturePtr ptr, int propId, double value);

extern "C"
double VideoCapture_get(VideoCapturePtr ptr, int propId);