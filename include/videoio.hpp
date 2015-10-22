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

struct VideoWriterPtr {
    void *ptr;

    inline cv::VideoWriter * operator->() { return static_cast<cv::VideoWriter *>(ptr); }
    inline VideoWriterPtr(cv::VideoWriter *ptr) { this->ptr = ptr; }
    inline cv::VideoWriter & operator*() { return *static_cast<cv::VideoWriter *>(this->ptr); }
};

extern "C"
struct VideoWriterPtr VideoWriter_ctor_default();

extern "C"
struct VideoWriterPtr VideoWriter_ctor(
        const char *filename, int fourcc, double fps, struct SizeWrapper frameSize, bool isColor);

extern "C"
void VideoWriter_dtor(struct VideoWriterPtr ptr);

extern "C"
bool VideoWriter_open(struct VideoWriterPtr ptr, const char *filename, int fourcc,
                      double fps, struct SizeWrapper frameSize, bool isColor);

extern "C"
bool VideoWriter_isOpened(struct VideoWriterPtr ptr);

extern "C"
void VideoWriter_release(struct VideoWriterPtr ptr);

extern "C"
void VideoWriter_write(struct VideoWriterPtr ptr, struct TensorWrapper image);

extern "C"
bool VideoWriter_set(VideoWriterPtr ptr, int propId, double value);

extern "C"
double VideoWriter_get(VideoWriterPtr ptr, int propId);

extern "C"
int VideoWriter_fourcc(char c1, char c2, char c3, char c4);