#include <videoio.hpp>

/****************** Classes ******************/

// VideoCapture

extern "C"
struct VideoCapturePtr VideoCapture_ctor_default()
{
    return new cv::VideoCapture();
}

extern "C"
struct VideoCapturePtr VideoCapture_ctor_device(int device)
{
    return new cv::VideoCapture(device);
}

extern "C"
struct VideoCapturePtr VideoCapture_ctor_filename(const char *filename)
{
    return new cv::VideoCapture(filename);
}

extern "C"
void VideoCapture_dtor(VideoCapturePtr ptr)
{
    delete static_cast<cv::VideoCapture *>(ptr.ptr);
}

extern "C"
bool VideoCapture_open(VideoCapturePtr ptr, int device)
{
    return ptr->open(device);
}

extern "C"
bool VideoCapture_isOpened(VideoCapturePtr ptr)
{
    return ptr->isOpened();
}

extern "C"
void VideoCapture_release(VideoCapturePtr ptr)
{
    ptr->release();
}

extern "C"
bool VideoCapture_grab(VideoCapturePtr ptr)
{
    return ptr->grab();
}

extern "C"
struct TensorPlusBool VideoCapture_retrieve(
        VideoCapturePtr ptr, struct TensorWrapper image, int flag)
{
    TensorPlusBool retval;
    MatT result = image.toMatT();
    retval.val = ptr->retrieve(result, flag);
    new (&retval.tensor) TensorWrapper(result);
    return retval;
}

extern "C"
struct TensorPlusBool VideoCapture_read(
        VideoCapturePtr ptr, struct TensorWrapper image)
{
    TensorPlusBool retval;
    MatT result = image.toMatT();
    retval.val = ptr->read(result);
    new (&retval.tensor) TensorWrapper(result);
    return retval;
}

extern "C"
bool VideoCapture_set(VideoCapturePtr ptr, int propId, double value)
{
    return ptr->set(propId, value);
}

extern "C"
double VideoCapture_get(VideoCapturePtr ptr, int propId)
{
    return ptr->get(propId);
}

// VideoWriter

extern "C"
struct VideoWriterPtr VideoWriter_ctor_default()
{
    return new cv::VideoWriter();
}

extern "C"
struct VideoWriterPtr VideoWriter_ctor(
        const char *filename, int fourcc, double fps, struct SizeWrapper frameSize, bool isColor)
{
    return new cv::VideoWriter(filename, fourcc, fps, frameSize, isColor);
}

extern "C"
void VideoWriter_dtor(struct VideoWriterPtr ptr)
{
    delete static_cast<cv::VideoWriter *>(ptr.ptr);
}

extern "C"
bool VideoWriter_open(struct VideoWriterPtr ptr, const char *filename, int fourcc,
                      double fps, struct SizeWrapper frameSize, bool isColor)
{
    return ptr->open(filename, fourcc, fps, frameSize, isColor);
}

extern "C"
bool VideoWriter_isOpened(struct VideoWriterPtr ptr)
{
    return ptr->isOpened();
}

extern "C"
void VideoWriter_release(struct VideoWriterPtr ptr)
{
    ptr->release();
}

extern "C"
void VideoWriter_write(struct VideoWriterPtr ptr, struct TensorWrapper image)
{
    ptr->write(image.toMat());
}

extern "C"
bool VideoWriter_set(VideoWriterPtr ptr, int propId, double value)
{
    return ptr->set(propId, value);
}

extern "C"
double VideoWriter_get(VideoWriterPtr ptr, int propId)
{
    return ptr->get(propId);
}

extern "C"
int VideoWriter_fourcc(char c1, char c2, char c3, char c4)
{
    return cv::VideoWriter::fourcc(c1, c2, c3, c4);
}
