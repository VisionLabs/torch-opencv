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
    if (image.isNull()) {
        cv::Mat result;
        retval.val = ptr->retrieve(result, flag);
        new (&retval.tensor) TensorWrapper(result);
    } else {
        retval.val = ptr->retrieve(image.toMat(), flag);
        retval.tensor.tensorPtr = nullptr;
    }
    return retval;
}

extern "C"
struct TensorPlusBool VideoCapture_read(
        VideoCapturePtr ptr, struct TensorWrapper image)
{
    TensorPlusBool retval;
    if (image.isNull()) {
        cv::Mat result;
        retval.val = ptr->read(result);
        new (&retval.tensor) TensorWrapper(result);
    } else {
        retval.val = ptr->read(image.toMat());
        retval.tensor.tensorPtr = nullptr;
    }
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