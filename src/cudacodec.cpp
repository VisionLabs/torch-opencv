#include <cudacodec.hpp>

extern "C"
struct cudacodec::EncoderParams EncoderParams_ctor_default()
{
    return cudacodec::EncoderParams();
}

extern "C"
struct cudacodec::EncoderParams EncoderParams_ctor(const char *configFile)
{
    return cudacodec::EncoderParams(configFile);
}

extern "C"
void EncoderParams_saveCuda(struct cudacodec::EncoderParams params, const char *configFile)
{
    params.save(configFile);
}

extern "C"
struct VideoWriterPtr VideoWriter_ctorCuda(
        const char *filename, struct SizeWrapper frameSize,
        double fps, struct cudacodec::EncoderParams params, int format)
{
    return rescueObjectFromPtr(cudacodec::createVideoWriter(
            cv::String(filename), frameSize, fps, params,
            static_cast<cudacodec::SurfaceFormat>(format)));
}

extern "C"
void VideoWriter_dtorCuda(struct VideoWriterPtr ptr)
{
    delete static_cast<cudacodec::VideoWriter *>(ptr.ptr);
}

extern "C"
void VideoWriter_writeCuda(struct VideoWriterPtr ptr, struct TensorWrapper frame, bool lastFrame)
{
    ptr->write(frame.toGpuMat(), lastFrame);
}

extern "C"
struct cudacodec::EncoderParams VideoWriter_getEncoderParams(struct VideoWriterPtr ptr)
{
    return ptr->getEncoderParams();
}

extern "C"
struct VideoReaderPtr VideoReader_ctorCuda(const char *filename)
{
    return rescueObjectFromPtr(cudacodec::createVideoReader(cv::String(filename)));
}

extern "C"
void VideoReader_dtorCuda(struct VideoReaderPtr ptr)
{
    delete static_cast<cudacodec::VideoReader *>(ptr.ptr);
}

extern "C"
struct TensorWrapper VideoReader_nextFrameCuda(
        struct cutorchInfo info, struct VideoReaderPtr ptr, struct TensorWrapper frame)
{
    if (frame.isNull()) {
        cuda::GpuMat retval;
        ptr->nextFrame(retval);
        return TensorWrapper(retval, info.state);
    } else {
        ptr->nextFrame(frame.toGpuMat());
        return frame;
    }
}

extern "C"
struct cudacodec::FormatInfo VideoReader_format(struct VideoReaderPtr ptr)
{
    return ptr->format();
}
