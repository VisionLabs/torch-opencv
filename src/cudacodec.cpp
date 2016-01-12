#include <cudacodec.hpp>

extern "C"
struct cudacodec::EncoderParams EncoderParams_ctor_default()
{
    std::cout << cudacodec::SF_UYVY << std::endl;
    std::cout << cudacodec::SF_YUY2 << std::endl;
    std::cout << cudacodec::SF_YV12 << std::endl;
    std::cout << cudacodec::SF_NV12 << std::endl;
    std::cout << cudacodec::SF_IYUV << std::endl;
    std::cout << cudacodec::SF_IYUV << std::endl;
    std::cout << cudacodec::SF_GRAY << std::endl;
    return cudacodec::EncoderParams();
}

extern "C"
struct cudacodec::EncoderParams EncoderParams_ctor(const char *configFile)
{
    return cudacodec::EncoderParams(configFile);
}

extern "C"
void EncoderParams_save(struct cudacodec::EncoderParams params, const char *configFile)
{
    params.save(configFile);
}

extern "C"
struct VideoWriterPtr VideoWriter_ctor(
        const char *filename, struct SizeWrapper frameSize,
        double fps, struct cudacodec::EncoderParams params, int format)
{
    return rescueObjectFromPtr(cudacodec::createVideoWriter(
            filename, frameSize, fps, params, static_cast<cudacodec::SurfaceFormat>(format)));
}

extern "C"
void VideoWriter_dtor(struct VideoWriterPtr ptr)
{
    delete static_cast<cudacodec::VideoWriter *>(ptr.ptr);
}

extern "C"
void VideoWriter_write(struct VideoWriterPtr ptr, struct TensorWrapper frame, bool lastFrame)
{
    ptr->write(frame.toGpuMat(), lastFrame);
}

extern "C"
struct cudacodec::EncoderParams VideoWriter_getEncoderParams(struct VideoWriterPtr ptr)
{
    return ptr->getEncoderParams();
}