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
void EncoderParams_save(struct cudacodec::EncoderParams params, const char *configFile)
{
    params.save(configFile);
}