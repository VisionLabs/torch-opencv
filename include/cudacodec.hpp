#include <CUDACommon.hpp>
#include <include/Classes.hpp>
#include <opencv2/cudacodec.hpp>

namespace cudacodec = cv::cudacodec;

extern "C"
struct cudacodec::EncoderParams EncoderParams_ctor_default();

extern "C"
struct cudacodec::EncoderParams EncoderParams_ctor(const char *configFile);

extern "C"
void EncoderParams_save(struct cudacodec::EncoderParams params, const char *configFile);

struct VideoWriterPtr {
    void *ptr;

    inline cudacodec::VideoWriter * operator->() { return static_cast<cudacodec::VideoWriter *>(ptr); }
    inline VideoWriterPtr(cudacodec::VideoWriter *ptr) { this->ptr = ptr; }
    inline cudacodec::VideoWriter & operator*() { return *static_cast<cudacodec::VideoWriter *>(this->ptr); }
};