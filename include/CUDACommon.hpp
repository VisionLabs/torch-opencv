#include <Common.hpp>
#include <opencv2/cudalegacy/NCV.hpp>

namespace cuda = cv::cuda;

#define CV_CUDA 66

cuda::GpuMat TensorWrapper::toGpuMat();

TensorWrapper::TensorWrapper(cuda::GpuMat & mat, THCState *state);
TensorWrapper::TensorWrapper(cuda::GpuMat && mat, THCState *state);
TensorArray::TensorArray(std::vector<cuda::GpuMat> & matList, THCState *state);

// Kill "destination" and assign "source" data to it.
// "destination" is always supposed to be an empty Tensor
extern "C"
void transfer_tensor_CUDA(THCState *state, THCudaTensor *dst, THCudaTensor *src);

struct cutorchInfo {
    int deviceID;
    THCState *state;
};

class FakeMemoryPool;
class FakeMemoryStack;
class FakeStackAllocator;
class FakeStreamImpl;
class FakeStream;

class FakeMemoryStack {
public:
    unsigned char* datastart;
    unsigned char* dataend;
    unsigned char* tip;

    bool isFree;
    FakeMemoryPool* pool;

#if !defined(NDEBUG)
    std::vector<size_t> allocations;
#endif

    unsigned char* requestMemory(size_t size);
    void returnMemory(unsigned char* ptr);
};

class FakeMemoryPool {
public:
    cv::Mutex mtx_;

    bool initialized_;
    size_t stackSize_;
    int stackCount_;

    unsigned char* mem_;

    std::vector<FakeMemoryStack> stacks_;

    void initilizeImpl();
    void release();
    FakeMemoryStack* getFreeMemStack();
};

void FakeMemoryPool::release() {
    if (mem_) {
#if !defined(NDEBUG)
        for (int i = 0; i < stackCount_; ++i) {
            CV_DbgAssert( stacks_[i].isFree );
            CV_DbgAssert( stacks_[i].tip == stacks_[i].datastart );
        }
#endif

        cudaFree(mem_);
        mem_ = 0;
        initialized_ = false;
    }
}

class FakeStream {
public:
    cv::Ptr<FakeStreamImpl> impl_;

    FakeStream() {}
    FakeStream(cv::Ptr<FakeStreamImpl> & impl): impl_(impl) {}
};

class FakeDefaultDeviceInitializer {
public:
    FakeDefaultDeviceInitializer();
    ~FakeDefaultDeviceInitializer();

    FakeStream & getNullStream(int deviceId);
    FakeMemoryPool* getMemoryPool(int deviceId);

private:
    void initStreams();
    void initPools();

    std::vector<cv::Ptr<FakeStream>> streams_;
    cv::Mutex streams_mtx_;

    std::vector<FakeMemoryPool> pools_;
    cv::Mutex pools_mtx_;
};

class FakeStackAllocator : public cuda::GpuMat::Allocator {
public:
    cudaStream_t stream_;
    FakeMemoryStack *memStack_;
    size_t alignment_;

    FakeStackAllocator(cudaStream_t stream);
    bool allocate(cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
    void free(cuda::GpuMat* mat);
};

class FakeStreamImpl {
public:
    cudaStream_t stream;
    bool ownStream;

    cv::Ptr<FakeStackAllocator> stackAllocator;

    FakeStreamImpl(cudaStream_t stream_) : stream(stream_), ownStream(false) {
        stackAllocator = cv::makePtr<FakeStackAllocator>(stream);
    }
};