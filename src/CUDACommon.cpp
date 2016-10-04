#include <CUDACommon.hpp>

GpuMatT::GpuMatT(cuda::GpuMat & mat) {
    this->mat = mat;
    this->tensor = nullptr;
}

GpuMatT::GpuMatT(cuda::GpuMat && mat) {
    new (this) GpuMatT(mat);
}

GpuMatT::GpuMatT() {
    this->tensor = nullptr;
}

TensorWrapper::TensorWrapper(GpuMatT & matT, THCState *state) {

    if (matT.tensor != nullptr) {
        // Mat is already constructed on another Tensor, so return that
        this->tensorPtr = reinterpret_cast<THByteTensor *>(matT.tensor);
        this->definedInLua = true;
        this->typeCode = static_cast<char>(matT.mat.depth());
        THAtomicIncrementRef(&this->tensorPtr->storage->refcount);
    } else {
        new (this) TensorWrapper(matT.mat, state);
    }
}

TensorWrapper::TensorWrapper(GpuMatT && mat, THCState *state) {
    new (this) TensorWrapper(mat, state);
}

cuda::GpuMat TensorWrapper::toGpuMat(int depth) {

    if (this->tensorPtr == nullptr or this->tensorPtr->nDimension == 0) {
        return cuda::GpuMat();
    }

    THCudaTensor *tensorPtr = reinterpret_cast<THCudaTensor *>(this->tensorPtr);

    assert(tensorPtr->nDimension <= 3);

    int numChannels = 1;
    if (tensorPtr->nDimension == 3) {
        numChannels = tensorPtr->size[2];
    }

    return cuda::GpuMat(
            tensorPtr->size[0],
            tensorPtr->size[1],
            (depth == -1 ? CV_32FC(numChannels) : CV_MAKE_TYPE(depth, numChannels)),
            tensorPtr->storage->data + tensorPtr->storageOffset * cv::getElemSize(CV_32F),
            tensorPtr->stride[0] * sizeof(float)
    );
}

TensorWrapper::TensorWrapper(cuda::GpuMat & mat, THCState *state) {

    this->definedInLua = false;

    if (mat.empty()) {
        this->typeCode = CV_CUDA;
        this->tensorPtr = nullptr;
        return;
    }

    this->typeCode = CV_CUDA;

    THCudaTensor *outputPtr = THCudaTensor_new(state);

    // Build new storage on top of the Mat
    outputPtr->storage = THCudaStorage_newWithData(
            state,
            reinterpret_cast<float *>(mat.data),
            mat.step * mat.rows * mat.channels() / cv::getElemSize(mat.depth())
    );

    int sizeMultiplier;
    if (mat.channels() == 1) {
        outputPtr->nDimension = 2;
        sizeMultiplier = cv::getElemSize(mat.depth());
    } else {
        outputPtr->nDimension = 3;
        sizeMultiplier = mat.elemSize1();
    }

    outputPtr->size = static_cast<long *>(THAlloc(sizeof(long) * outputPtr->nDimension));
    outputPtr->stride = static_cast<long *>(THAlloc(sizeof(long) * outputPtr->nDimension));

    if (mat.channels() > 1) {
        outputPtr->size[2] = mat.channels();
        outputPtr->stride[2] = 1;
    }

    outputPtr->size[0] = mat.rows;
    outputPtr->size[1] = mat.cols;

    outputPtr->stride[0] = mat.step / sizeMultiplier;
    outputPtr->stride[1] = mat.channels();

    outputPtr->storageOffset = 0;

    // Make OpenCV treat underlying data as user-allocated
    mat.refcount = nullptr;

    this->tensorPtr = reinterpret_cast<THByteTensor *>(outputPtr);
}

TensorWrapper::TensorWrapper(cuda::GpuMat && mat, THCState *state) {
    // invokes TensorWrapper(cuda::GpuMat & mat)
    new (this) TensorWrapper(mat, state);
}

// Kill "destination" and assign "source" data to it.
// "destination" is always supposed to be an empty CudaTensor
extern "C"
void transfer_tensor_CUDA(THCState *state, THCudaTensor *dst, struct TensorWrapper srcWrapper) {

    THCudaTensor *src = reinterpret_cast<THCudaTensor *>(srcWrapper.tensorPtr);

    dst->nDimension = src->nDimension;
    dst->refcount = src->refcount;

    dst->storage = src->storage;

    if (!srcWrapper.definedInLua) {
        // Don't let Torch deallocate size and stride arrays
        dst->size = src->size;
        dst->stride = src->stride;
        src->size = nullptr;
        src->stride = nullptr;
        THAtomicIncrementRef(&src->storage->refcount);
        THCudaTensor_free(state, src);
    } else {
        dst->size   = static_cast<long *>(THAlloc(sizeof(long) * dst->nDimension));
        dst->stride = static_cast<long *>(THAlloc(sizeof(long) * dst->nDimension));
        memcpy(dst->size,   src->size,   src->nDimension * sizeof(long));
        memcpy(dst->stride, src->stride, src->nDimension * sizeof(long));
    }
}

TensorArray::TensorArray(std::vector<cuda::GpuMat> & matList, THCState *state):
        tensors(static_cast<TensorWrapper *>(malloc(matList.size() * sizeof(TensorWrapper)))),
        size(matList.size())
{
    for (size_t i = 0; i < matList.size(); ++i) {
        // invoke the constructor, memory is already allocated
        new (tensors + i) TensorWrapper(matList[i], state);
    }
}

std::vector<cv::cuda::GpuMat> TensorArray::toGpuMatList() {
    std::vector<cuda::GpuMat> retval(this->size);
    for (int i = 0; i < this->size; ++i) {
        // TODO: avoid temporary object
        retval[i] = this->tensors[i].toGpuMat();
    }
    return retval;
}

/************************ Fake OpenCV/CUDA classes *************************/

FakeDefaultDeviceInitializer initializer;

unsigned char* FakeMemoryStack::requestMemory(size_t size)
{
    const size_t freeMem = dataend - tip;

    if (size > freeMem)
        return 0;

    unsigned char* ptr = tip;

    tip += size;

#if !defined(NDEBUG)
    allocations.push_back(size);
#endif

    return ptr;
}

void FakeMemoryStack::returnMemory(unsigned char* ptr)
{
    CV_DbgAssert( ptr >= datastart && ptr < dataend );

#if !defined(NDEBUG)
    const size_t allocSize = tip - ptr;
    CV_Assert( allocSize == allocations.back() );
    allocations.pop_back();
#endif

    tip = ptr;
}

void FakeMemoryPool::initilizeImpl()
{
    const size_t totalSize = stackSize_ * stackCount_;

    if (totalSize > 0)
    {
        cudaError_t err = cudaMalloc(&mem_, totalSize);
        if (err != cudaSuccess)
            return;

        stacks_.resize(stackCount_);

        unsigned char* ptr = mem_;

        for (int i = 0; i < stackCount_; ++i)
        {
            stacks_[i].datastart = ptr;
            stacks_[i].dataend = ptr + stackSize_;
            stacks_[i].tip = ptr;
            stacks_[i].isFree = true;
            stacks_[i].pool = this;

            ptr += stackSize_;
        }

        initialized_ = true;
    }
}

FakeMemoryStack* FakeMemoryPool::getFreeMemStack()
{
    cv::AutoLock lock(mtx_);

    if (!initialized_)
        initilizeImpl();

    if (!mem_)
        return 0;

    for (int i = 0; i < stackCount_; ++i)
    {
        if (stacks_[i].isFree)
        {
            stacks_[i].isFree = false;
            return &stacks_[i];
        }
    }

    return 0;
}

FakeDefaultDeviceInitializer::FakeDefaultDeviceInitializer() {}

FakeDefaultDeviceInitializer::~FakeDefaultDeviceInitializer() {
    streams_.clear();

    for (size_t i = 0; i < pools_.size(); ++i)
    {
        cudaSetDevice(static_cast<int>(i));
        pools_[i].release();
    }

    pools_.clear();
}

FakeStream & FakeDefaultDeviceInitializer::getNullStream(int deviceId) {
    cv::AutoLock lock(streams_mtx_);

    if (streams_.empty())
    {
        int deviceCount = cuda::getCudaEnabledDeviceCount();

        if (deviceCount > 0)
            streams_.resize(deviceCount);
    }

    CV_DbgAssert( deviceId >= 0 && deviceId < static_cast<int>(streams_.size()) );

    if (streams_[deviceId].empty())
    {
        cudaStream_t stream = NULL;
        cv::Ptr<FakeStreamImpl> impl = cv::makePtr<FakeStreamImpl>(stream);
        streams_[deviceId] = cv::Ptr<FakeStream>(new FakeStream(impl));
    }

    return *streams_[deviceId];
}

FakeMemoryPool* FakeDefaultDeviceInitializer::getMemoryPool(int deviceId) {
    cv::AutoLock lock(pools_mtx_);

    if (pools_.empty())
    {
        int deviceCount = cuda::getCudaEnabledDeviceCount();

        if (deviceCount > 0)
            pools_.resize(deviceCount);
    }

    CV_DbgAssert( deviceId >= 0 && deviceId < static_cast<int>(pools_.size()) );

    return &pools_[deviceId];
}

FakeStackAllocator::FakeStackAllocator(cudaStream_t stream) : stream_(stream), memStack_(0) {
    const int deviceId = cuda::getDevice();
    memStack_ = initializer.getMemoryPool(deviceId)->getFreeMemStack();
    cuda::DeviceInfo devInfo(deviceId);
    alignment_ = devInfo.textureAlignment();
}

bool FakeStackAllocator::allocate(cuda::GpuMat* mat, int rows, int cols, size_t elemSize) {
    if (memStack_ == 0)
        return false;

    size_t pitch, memSize;

    if (rows > 1 && cols > 1)
    {
        pitch = alignUp(cols * elemSize, alignment_);
        memSize = pitch * rows;
    }
    else
    {
        // Single row or single column must be continuous
        pitch = elemSize * cols;
        memSize = alignUp(elemSize * cols * rows, 64);
    }

    unsigned char* ptr = memStack_->requestMemory(memSize);

    if (ptr == 0)
        return false;

    mat->data = ptr;
    mat->step = pitch;
    mat->refcount = (int*) cv::fastMalloc(sizeof(int));

    return true;
}

void FakeStackAllocator::free(cuda::GpuMat* mat) {
    if (memStack_ == 0)
        return;

    memStack_->returnMemory(mat->datastart);
    cv::fastFree(mat->refcount);
}

cuda::Stream & prepareStream(cutorchInfo info) {
    cuda::setDevice(info.deviceID - 1);
    fakeStream.impl_ = cv::makePtr<FakeStreamImpl>(THCState_getCurrentStream(info.state));
    return *reinterpret_cast<cuda::Stream *>(&fakeStream);
}