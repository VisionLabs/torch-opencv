#include <CUDACommon.hpp>

cuda::GpuMat TensorWrapper::toGpuMat() {

    if (this->tensorPtr == nullptr) {
        return cuda::GpuMat();
    }

    THCudaTensor *tensorPtr = static_cast<THCudaTensor *>(this->tensorPtr);

    assert(this->typeCode == CV_CUDA);
    assert(tensorPtr->nDimension <= 3);

    int numChannels = 1;
    if (tensorPtr->nDimension == 3) {
        numChannels = tensorPtr->size[2];
    }

    return cuda::GpuMat(
            tensorPtr->size[0],
            tensorPtr->size[1],
            CV_32FC(numChannels),
            tensorPtr->storage->data,
            tensorPtr->stride[0] * sizeof(float)
    );
}

TensorWrapper::TensorWrapper(cuda::GpuMat & mat, THCState *state) {

    if (mat.empty()) {
        this->tensorPtr = nullptr;
        return;
    }

    assert(mat.depth() == CV_32F);
    this->typeCode = 66;

    THCudaTensor *outputPtr = new THCudaTensor;

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
        outputPtr->stride[2] = 1; //cv::getElemSize(returnValue.typeCode);
    }

    outputPtr->size[0] = mat.rows;
    outputPtr->size[1] = mat.cols;

    outputPtr->stride[0] = mat.step / sizeMultiplier;
    outputPtr->stride[1] = 1;

    outputPtr->storageOffset = 0;

    // Make OpenCV treat underlying data as user-allocated
    mat.refcount = nullptr;

    outputPtr->refcount = 0;

    this->tensorPtr = outputPtr;
}

TensorWrapper::TensorWrapper(cuda::GpuMat && mat, THCState *state) {
    // invokes TensorWrapper(cuda::GpuMat & mat)
    new (this) TensorWrapper(mat, state);
}

// Kill "destination" and assign "source" data to it.
// "destination" is always supposed to be an empty Tensor
extern "C"
void transfer_tensor_CUDA(THCState *state, THCudaTensor *dst, THCudaTensor *src) {
    if (dst->storage)
        THCudaStorage_free(state, dst->storage);
    if (dst->size)
        THFree(dst->size);
    if (dst->stride)
        THFree(dst->stride);

    dst->storage = src->storage;
    dst->size = src->size;
    dst->stride = src->stride;
    dst->nDimension = src->nDimension;
    ++dst->refcount;
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