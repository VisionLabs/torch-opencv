#include <CUDACommon.hpp>
#include <array>

cuda::GpuMat TensorWrapper::toGpuMat() {

    if (this->tensorPtr == nullptr) {
        return cuda::GpuMat();
    }

    THCudaTensor *tensorPtr = static_cast<THCudaTensor *>(this->tensorPtr);

    int numChannels = 1;
    if (tensorPtr->nDimension == 3) {
        numChannels = tensorPtr->size[2];
    }

    return cuda::GpuMat(
            tensorPtr->size[0],
            tensorPtr->size[1],
            CV_32FC1,
            tensorPtr->storage->data,
            tensorPtr->stride[0] * sizeof(float)
    );
}
