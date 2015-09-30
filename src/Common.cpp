#include <Common.hpp>

/***************** Tensor <=> Mat conversion *****************/

TensorWrapper::TensorWrapper(): tensorPtr(nullptr) {}

TensorWrapper::TensorWrapper(cv::Mat & mat) {

    this->typeCode = static_cast<char>(mat.depth());

    THByteTensor *outputPtr = new THByteTensor;

    // Build new storage on top of the Mat
    outputPtr->storage = THByteStorage_newWithData(
            mat.data,
            mat.step[0] * mat.rows
    );

    int sizeMultiplier;
    if (mat.channels() == 1) {
        outputPtr->nDimension = mat.dims;
        sizeMultiplier = cv::getElemSize(mat.depth());
    } else {
        outputPtr->nDimension = mat.dims + 1;
        sizeMultiplier = mat.elemSize1();
    }

    outputPtr->size = static_cast<long *>(THAlloc(sizeof(long) * outputPtr->nDimension));
    outputPtr->stride = static_cast<long *>(THAlloc(sizeof(long) * outputPtr->nDimension));

    if (mat.channels() > 1) {
        outputPtr->size[outputPtr->nDimension - 1] = mat.channels();
        outputPtr->stride[outputPtr->nDimension - 1] = 1; //cv::getElemSize(returnValue.typeCode);
    }

    for (int i = 0; i < mat.dims; ++i) {
        outputPtr->size[i] = mat.size[i];
        outputPtr->stride[i] = mat.step[i] / sizeMultiplier;
    }

    // Prevent OpenCV from deallocating Mat data
    mat.addref();

    outputPtr->refcount = 0;

    this->tensorPtr = outputPtr;
}

TensorWrapper::TensorWrapper(cv::Mat && mat) {
    // invokes TensorWrapper(cv::Mat & mat)
    new (this) TensorWrapper(mat);
}

cv::Mat TensorWrapper::toMat() {

    THByteTensor *tensorPtr = static_cast<THByteTensor *>(this->tensorPtr);

    int numberOfDims = tensorPtr->nDimension;
    // THTensor stores its dimensions sizes under long *.
    // In a constructor for cv::Mat, we need const int *.
    // We can't guarantee int and long to be equal.
    // So we somehow need to static_cast THTensor sizes.
    // TODO: we should somehow get rid of array allocation

    std::array<int, 3> size;
    std::copy(tensorPtr->size, tensorPtr->size + tensorPtr->nDimension, size.begin());

    // Same thing for stride values.
    std::array<size_t, 3> stride;
    std::copy(tensorPtr->stride, tensorPtr->stride + tensorPtr->nDimension, stride.begin());

    int depth = this->typeCode;

    // Determine the number of channels.
    int numChannels;
    // cv::Mat() takes stride values in bytes, so we have to multiply by the element size:
    size_t sizeMultiplier = cv::getElemSize(depth);

    if (tensorPtr->nDimension <= 2) {
        // If such tensor is passed, assume that it is single-channel:
        numChannels = 1;
    } else {
        // Otherwise depend on the 3rd dimension:
        numChannels = tensorPtr->size[2];
        numberOfDims = 2;
    }

    std::for_each(stride.begin(),
                  stride.end(),
                  [sizeMultiplier] (size_t & x) { x *= sizeMultiplier; });

    return cv::Mat(
            numberOfDims,
            size.data(),
            CV_MAKE_TYPE(depth, numChannels),
            tensorPtr->storage->data,
            stride.data()
    );
}

MultipleTensorWrapper::MultipleTensorWrapper(): tensors(nullptr) {}

MultipleTensorWrapper::MultipleTensorWrapper(std::vector<cv::Mat> & matList):
        tensors(static_cast<TensorWrapper *>(malloc(matList.size() * sizeof(TensorWrapper)))),
        size(matList.size())
{
    for (size_t i = 0; i < matList.size(); ++i) {
        // invoke the constructor, memory is already allocated
        new (tensors + i) TensorWrapper(matList[i]);
    }
}

std::vector<cv::Mat> MultipleTensorWrapper::toMat() {
    std::vector<cv::Mat> retval(this->size);
    for (int i = 0; i < this->size; ++i) {
        retval[i] = this->tensors[i].toMat();
    }
    return retval;
}

// Kill "destination" and assign "source" data to it.
// "destination" is always supposed to be an empty Tensor
extern "C"
void transfer_tensor(void *destination, void *source) {
    THByteTensor * s = static_cast<THByteTensor *>(source);
    THByteTensor * d = static_cast<THByteTensor *>(destination);

    if (d->storage)
        THFree(d->storage);
    if (d->size)
        THFree(d->size);
    if (d->stride)
        THFree(d->stride);

    d->storage = s->storage;
    d->size = s->size;
    d->stride = s->stride;
    d->nDimension = s->nDimension;
    ++d->refcount;
}

/***************** Wrappers for small classes *****************/

cv::TermCriteria TermCriteriaWrapper::toCV() {
    return cv::TermCriteria(type, maxCount, epsilon);
}

cv::Scalar ScalarWrapper::toCV() {
    return cv::Scalar(v0, v1, v2, v3);
}

extern "C"
cv::Algorithm *createAlgorithm() {
    return new cv::Algorithm();
}

extern "C"
void destroyAlgorithm(cv::Algorithm *ptr) {
    delete ptr;
}