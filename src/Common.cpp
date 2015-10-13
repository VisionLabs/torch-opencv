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

TensorWrapper::operator cv::Mat() {

    if (this->tensorPtr == nullptr) {
        return cv::Mat();
    }

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

MultipleTensorWrapper::MultipleTensorWrapper(short size):
        size(size),
        tensors(static_cast<TensorWrapper *>(malloc(size * sizeof(TensorWrapper)))) {}

MultipleTensorWrapper::MultipleTensorWrapper(std::vector<cv::Mat> & matList):
        tensors(static_cast<TensorWrapper *>(malloc(matList.size() * sizeof(TensorWrapper)))),
        size(matList.size())
{
    for (size_t i = 0; i < matList.size(); ++i) {
        // invoke the constructor, memory is already allocated
        new (tensors + i) TensorWrapper(matList[i]);
    }
}

MultipleTensorWrapper::operator std::vector<cv::Mat>() {
    std::vector<cv::Mat> retval(this->size);
    for (int i = 0; i < this->size; ++i) {
        retval[i] = this->tensors[i];
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

MomentsWrapper::MomentsWrapper(cv::Moments other) {
    m00 = other.m00; m01 = other.m01; m02 = other.m02; m03 = other.m03; m10 = other.m10;
    m11 = other.m11; m12 = other.m12; m20 = other.m20; m21 = other.m21; m30 = other.m30;
    mu20 = other.mu20; mu11 = other.mu11; mu02 = other.mu02; mu30 = other.mu30;
    mu21 = other.mu21; mu12 = other.mu12; mu03 = other.mu03;
    nu20 = other.nu20; nu11 = other.nu11; nu02 = other.nu02; nu30 = other.nu30;
    nu21 = other.nu21; nu12 = other.nu12; nu03 = other.nu03;
}

/***************** Helper wrappers for [OpenCV class + some primitive] *****************/

RectWrapper & RectWrapper::operator=(cv::Rect & other) {
    this->x = other.x;
    this->y = other.y;
    this->width = other.width;
    this->height = other.height;
    return *this;
}