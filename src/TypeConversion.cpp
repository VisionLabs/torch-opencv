#include <TypeConversion.hpp>

cv::Mat tensorToMat(TensorWrapper tensor) {

    THByteTensor *tensorPtr = static_cast<THByteTensor *>(tensor.tensorPtr);

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

    int depth = tensor.tensorType;

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

void matToTensor(cv::Mat & mat, TensorWrapper output) {

    // This is awful: we first allocate this space in Lua,
    // then deallocate it here, and then allocate again.
    // But it works though.
    // TODO: avoid extra allocations!

    outputPtr = static_cast<THByteTensor *>(output.tensorPtr);

    // Build new storage on top of the Mat
    outputPtr->storage = THByteStorage_newWithData(
            reinterpret_cast<unsigned char *>(mat.data),
            mat.step[0] * mat.rows
    );

    int sizeMultiplier;
    if (mat.channels() == 1) {
        outputPtr->nDimension = mat.dims;
        sizeMultiplier = mat.elemSize1() / mat.numChannels();
    } else {
        outputPtr->nDimension = mat.dims + 1;
        sizeMultiplier = mat.elemSize1();
    }

    outputPtr->size = static_cast<long *>(THRealloc(
            outputPtr->size,
            sizeof(long) * outputPtr->nDimension));
    outputPtr->stride = static_cast<long *>(THRealloc(
            outputPtr->stride,
            sizeof(long) * outputPtr->nDimension));

    if (mat.channels() > 1) {
        outputPtr->size[outputPtr->nDimension - 1] = mat.channels();
        outputPtr->stride[outputPtr->nDimension - 1] = sizeof(float);
    }

    for (int i = 0; i < mat.dims; ++i) {
        outputPtr->size[i] = mat.size[i];
        outputPtr->stride[i] = mat.step[i] / sizeMultiplier;
    }

    // Prevent OpenCV from deallocating Mat at the end of the scope
    outputMat.addref();
}

extern "C"
void test_mat_to_tensor(TensorWrapper tensor) {
    cv::Mat outputMat = cv::Mat::eye(5, 5, CV_32FC1) * 7.;
    matToTensor(outputMat, tensor);
}

extern "C"
void test_tensor_to_mat(TensorWrapper tensor) {
    cv::Mat temp = tensorToMat(tensor);
    std::cout << temp * 10. << std::endl;
}