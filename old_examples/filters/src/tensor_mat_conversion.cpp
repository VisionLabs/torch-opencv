#include <opencv2/core/core.hpp>

extern "C" {
#include <TH/TH.h>
}

#include <iostream>
#include <array>

// Although we don't yet care about real THTensor type (always using
// float), gaps between rows are (hopefully) handled correctly.

cv::Mat tensorToMat(THFloatTensor *tensor) {
    int numberOfDims = tensor->nDimension;
    // THTensor stores its dimensions sizes under long *.
    // In a constructor for cv::Mat, we need const int *.
    // We can't guarantee int and long to be equal.
    // So we somehow need to static_cast THTensor sizes.
    // TODO: we should somehow get rid of array allocation
    std::array<int, 3> size;
    std::copy(tensor->size, tensor->size+ tensor->nDimension, size.begin());

    // Same thing for stride values.
    std::array<size_t, 3> stride;
    std::copy(tensor->stride, tensor->stride + tensor->nDimension, stride.begin());

    // Determine the number of channels.
    int type;
    // cv::Mat() takes stride values in bytes, so we have to multiply by the element size:
    size_t sizeMultiplier;

    if (tensor->nDimension <= 2) {
        // If such tensor is passed, assume that it is single-channel:
        type = CV_32FC1;
        sizeMultiplier = cv::getElemSize(type);
    } else {
        // Otherwise depend on the 3rd dimension:
        type = CV_32FC(tensor->size[2]);
        // In this case, stride values are already multiplied by the number of channels
        sizeMultiplier = sizeof(float);
        numberOfDims = 2;
    }

    std::for_each(stride.begin(),
                  stride.end(),
                  [sizeMultiplier] (size_t & x) { x *= sizeMultiplier; });

    return cv::Mat(
            numberOfDims,
            size.data(),
            type,
            THFloatTensor_data(tensor),
            stride.data()
    );
}

void matToTensor(cv::Mat & mat, THFloatTensor * output) {

    // This is awful: we first allocate this space in Lua,
    // then deallocate it here, and then allocate again.
    // But it works though.
    // TODO: avoid extra allocations!
    THFloatStorage_free(output->storage);
    THFree(output->size);
    THFree(output->stride);

    // Build new storage on top of the Mat
    output->storage = THFloatStorage_newWithData(
            reinterpret_cast<float *>(mat.data),
            (mat.step[0] * mat.rows) / sizeof(float)
    );

    int sizeMultiplier;
    if (mat.channels() == 1) {
        output->nDimension = mat.dims;
        sizeMultiplier = sizeof(float);
    } else {
        output->nDimension = mat.dims + 1;
        sizeMultiplier = mat.elemSize1();
    }

    output->size = static_cast<long *>(THRealloc(
            output->size,
            sizeof(long) * output->nDimension));
    output->stride = static_cast<long *>(THRealloc(
            output->stride,
            sizeof(long) * output->nDimension));

    if (mat.channels() > 1) {
        output->size[output->nDimension - 1] = mat.channels();
        output->stride[output->nDimension - 1] = sizeof(float);
    }

    for (int i = 0; i < mat.dims; ++i) {
        output->size[i] = mat.size[i];
        output->stride[i] = mat.step[i] / sizeMultiplier;
    }
}

extern "C"
void test_tensor_to_mat(THFloatTensor *input) {
    cv::Mat temp = tensorToMat(input);
    std::cout << temp * 10. << std::endl;
}

extern "C"
void test_mat_to_tensor(THFloatTensor *output) {
    cv::Mat outputMat = cv::Mat::eye(5, 5, CV_32FC1) * 7.;
    // Prevent OpenCV from deallocating Mat at the end of the scope
    outputMat.addref();
    matToTensor(outputMat, output);
}
