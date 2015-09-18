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
    size_t multiplier;

    if (tensor->nDimension <= 2) {
        // If such tensor is passed, assume that it is single-channel:
        type = CV_32FC1;
        multiplier = cv::getElemSize(type);
    } else {
        // Otherwise depend on the 3rd dimension:
        type = CV_32FC(tensor->size[2]);
        // In this case, stride values are already multiplied by the number of channels
        multiplier = sizeof(float);
        numberOfDims = 2;
    }

    std::for_each(stride.begin(),
                  stride.end(),
                  [multiplier] (size_t & x) { x *= multiplier; });

    return cv::Mat(
            numberOfDims,
            size.data(),
            type,
            THFloatTensor_data(tensor),
            stride.data()
    );
}

THFloatTensor *matToTensor(cv::Mat & mat) {
    // TODO this
    return NULL;
}

extern "C"
void test_tensor_conversion(THFloatTensor *tensor) {
    cv::Mat temp = tensorToMat(tensor);
    std::cout << temp << std::endl;
}