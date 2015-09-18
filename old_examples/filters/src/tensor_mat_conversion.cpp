#include <opencv2/core/core.hpp>

extern "C" {
    #include <TH/TH.h>
}

#include <vector>
#include <iostream>

// Although we don't yet care about real THTensor type (always using
// float), gaps between rows are (hopefully) handled correctly.

cv::Mat tensorToMat(THFloatTensor *tensor) {
    // THTensor stores its dimensions sizes under long *.
    // In a constructor for cv::Mat, we need const int *.
    // We can't guarantee int and long to be equal.
    // So (for now) let's copy and cast THTensor sizes.
    std::vector<int> size(tensor->size, tensor->size + tensor->nDimension);
    // Same thing for stride values.
    std::vector<size_t> stride(tensor->stride, tensor->stride + tensor->nDimension);

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
        type = CV_32FC(size[2]);
        // In this case, stride values are already multiplied by the number of channels
        multiplier = sizeof(float);
        size.pop_back();
    } // No need to handle `nDimension > 3` case as OpenCV itself will throw.

    std::for_each(stride.begin(),
                  stride.end(),
                  [multiplier] (size_t & x) { x *= multiplier; });

    return cv::Mat(
            size.size(),
            size.data(),
            type,
            tensor->storage->data + tensor->storageOffset,
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