#include <imgcodecs.hpp>

extern "C"
struct TensorWrapper imread(const char *filename, int flags) {
    cv::Mat retval = cv::imread(filename, flags);
    return matToTensor(retval);
}
