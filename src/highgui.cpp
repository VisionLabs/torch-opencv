#include <highgui.hpp>

extern "C"
void imshow(const char *winname, struct TensorWrapper mat) {
    cv::Mat temp = tensorToMat(mat);
    cv::imshow(winname, temp);
}

extern "C"
int waitKey(int delay) {
    cv::waitKey(delay);
}
