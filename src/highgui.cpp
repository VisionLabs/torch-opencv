#include <highgui.hpp>

extern "C"
void imshow(const char *winname, struct TensorWrapper image) {
    cv::imshow(winname, image.toMat());
}

extern "C"
int waitKey(int delay) {
    return cv::waitKey(delay);
}

extern "C"
void namedWindow(const char *winname, int flags) {
    cv::namedWindow(winname, flags);
}