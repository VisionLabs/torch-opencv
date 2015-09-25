#include <highgui.hpp>

extern "C"
void imshow(const char *winname, struct TensorWrapper image) {
    cv::Mat temp = image.toMat();
    cv::imshow(winname, temp);
}

extern "C"
int waitKey(int delay) {
    cv::waitKey(delay);
}
