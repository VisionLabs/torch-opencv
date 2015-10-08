#include <highgui.hpp>

extern "C"
void imshow(const char *winname, struct TensorWrapper image) {
    cv::imshow(winname, image.toMat());
}

extern "C"
int waitKey(int delay) {
    cv::waitKey(delay);
}
