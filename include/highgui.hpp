#include <Common.hpp>
#include <opencv2/highgui.hpp>

extern "C"
void imshow(const char *winname, struct TensorWrapper mat);

extern "C"
int waitKey(int delay);

extern "C"
void namedWindow(const char *winname, int flags);