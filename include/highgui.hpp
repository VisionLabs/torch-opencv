#include <Common.hpp>
#include <opencv2/highgui.hpp>

extern "C"
void imshow(const char *winname, struct TensorWrapper mat);

extern "C"
int waitKey(int delay);

extern "C"
void namedWindow(const char *winname, int flags);

extern "C"
void destroyWindow(const char *winname);

extern "C"
void destroyAllWindows();

extern "C"
int startWindowThread();

extern "C"
void resizeWindow(const char *winname, int width, int height);

extern "C"
void moveWindow(const char *winname, int x, int y);

extern "C"
void setWindowProperty(const char *winname, int prop_id, double prop_value);

extern "C"
void setWindowTitle(const char *winname, const char *title);

extern "C"
double getWindowProperty(const char *winname, int prop_id);

extern "C"
void setMouseCallback(const char *winname, cv::MouseCallback onMouse, void *userdata);

extern "C"
int getMouseWheelData(int flags);

extern "C"
int createTrackbar(
        const char *trackbarname, const char *winname, int *value,
        int count, cv::TrackbarCallback onChange, void *userdata);

extern "C"
int getTrackbarPos(const char *trackbarname, const char *winname);

extern "C"
void setTrackbarPos(const char *trackbarname, const char *winname, int pos);

extern "C"
void setTrackbarMax(const char *trackbarname, const char *winname, int maxval);

extern "C"
void updateWindow(const char *winname);

extern "C"
void displayOverlay(const char *winname, const char *text, int delayms);

extern "C"
void displayStatusBar(const char *winname, const char *text, int delayms);

extern "C"
void saveWindowParameters(const char *windowName);

extern "C"
void loadWindowParameters(const char *windowName);
