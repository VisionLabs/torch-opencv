#include <highgui.hpp>

extern "C"
void imshow(const char *winname, struct TensorWrapper image)
{
    cv::imshow(winname, image.toMat());
}

extern "C"
int waitKey(int delay)
{
    return cv::waitKey(delay);
}

extern "C"
void namedWindow(const char *winname, int flags)
{
    cv::namedWindow(winname, flags);
}

extern "C"
void destroyWindow(const char *winname)
{
    cv::destroyWindow(winname);
}

extern "C"
void destroyAllWindows()
{
    cv::destroyAllWindows();
}

extern "C"
int startWindowThread()
{
    return cv::startWindowThread();
}

extern "C"
void resizeWindow(const char *winname, int width, int height)
{
    cv::resizeWindow(winname, width, height);
}

extern "C"
void moveWindow(const char *winname, int x, int y)
{
    cv::moveWindow(winname, x, y);
}

extern "C"
void setWindowProperty(const char *winname, int prop_id, double prop_value)
{
    cv::setWindowProperty(winname, prop_id, prop_value);
}

extern "C"
void setWindowTitle(const char *winname, const char *title)
{
    cv::setWindowTitle(winname, title);
}

extern "C"
double getWindowProperty(const char *winname, int prop_id)
{
    return cv::getWindowProperty(winname, prop_id);
}

extern "C"
void setMouseCallback(const char *winname, cv::MouseCallback onMouse, void *userdata)
{
    cv::setMouseCallback(winname, onMouse, userdata);
}

extern "C"
int getMouseWheelData(int flags)
{
    return cv::getMouseWheelDelta(flags);
}

extern "C"
int createTrackbar(
        const char *trackbarname, const char *winname, int *value,
        int count, cv::TrackbarCallback onChange, void *userdata)
{
    return cv::createTrackbar(trackbarname, winname, value, count, onChange, userdata);
}

extern "C"
int getTrackbarPos(const char *trackbarname, const char *winname)
{
    return cv::getTrackbarPos(trackbarname, winname);
}

extern "C"
void setTrackbarPos(const char *trackbarname, const char *winname, int pos)
{
    cv::setTrackbarPos(trackbarname, winname, pos);
}

extern "C"
void setTrackbarMax(const char *trackbarname, const char *winname, int maxval)
{
    cv::setTrackbarMax(trackbarname, winname, maxval);
}

extern "C"
void updateWindow(const char *winname)
{
    cv::updateWindow(winname);
}

extern "C"
void displayOverlay(const char *winname, const char *text, int delayms)
{
    cv::displayOverlay(winname, text, delayms);
}

extern "C"
void displayStatusBar(const char *winname, const char *text, int delayms)
{
    cv::displayStatusBar(winname, text, delayms);
}

extern "C"
void saveWindowParameters(const char *windowName)
{
    cv::saveWindowParameters(windowName);
}

extern "C"
void loadWindowParameters(const char *windowName)
{
    cv::loadWindowParameters(windowName);
}

int createButton(const char *bar_name, cv::ButtonCallback on_change,
                  void* userdata, int type, bool initial_button_state)
{
    return cv::createButton(bar_name, on_change, userdata, type, initial_button_state);
}
