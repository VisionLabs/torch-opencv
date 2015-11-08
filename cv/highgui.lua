require 'cv'

local ffi = require 'ffi'

ffi.cdef[[
typedef void (*MouseCallback)(int event, int x, int y, int flags, void* userdata);
typedef void (*TrackbarCallback)(int pos, void* userdata);
typedef void (*OpenGlDrawCallback)(void* userdata);
typedef void (*ButtonCallback)(int state, void* userdata);

void imshow(const char *winname, struct TensorWrapper mat);

int waitKey(int delay);

void namedWindow(const char *winname, int flags);

void destroyWindow(const char *winname);

void destroyAllWindows();

int startWindowThread();

void resizeWindow(const char *winname, int width, int height);

void moveWindow(const char *winname, int x, int y);

void setWindowProperty(const char *winname, int prop_id, double prop_value);

void setWindowTitle(const char *winname, const char *title);

double getWindowProperty(const char *winname, int prop_id);

void setMouseCallback(const char *winname, MouseCallback onMouse, void *userdata);

int getMouseWheelData(int flags);

int createTrackbar(
        const char *trackbarname, const char *winname, int *value,
        int count, TrackbarCallback onChange, void *userdata);

int getTrackbarPos(const char *trackbarname, const char *winname);

void setTrackbarPos(const char *trackbarname, const char *winname, int pos);

void setTrackbarMax(const char *trackbarname, const char *winname, int maxval);

void updateWindow(const char *winname);

void displayOverlay(const char *winname, const char *text, int delayms);

void displayStatusBar(const char *winname, const char *text, int delayms);

void saveWindowParameters(const char *windowName);

void loadWindowParameters(const char *windowName);
]]

local C = ffi.load(cv.libPath('highgui'))

function cv.imshow(t)
    local argRules = {
        {"winname", default = "Window 1"},
        {"image", required = true}
    }
    local winname, image = cv.argcheck(t, argRules)
    
    C.imshow(winname, cv.wrap_tensor(image))
end

function cv.waitKey(t)
    local argRules = {
        {"delay", default = 0}
    }
    local delay = cv.argcheck(t, argRules)

    return C.waitKey(delay or 0)
end

function cv.namedWindow(t)
    local argRules = {
        {"winname", default = "Window 1"},
        {"flags", default = cv.WINDOW_AUTOSIZE}
    }
    local winname, flags = cv.argcheck(t, argRules)

    C.namedWindow(winname, flags)
end

function cv.destroyWindow(t)
    local argRules = {
        {"winname", required = true}
    }
    local winname = cv.argcheck(t, argRules)
    
    return C.destroyWindow(winname)
end

function cv.destroyAllWindows(t)
    return C.destroyAllWindows()
end

function cv.startWindowThread(t)
    return C.startWindowThread()
end

function cv.resizeWindow(t)
    local argRules = {
        {"winname", required = true},
        {"width", required = true},
        {"height", required = true}
    }
    local winname, width, height = cv.argcheck(t, argRules)

    return C.resizeWindow(winname, width, height)
end

function cv.moveWindow(t)
    local argRules = {
        {"winname", required = true},
        {"x", required = true},
        {"y", required = true}
    }
    local winname, x, y = cv.argcheck(t, argRules)
    
    return C.moveWindow(winname, x, y)
end

function cv.setWindowProperty(t)
    local argRules = {
        {"winname", required = true},
        {"prop_id", required = true},
        {"prop_value", required = true}
    }
    local winname, prop_id, prop_value = cv.argcheck(t, argRules)
    
    return C.setWindowProperty(winname, prop_id, prop_value)
end

function cv.setWindowTitle(t)
    local argRules = {
        {"winname", required = true},
        {"title", required = true}
    }
    local winname, title = cv.argcheck(t, argRules)
    
    return C.setWindowTitle(winname, title)
end

function cv.getWindowProperty(t)
    local argRules = {
        {"winname", required = true},
        {"prop_id", required = true}
    }
    local winname, prop_id = cv.argcheck(t, argRules)
    
    return C.getWindowProperty(winname, prop_id)
end

function cv.setMouseCallback(t)
    local argRules = {
        {"winname", required = true},
        {"onMouse", default = nil},
        {"userdata", required = true}
    }
    local winname, onMouse, userdata = cv.argcheck(t, argRules)
    
    return C.setMouseCallback(winname, onMouse, userdata)
end

function cv.getMouseWheelData(t)
    local argRules = {
        {"flags", required = true}
    }
    local flags = cv.argcheck(t, argRules)
    
    return C.getMouseWheelData(flags)
end

function cv.createTrackbar(t)
    local argRules = {
        {"trackbarname", required = true},
        {"winname", required = true},
        {"value", default = nil},
        {"count", required = true},
        {"onChange", default = nil},
        {"userdata", default = nil}
    }
    local trackbarname, winname, value, count, onChange, userdata = cv.argcheck(t, argRules)
    
    return C.createTrackbar(trackbarname, winname, value, count, onChange, userdata)
end

function cv.getTrackbarPos(t)
    local argRules = {
        {"trackbarname", required = true},
        {"winname", required = true}
    }
    local trackbarname, winname = cv.argcheck(t, argRules)
    
    return C.getTrackbarPos(trackbarname, winname)
end

function cv.setTrackbarPos(t)
    local argRules = {
        {"trackbarname", required = true},
        {"winname", required = true},
        {"pos", required = true}
    }
    local trackbarname, winname, pos = cv.argcheck(t, argRules)
    
    return C.setTrackbarPos(trackbarname, winname, pos)
end

function cv.setTrackbarMax(t)
    local argRules = {
        {"trackbarname", required = true},
        {"winname", required = true},
        {"maxval", required = true}
    }
    local trackbarname, winname, maxval = cv.argcheck(t, argRules)
    
    return C.setTrackbarMax(trackbarname, winname, maxval)
end

function cv.updateWindow(t)
    local argRules = {
        {"winname", required = true}
    }
    local winname = cv.argcheck(t, argRules)
    
    return C.updateWindow(winname)
end

function cv.displayOverlay(t)
    local argRules = {
        {"winname", required = true},
        {"text", required = true},
        {"delayms", required = true}
    }
    local winname, text, delayms = cv.argcheck(t, argRules)
    
    return C.displayOverlay(winname, text, delayms)
end

function cv.displayStatusBar(t)
    local argRules = {
        {"winname", required = true},
        {"text", required = true},
        {"delayms", required = true}
    }
    local winname, text, delayms = cv.argcheck(t, argRules)
    
    return C.displayStatusBar(winname, text, delayms)
end

function cv.saveWindowParameters(t)
    local argRules = {
        {"windowName", required = true}
    }
    local windowName = cv.argcheck(t, argRules)
    
    return C.saveWindowParameters(windowName)
end

function cv.loadWindowParameters(t)
    local argRules = {
        {"windowName", required = true}
    }
    local windowName = cv.argcheck(t, argRules)
    
    return C.loadWindowParameters(windowName)
end
