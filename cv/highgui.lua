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

local C = ffi.load(libPath('highgui'))

function cv.imshow(t)
	local winname = t.winname or "Window 1"
	local image = assert(t.image)
    
    C.imshow(winname, cv.wrap_tensors(image))
end

function cv.waitKey(t)
	local delay = t.delay or 0

    return C.waitKey(delay or 0)
end

function cv.namedWindow(t)
	local winname = t.winname or "Window 1"
	local flags = t.flags or cv.WINDOW_AUTOSIZE

	C.namedWindow(winname, flags)
end

function cv.destroyWindow(t)
    local winname = assert(t.winname)
    
    return C.destroyWindow(winname)
end

function cv.destroyAllWindows(t)
    return C.destroyAllWindows()
end

function cv.startWindowThread(t)
    return C.startWindowThread()
end

function cv.resizeWindow(t)
    local winname = assert(t.winname)
    local width = assert(t.width)
    local height = assert(t.height)

    return C.resizeWindow(winname, width, height)
end

function cv.moveWindow(t)
    local winname = assert(t.winname)
    local x = assert(t.x)
    local y = assert(t.y)
    
    return C.moveWindow(winname, x, y)
end

function cv.setWindowProperty(t)
    local winname = assert(t.winname)
    local prop_id = assert(t.prop_id)
    local prop_value = assert(t.prop_value)
    
    return C.setWindowProperty(winname, prop_id, prop_value)
end

function cv.setWindowTitle(t)
    local winname = assert(t.winname)
    local title = assert(t.title)
    
    return C.setWindowTitle(winname, title)
end

function cv.getWindowProperty(t)
    local winname = assert(t.winname)
    local prop_id = assert(t.prop_id)
    
    return C.getWindowProperty(winname, prop_id)
end

function cv.setMouseCallback(t)
    local winname = assert(t.winname)
    local onMouse = t.onMouse or nil
    local userdata = assert(t.userdata)
    
    return C.setMouseCallback(winname, onMouse, userdata)
end

function cv.getMouseWheelData(t)
    local flags = assert(t.flags)
    
    return C.getMouseWheelData(flags)
end

function cv.createTrackbar(t)
    local trackbarname = assert(t.trackbarname)
    local winname = assert(t.winname)
    local value = t.value or nil
    local count = assert(t.count)
    local onChange = t.onChange or nil
    local userdata = t.userdata or nil
    
    return C.createTrackbar(trackbarname, winname, value, count, onChange, userdata)
end

function cv.getTrackbarPos(t)
    local trackbarname = assert(t.trackbarname)
    local winname = assert(t.winname)
    
    return C.getTrackbarPos(trackbarname, winname)
end

function cv.setTrackbarPos(t)
    local trackbarname = assert(t.trackbarname)
    local winname = assert(t.winname)
    local pos = assert(t.pos)
    
    return C.setTrackbarPos(trackbarname, winname, pos)
end

function cv.setTrackbarMax(t)
    local trackbarname = assert(t.trackbarname)
    local winname = assert(t.winname)
    local maxval = assert(t.maxval)
    
    return C.setTrackbarMax(trackbarname, winname, maxval)
end

function cv.updateWindow(t)
    local winname = assert(t.winname)
    
    return C.updateWindow(winname)
end

function cv.displayOverlay(t)
    local winname = assert(t.winname)
    local text = assert(t.text)
    local delayms = assert(t.delayms)
    
    return C.displayOverlay(winname, text, delayms)
end

function cv.displayStatusBar(t)
    local winname = assert(t.winname)
    local text = assert(t.text)
    local delayms = assert(t.delayms)
    
    return C.displayStatusBar(winname, text, delayms)
end

function cv.saveWindowParameters(t)
    local windowName = assert(t.windowName)
    
    return C.saveWindowParameters(windowName)
end

function cv.loadWindowParameters(t)
    local windowName = assert(t.windowName)
    
    return C.loadWindowParameters(windowName)
end
