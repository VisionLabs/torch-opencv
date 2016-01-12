local cv = require 'cv._env'

local ffi = require 'ffi'

ffi.cdef[[
struct TensorPlusRectArray groupRectangles(struct RectArray rectList, int groupThreshold, double eps);

bool BaseCascadeClassifier_empty(struct PtrWrapper ptr);

bool BaseCascadeClassifier_load(struct PtrWrapper ptr, const char *filename);

bool BaseCascadeClassifier_isOldFormatCascade(struct PtrWrapper ptr);

struct SizeWrapper BaseCascadeClassifier_getOriginalWindowSize(struct PtrWrapper ptr);

int BaseCascadeClassifier_getFeatureType(struct PtrWrapper ptr);

struct PtrWrapper CascadeClassifier_ctor_default();

struct PtrWrapper CascadeClassifier_ctor(const char *filename);

bool CascadeClassifier_read(struct PtrWrapper ptr, struct PtrWrapper node);

struct RectArray CascadeClassifier_detectMultiScale(struct PtrWrapper ptr,
        struct TensorWrapper image, double scaleFactor, int minNeighbors, int flags,
        struct SizeWrapper minSize, struct SizeWrapper maxSize);

struct TensorPlusRectArray CascadeClassifier_detectMultiScale2(struct PtrWrapper ptr,
        struct TensorWrapper image, double scaleFactor, int minNeighbors, int flags,
        struct SizeWrapper minSize, struct SizeWrapper maxSize);

struct TensorArrayPlusRectArray CascadeClassifier_detectMultiScale3(
        struct PtrWrapper ptr, struct TensorWrapper image, double scaleFactor,
        int minNeighbors, int flags, struct SizeWrapper minSize, struct SizeWrapper maxSize,
        bool outputRejectLevels);

bool CascadeClassifier_convert(
        struct PtrWrapper ptr, const char *oldcascade, const char *newcascade);

struct PtrWrapper HOGDescriptor_ctor(
        struct SizeWrapper winSize, struct SizeWrapper blockSize, struct SizeWrapper blockStride,
        struct SizeWrapper cellSize, int nbins, int derivAperture, double winSigma,
        int histogramNormType, double L2HysThreshold, bool gammaCorrection,
        int nlevels, bool signedGradient);

void HOGDescriptor_dtor(struct PtrWrapper ptr);

size_t HOGDescriptor_getDescriptorSize(struct PtrWrapper ptr);

bool HOGDescriptor_checkDetectorSize(struct PtrWrapper ptr);

double HOGDescriptor_getWinSigma(struct PtrWrapper ptr);

void HOGDescriptor_setSVMDetector(struct PtrWrapper ptr, struct TensorWrapper _svmdetector);

bool HOGDescriptor_load(
        struct PtrWrapper ptr, const char *filename, const char *objname);

void HOGDescriptor_save(
        struct PtrWrapper ptr, const char *filename, const char *objname);

struct TensorWrapper HOGDescriptor_compute(
        struct PtrWrapper ptr, struct TensorWrapper img, struct SizeWrapper winStride,
        struct SizeWrapper padding, struct PointArray locations);

struct TensorPlusPointArray HOGDescriptor_detect(
        struct PtrWrapper ptr, struct TensorWrapper img, double hitThreshold,
        struct SizeWrapper winStride, struct SizeWrapper padding, struct PointArray searchLocations);

struct TensorPlusRectArray HOGDescriptor_detectMultiScale(
        struct PtrWrapper ptr, struct TensorWrapper img, double hitThreshold,
        struct SizeWrapper winStride, struct SizeWrapper padding, double scale,
        double finalThreshold, bool useMeanshiftGrouping);

struct TensorArray HOGDescriptor_computeGradient(
        struct PtrWrapper ptr, struct TensorWrapper img,
        struct SizeWrapper paddingTL, struct SizeWrapper paddingBR);

struct TensorWrapper HOGDescriptor_getDefaultPeopleDetector();

struct TensorWrapper HOGDescriptor_getDaimlerPeopleDetector();
]]

local C = ffi.load(cv.libPath('objdetect'))

require 'cv.Classes'
local Classes = ffi.load(cv.libPath('Classes'))

function cv.groupRectangles(t)
    local argRules = {
        {"rectList", required = true},
        {"groupThreshold", required = true},
        {"eps", default = 0.2}
    }
    local rectList, groupThreshold, eps = cv.argcheck(t, argRules)

    assert(ffi.typeof(rectList) == ffi.typeof('struct RectArray'))

    local result = C.groupRectangles(rectList, groupThreshold, eps)
    return result.rects, cv.unwrap_tensors(result.tensor)
end

do
    local BaseCascadeClassifier = torch.class('cv.BaseCascadeClassifier', 'cv.Algorithm', cv)

    function BaseCascadeClassifier:empty()
        return C.BaseCascadeClassifier_empty(self.ptr)
    end

    function BaseCascadeClassifier:load(t)
        local argRules = {
            {"filename", required = "true"}
        }
        local filename = cv.argcheck(t, argRules)

        return C.BaseCascadeClassifier_load(self.ptr, filename)
    end

    function BaseCascadeClassifier:isOldFormatCascade()
        return C.BaseCascadeClassifier_isOldFormatCascade(self.ptr)
    end

    function BaseCascadeClassifier:getOriginalWindowSize()
        return C.BaseCascadeClassifier_getOriginalWindowSize(self.ptr)
    end

    function BaseCascadeClassifier:getFeatureType()
        return C.BaseCascadeClassifier_getFeatureType(self.ptr)
    end
end

do
    local CascadeClassifier = torch.class('cv.CascadeClassifier', 'cv.BaseCascadeClassifier', cv)

    function CascadeClassifier:__init(t)
        local argRules = {
            {"filename", default = nil}
        }
        local filename = cv.argcheck(t, argRules)

        if filename then
            self.ptr = ffi.gc(C.CascadeClassifier_ctor(filename), Classes.Algorithm_dtor)
        else
            self.ptr = ffi.gc(C.CascadeClassifier_ctor_default(), Classes.Algorithm_dtor)
        end
    end

    function CascadeClassifier:read(t)
        local argRules = {
            {"node", required = true}
        }
        local node = cv.argcheck(t, argRules)

        assert(torch.type(node) == "cv.FileNode")

        return C.CascadeClassifier_read(self.ptr, node.ptr)
    end

    function CascadeClassifier:detectMultiScale(t)
        local argRules = {
            {"image", required = true},
            {"scaleFactor", default = 1.1},
            {"minNeighbors", default = 3},
            {"flags", default = 0},
            {"minSize", default = {0, 0}, operator = cv.Size},
            {"maxSize", default = {0, 0}, operator = cv.Size}
        }
        local image, scaleFactor, minNeighbors, flags, minSize, maxSize = cv.argcheck(t, argRules)

        return cv.gcarray(C.CascadeClassifier_detectMultiScale(self.ptr, cv.wrap_tensor(image),
            scaleFactor, minNeighbors, flags, minSize, maxSize))
    end

    function CascadeClassifier:detectMultiScale2(t)
        local argRules = {
            {"image", required = true},
            {"scaleFactor", default = 1.1},
            {"minNeighbors", default = 3},
            {"flags", default = 0},
            {"minSize", default = {0, 0}, operator = cv.Size},
            {"maxSize", default = {0, 0}, operator = cv.Size}
        }
        local image, scaleFactor, minNeighbors, flags, minSize, maxSize = cv.argcheck(t, argRules)

        local result = C.CascadeClassifier_detectMultiScale(self.ptr, cv.wrap_tensor(image),
            scaleFactor, minNeighbors, flags, minSize, maxSize)
        return cv.gcarray(result.rects), result.tensor
    end

    function CascadeClassifier:detectMultiScale3(t)
        local argRules = {
            {"image", required = true},
            {"scaleFactor", default = 1.1},
            {"minNeighbors", default = 3},
            {"flags", default = 0},
            {"minSize", default = {0, 0}, operator = cv.Size},
            {"maxSize", default = {0, 0}, operator = cv.Size},
            {"outputRejectLevels", default = false}
        }
        local image, scaleFactor, minNeighbors, flags, minSize, maxSize = cv.argcheck(t, argRules)

        local result = C.CascadeClassifier_detectMultiScale(self.ptr, cv.wrap_tensor(image),
            scaleFactor, minNeighbors, flags, minSize, maxSize)
        return cv.gcarray(result.rects), cv.unwrap_tensors(result.tensors)
    end

    function CascadeClassifier:convert(t)
        local argRules = {
            {"oldcascade", required = true},
            {"newcascade", required = true}
        }
        local oldcascade, newcascade = cv.argcheck(t, argRules)

        return C.CascadeClassifier_convert(self.ptr, oldcascade, newcascade)
    end
end

do
    local HOGDescriptor = torch.class('cv.HOGDescriptor', cv)

    function HOGDescriptor:__init(t)
        local argRules = {
            {"winSize", default = {64, 128}, operator = cv.Size},
            {"blockSize", default = {16, 16}, operator = cv.Size},
            {"blockStride", default = {8, 8}, operator = cv.Size},
            {"cellSize", default = {8, 8}, operator = cv.Size},
            {"nbins", default = 9},
            {"derivAperture", default = 1},
            {"winSigma", default = -1},
            {"histogramNormType", default = cv.HOGDESCRIPTOR_L2HYS},
            {"L2HysThreshold", default = 0.2},
            {"gammaCorrection", default = true},
            {"nlevels", default = cv.HOGDESCRIPTOR_DEFAULT_NLEVELS},
            {"signedGradient", default = false}
        }
        local winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, 
            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, 
            signedGradient = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.HOGDescriptor_ctor(winSize, blockSize, blockStride, cellSize, 
            nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, 
            nlevels, signedGradient), C.HOGDescriptor_dtor)
    end

    function HOGDescriptor:getDescriptorSize()
        return C.HOGDescriptor_getDescriptorSize(self.ptr)
    end

    function HOGDescriptor:checkDetectorSize()
        return C.HOGDescriptor_checkDetectorSize(self.ptr)
    end

    function HOGDescriptor:getWinSigma()
        return C.HOGDescriptor_getWinSigma(self.ptr)
    end

    function HOGDescriptor:setSVMDetector(t)
        local argRules = {
            {"_svmdetector", required = true}
        }
        local _svmdetector = cv.argcheck(t, argRules)

        C.HOGDescriptor_setSVMDetector(self.ptr, cv.wrap_tensor(_svmdetector))
    end

    function HOGDescriptor:load(t)
        local argRules = {
            {"filename", required = true},
            {"objname", default = ""}
        }
        local filename, objname = cv.argcheck(t, argRules)

        return C.HOGDescriptor_load(self.ptr, filename, objname)
    end

    function HOGDescriptor:save(t)
        local argRules = {
            {"filename", required = true},
            {"objname", default = ""}
        }
        local filename, objname = cv.argcheck(t, argRules)

        C.HOGDescriptor_save(self.ptr, filename, objname)
    end

    function HOGDescriptor:compute(t)
        local argRules = {
            {"img", required = true},
            {"winStride", default = {0, 0}, operator = cv.Size},
            {"padding", default = {0, 0}, operator = cv.Size},
            {"locations", default = ffi.new('struct PointArray', nil)},
        }
        local img, winStride, padding, locations = cv.argcheck(t, argRules)

        assert(ffi.typeof(locations) == ffi.typeof('struct PointArray'))

        return cv.unwrap_tensors(
            C.HOGDescriptor_compute(self.ptr, cv.wrap_tensor(img), winStride, padding, locations))
    end

    function HOGDescriptor:detect(t)
        local argRules = {
            {"img", required = true},
            {"hitThreshold", default = 0},
            {"winStride", default = {0, 0}, operator = cv.Size},
            {"padding", default = {0, 0}, operator = cv.Size},
            {"searchLocations", default = ffi.new('struct PointArray', nil)},
        }
        local img, hitThreshold, winStride, padding, searchLocations = cv.argcheck(t, argRules)

        assert(ffi.typeof(searchLocations) == ffi.typeof('struct PointArray'))

        local result = C.HOGDescriptor_detect(self.ptr, cv.wrap_tensor(img), hitThreshold, 
            winStride, padding, searchLocations)
        return cv.gcarray(result.points), cv.unwrap_tensors(result.tensor)
    end

    function HOGDescriptor:detectMultiScale(t)
        local argRules = {
            {"img", required = true},
            {"hitThreshold", default = 0},
            {"winStride", default = {0, 0}, operator = cv.Size},
            {"padding", default = {0, 0}, operator = cv.Size},
            {"scale", default = 1.05},
            {"finalThreshold", default = 2.0},
            {"useMeanshiftGrouping", default = false}
        }
        local img, hitThreshold, winStride, padding, scale, finalThreshold, 
            useMeanshiftGrouping = cv.argcheck(t, argRules)

        local result = C.HOGDescriptor_detectMultiScale(self.ptr, cv.wrap_tensor(img), hitThreshold, 
            winStride, padding, scale, finalThreshold, useMeanshiftGrouping)
        return cv.gcarray(result.rects), cv.unwrap_tensors(result.tensor)
    end

    function HOGDescriptor:computeGradient(t)
        local argRules = {
            {"img", required = true},
            {"paddingTL", default = {0, 0}, operator = cv.Size},
            {"paddingBR", default = {0, 0}, operator = cv.Size}
        }
        local img, paddingTL, paddingBR = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.HOGDescriptor_computeGradient(
            self.ptr, cv.wrap_tensor(img), paddingTl, paddingBR))
    end

    function HOGDescriptor.getDefaultPeopleDetector()
        return cv.unwrap_tensors(C.HOGDescriptor_getDefaultPeopleDetector())
    end

    function HOGDescriptor.getDaimlerPeopleDetector()
        return cv.unwrap_tensors(C.HOGDescriptor_getDaimlerPeopleDetector())
    end
end
