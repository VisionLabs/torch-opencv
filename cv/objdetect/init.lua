local cv = require 'cv._env'

local ffi = require 'ffi'

ffi.cdef[[
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

    assert(ffi.typeof(rectList) == ffi.typeof(ffi.new('struct RectArray')))

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
            self.ptr = ffi.gc(CascadeClassifier_ctor(filename), Classes.Algorithm_dtor)
        else
            self.ptr = ffi.gc(CascadeClassifier_ctor_default(), Classes.Algorithm_dtor)
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
