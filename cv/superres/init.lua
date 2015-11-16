local cv = require 'cv._env'

local ffi = require 'ffi'

ffi.cdef[[

]]

local C = ffi.load(cv.libPath('superres'))

do
    local FrameSource = cv.newTorchClass('cv.FrameSource')

    function FrameSource:__init(t)
        if t.fileName then
            self.ptr = ffi.gc(C.createFrameSource_Video(t.fileName), C.FrameSource_dtor)
        elseif t.fileName_CUDA then
            self.ptr = ffi.gc(C.createFrameSource_Video_CUDA(t.fileName), C.FrameSource_dtor)
        elseif t.deviceId then
            self.ptr = ffi.gc(C.createFrameSource_Camera(t.deviceId), C.FrameSource_dtor)
        else
            self.ptr = ffi.gc(C.createFrameSource_Empty(), C.FrameSource_dtor)
        end
    end

    function FrameSource:nextFrame(t)
        local argRules = {
            {"frame", default = nil}
        }
        local frame = cv.argcheck(t, argRules)
        
        return cv.unwrap_tensors(C.FrameSource_nextFrame(self.ptr, cv.wrap_tensor(frame)))
    end
end

function cv.createFrameSource_Empty()
    return cv.FrameSource()
end

function cv.createFrameSource_Video(t)
    local argRules = {
        {"fileName", required = true}
    }
    local fileName = cv.argcheck(t, argRules)

    return cv.FrameSource{fileName = fileName}
end

function cv.createFrameSource_Video_CUDA(t)
    local argRules = {
        {"fileName", required = true}
    }
    local fileName = cv.argcheck(t, argRules)

    return cv.FrameSource{fileName_CUDA = fileName}
end

function cv.createFrameSource_Camera(t)
    local argRules = {
        {"deviceId", default = 0}
    }
    local deviceId = cv.argcheck(t, argRules)

    return cv.FrameSource{deviceId = deviceId}
end



return cv
