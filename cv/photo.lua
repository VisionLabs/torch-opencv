require 'cv'

local ffi = require 'ffi'

ffi.cdef[[

struct TensorWrapper inpaint(struct TensorWrapper src, struct TensorWrapper inpaintMask,
                            struct TensorWrapper dst, double inpaintRadius, int flags);

struct TensorWrapper fastNlMeansDenoising(struct TensorWrapper src, struct TensorWrapper dst,
                            float h, int templateWindowSize, int searchWindowSize);

struct TensorWrapper fastNlMeansDenoisingColored(struct TensorWrapper src, struct TensorWrapper dst,
                            float h, float hColor, int templateWindowSize, int searchWindowSize);

struct TensorWrapper fastNlMeansDenoisingMulti(struct TensorArray srcImgs, struct TensorWrapper dst,
                            int imgToDenoiseIndex, int temporalWindowSize, float h,
                            int templateWindowSize, int searchWindowSize);

struct TensorWrapper fastNlMeansDenoisingColoredMulti(struct TensorArray srcImgs, struct TensorWrapper dst,
                            int imgToDenoiseIndex, int temporalWindowSize, float h,
                            float hColor, int templateWindowSize, int searchWindowSize);

]]

local C = ffi.load(libPath('photo'))

function cv.inpaint(t)
    local src =           assert(t.src)
    local inpaintMask =   assert(t.inpaintMask)
    local dst =           t.dst
    local inpaintRadius = assert(t.inpaintRadius)
    local flags =         assert(t.flags)
    
    assert(src:size(1) == inpaintMask:size(1) and src:size(2) == inpaintMask:size(2))

    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end
    
    return cv.unwrap_tensors(
        C.inpaint(
            cv.wrap_tensors(src), cv.wrap_tensors(inpaintMask), cv.wrap_tensors(dst), inpaintRadius, flags))
end

function cv.fastNlMeansDenoising(t)
    local src =                assert(t.src)
    local dst =                t.dst
    local h =                  t.h or 3
    local templateWindowSize = t.templateWindowSize or 7
    local searchWindowSize =   t.searchWindowSize or 21
    
    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end
    
    return cv.unwrap_tensors(
        C.fastNlMeansDenoising(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), h, templateWindowSize, searchWindowSize))
end

function cv.fastNlMeansDenoisingColored(t)
    local src =                assert(t.src)
    local dst =                t.dst
    local h =                  t.h or 3
    local hColor =             t.hColor or 3
    local templateWindowSize = t.templateWindowSize or 7
    local searchWindowSize =   t.searchWindowSize or 21
    
    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end
    
    return cv.unwrap_tensors(
        C.fastNlMeansDenoisingColored(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), h, hColor, templateWindowSize, searchWindowSize))
end

function cv.fastNlMeansDenoisingMulti(t)
    local srcImgs =            assert(t.srcImgs)
    local dst =                t.dst
    local imgToDenoiseIndex =  assert(t.imgToDenoiseIndex)
    local temporalWindowSize = assert(t.temporalWindowSize)
    local h =                  t.h or 3
    local templateWindowSize = t.templateWindowSize or 7
    local searchWindowSize =   t.searchWindowSize or 21

    if #srcImgs > 1 then 
        for i = 2, #srcImgs do
            assert(srcImgs[i - 1]:type() == srcImgs[i]:type() and srcImgs[i - 1]:isSameSizeAs(srcImgs[i]))
        end
    end

    if dst then
        assert(dst:type() == srcImgs[1]:type() and srcImgs[1]:isSameSizeAs(dst))
    end

    if #srcImgs == 1 then
        return cv.unwrap_tensors(
            C.fastNlMeansDenoising(
                cv.wrap_tensors(srcImgs[1]), cv.wrap_tensors(dst), h, templateWindowSize, searchWindowSize))
    end
    return cv.unwrap_tensors(
        C.fastNlMeansDenoisingMulti(
            cv.wrap_tensors(srcImgs), cv.wrap_tensors(dst), imgToDenoiseIndex, temporalWindowSize, h, templateWindowSize, searchWindowSize))
end

function cv.fastNlMeansDenoisingColoredMulti(t)
    local srcImgs =            assert(t.srcImgs)
    local dst =                t.dst
    local imgToDenoiseIndex =  assert(t.imgToDenoiseIndex)
    local temporalWindowSize = assert(t.temporalWindowSize)
    local h =                  t.h or 3
    local hColor =             t.hColor or 3
    local templateWindowSize = t.templateWindowSize or 7
    local searchWindowSize =   t.searchWindowSize or 21

    if #srcImgs > 1 then 
        for i = 2, #srcImgs do
            assert(srcImgs[i - 1]:type() == srcImgs[i]:type() and srcImgs[i - 1]:isSameSizeAs(srcImgs[i]))
        end
    end

    if dst then
        assert(dst:type() == srcImgs[1]:type() and srcImgs[1]:isSameSizeAs(dst))
    end

    if #srcImgs == 1 then
        return cv.unwrap_tensors(
            C.fastNlMeansDenoisingColored(
                cv.wrap_tensors(srcImgs[1]), cv.wrap_tensors(dst), h, hColor, templateWindowSize, searchWindowSize))
    end

    return cv.unwrap_tensors(
        C.fastNlMeansDenoisingColoredMulti(
            cv.wrap_tensors(srcImgs), cv.wrap_tensors(dst), imgToDenoiseIndex, temporalWindowSize, h, hColor, templateWindowSize, searchWindowSize))
end