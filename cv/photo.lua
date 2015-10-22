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


struct TensorWrapper decolor(struct TensorWrapper src, struct TensorWrapper grayscale,
                            struct TensorWrapper color_boost);

struct TensorWrapper seamlessClone(struct TensorWrapper src, struct TensorWrapper dst,
                            struct TensorWrapper mask, struct PointWrapper p,
                            struct TensorWrapper blend, int flags);

struct TensorWrapper colorChange(struct TensorWrapper src, struct TensorWrapper mask,
                            struct TensorWrapper dst, float red_mul,
                            float green_mul, float blue_mul);

struct TensorWrapper illuminationChange(struct TensorWrapper src, struct TensorWrapper mask,
                            struct TensorWrapper dst, float alpha, float beta);

struct TensorWrapper textureFlattening(struct TensorWrapper src, struct TensorWrapper mask,
                            struct TensorWrapper dst, float low_threshold, float high_threshold,
                            int kernel_size);

struct TensorWrapper edgePreservingFilter(struct TensorWrapper src, struct TensorWrapper dst,
                            int flags, float sigma_s, float sigma_r);

struct TensorWrapper detailEnhance(struct TensorWrapper src, struct TensorWrapper dst,
                            float sigma_s, float sigma_r);

struct TensorWrapper pencilSketch(struct TensorWrapper src, struct TensorWrapper dst1,
                            struct TensorWrapper dst2, float sigma_s, float sigma_r, float shade_factor);

struct TensorWrapper stylization(struct TensorWrapper src, struct TensorWrapper dst,
                            float sigma_s, float sigma_r);
]]

local C = ffi.load(libPath('photo'))

function cv.inpaint(t)
    local src =           assert(t.src)
    local inpaintMask =   assert(t.inpaintMask)
    local dst =           t.dst
    local inpaintRadius = assert(t.inpaintRadius)
    local flags =         assert(t.flags)
    
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
    
    assert(templateWindowSize % 2 == 1)
    
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
    
    assert(templateWindowSize % 2 == 1)
    assert(searchWindowSize % 2 == 1)

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

    assert(temporalWindowSize % 2 == 1)
    assert(templateWindowSize % 2 == 1)
    assert(searchWindowSize % 2 == 1)

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

    assert(temporalWindowSize % 2 == 1)
    assert(templateWindowSize % 2 == 1)
    assert(searchWindowSize % 2 == 1)

    if #srcImgs == 1 then
        return cv.unwrap_tensors(
            C.fastNlMeansDenoisingColored(
                cv.wrap_tensors(srcImgs[1]), cv.wrap_tensors(dst), h, hColor, templateWindowSize, searchWindowSize))
    end

    return cv.unwrap_tensors(
        C.fastNlMeansDenoisingColoredMulti(
            cv.wrap_tensors(srcImgs), cv.wrap_tensors(dst), imgToDenoiseIndex, temporalWindowSize, h, hColor, templateWindowSize, searchWindowSize))
end

function cv.decolor(t)
    local src         = assert(t.src)
    local grayscale   = t.grayscale
    local color_boost = assert(t.color_boost)

    return cv.unwrap_tensors(
        C.decolor(
            cv.wrap_tensors(src), cv.wrap_tensors(grayscale), cv.wrap_tensors(color_boost)))
end

function cv.seamlessClone(t)
    local src   = assert(t.src)
    local dst   = assert(t.dst)
    local mask  = assert(t.mask)
    local p     = cv.Point(assert(t.p))
    local blend = t.blend
    local flags = assert(t.flags)

    if blend then
        assert(blend:type() == dst:type() and dst:isSameSizeAs(blend))
    end

    return cv.unwrap_tensors(
        C.seamlessClone(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), cv.wrap_tensors(mask),
            p, cv.wrap_tensors(blend), flags))
end

function cv.colorChange(t)
    local src       = assert(t.src)
    local mask      = assert(t.mask)
    local dst       = t.dst
    local red_mul   = t.red_mul or 1.0
    local green_mul = t.green_mul or 1.0
    local blue_mul  = t.blue_mul or 1.0

    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.colorChange(
            cv.wrap_tensors(src), cv.wrap_tensors(mask), cv.wrap_tensors(dst),
            red_mul, green_mul, blue_mul))
end

function cv.illuminationChange(t)
    local src   = assert(t.src)
    local mask  = assert(t.mask)
    local dst   = t.dst
    local alpha = t.alpha or 0.2
    local beta  = t.beta or 0.4

    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.illuminationChange(
            cv.wrap_tensors(src), cv.wrap_tensors(mask), cv.wrap_tensors(dst),
            alpha, beta))
end


function cv.textureFlattening(t)
    local src            = assert(t.src)
    local mask           = assert(t.mask)
    local dst            = t.dst
    local low_threshold  = t.low_threshold or 30
    local high_threshold = t.high_threshold or 45
    local kernel_size    = t.kernel_size or 3

    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.textureFlattening(
            cv.wrap_tensors(src), cv.wrap_tensors(mask), cv.wrap_tensors(dst),
            low_threshold, high_threshold, kernel_size))
end

function cv.edgePreservingFilter(t)
    local src     = assert(t.src)
    local dst     = t.dst
    local flags   = t.flags or 1
    local sigma_s = t.sigma_s or 60
    local sigma_r = t.sigma_r or 0.4

    return cv.unwrap_tensors(
        C.edgePreservingFilter(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), flags, sigma_s, sigma_r))
end

function cv.detailEnhance(t)
    local src     = assert(t.src)
    local dst     = t.dst
    local sigma_s = t.sigma_s or 10
    local sigma_r = t.sigma_r or 0.15
    
    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.detailEnhance(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), sigma_s, sigma_r))
end

function cv.pencilSketch(t)
    local src          = assert(t.src)
    local dst1         = t.dst1
    local dst          = t.dst2
    local sigma_s      = t.sigma_s or 60
    local sigma_r      = t.sigma_r or 0.07
    local shade_factor = t.shade_factor or 0.02

    if dst2 then
        assert(dst2:type() == src:type() and src:isSameSizeAs(dst2))
    end

    return cv.unwrap_tensors(
        C.pencilSketch(
            cv.wrap_tensors(src), cv.wrap_tensors(dst1), cv.wrap_tensors(dst2),
            sigma_s, sigma_r, shade_factor))
end

function cv.stylization(t)
    local src     = assert(t.src)
    local dst     = t.dst
    local sigma_s = t.sigma_s or 60
    local sigma_r = t.sigma_r or 0.45
    
    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.stylization(
            cv.wrap_tensors(src), cv.wrap_tensors(dst), sigma_s, sigma_r))
end