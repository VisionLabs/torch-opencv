require 'cv'

local ffi = require 'ffi'

ffi.cdef[[

struct TensorWrapper inpaint(struct TensorWrapper src, struct TensorWrapper inpaintMask,
                            struct TensorWrapper dst, double inpaintRadius, int flags);

struct TensorWrapper fastNlMeansDenoising1(struct TensorWrapper src, struct TensorWrapper dst,
                            float h, int templateWindowSize,
                            int searchWindowSize);

struct TensorWrapper fastNlMeansDenoising2(struct TensorWrapper src, struct TensorWrapper dst,
                            struct TensorWrapper h, int templateWindowSize,
                            int searchWindowSize, int normType);

struct TensorWrapper fastNlMeansDenoisingColored(struct TensorWrapper src, struct TensorWrapper dst,
                            float h, float hColor, int templateWindowSize, int searchWindowSize);

struct TensorWrapper fastNlMeansDenoisingMulti1(struct TensorArray srcImgs, struct TensorWrapper dst,
                            int imgToDenoiseIndex, int temporalWindowSize, float h,
                            int templateWindowSize, int searchWindowSize);

struct TensorWrapper fastNlMeansDenoisingMulti2(struct TensorArray srcImgs, struct TensorWrapper dst,
                            int imgToDenoiseIndex, int temporalWindowSize, struct TensorWrapper h,
                            int templateWindowSize, int searchWindowSize, int normType);

struct TensorWrapper fastNlMeansDenoisingColoredMulti(struct TensorArray srcImgs, struct TensorWrapper dst,
                            int imgToDenoiseIndex, int temporalWindowSize, float h,
                            float hColor, int templateWindowSize, int searchWindowSize);

struct TensorWrapper denoise_TVL1(struct TensorArray observations, struct TensorWrapper result,
                            double lambda, int niters);

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

struct PtrWrapper Tonemap_ctor();

struct TensorWrapper Tonemap_process(struct PtrWrapper ptr, struct TensorArray src, struct TensorWrapper dst);

float Tonemap_getGamma(struct PtrWrapper ptr);

void Tonemap_setGamma(struct PtrWrapper ptr, float gamma);

]]

local C = ffi.load(libPath('photo'))

function cv.inpaint(t)
    local argRules = {
        {"src"},
        {"inpaintMask"},
        {"dst", default = nil},
        {"inpaintRadius"},
        {"flags"}
    }
    local src, inpaintMask, dst, inpaintRadius, flags = cv.argcheck(t, argRules)
    
    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end
    
    return cv.unwrap_tensors(
        C.inpaint(
            cv.wrap_tensor(src), cv.wrap_tensor(inpaintMask), cv.wrap_tensor(dst), inpaintRadius, flags))
end

function cv.fastNlMeansDenoising(t)
    local argRules = {
        {"src"},
        {"dst", default = nil},
        {"h", default = nil}
        {"templateWindowSize", default = 7},
        {"searchWindowSize", default = 21},
        {"normType", default = cv.NORM_L2}
    }
    local src, dst, h, templateWindowSize, searchWindowSize, normType = cv.argcheck(t, argRules)
    
    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    assert(templateWindowSize % 2 == 1)

    if type(h) == "number" or h == nil then
        h = h or 3
        
        return cv.unwrap_tensors(
            C.fastNlMeansDenoising1(
                cv.wrap_tensor(src), cv.wrap_tensor(dst), h, templateWindowSize, searchWindowSize))
    end

    if type(h) == "table" then
        h = torch.FloatTensor(h)
    end

    return cv.unwrap_tensors(
            C.fastNlMeansDenoising2(
                cv.wrap_tensor(src), cv.wrap_tensor(dst), cv.wrap_tensor(h),
                templateWindowSize, searchWindowSize, normType))
end

function cv.fastNlMeansDenoisingColored(t)
    local argRules = {
        {"src"},
        {"dst", default = nil},
        {"h", default = 3},
        {"hColor", default = 3},
        {"templateWindowSize", default = 7},
        {"searchWindowSize", default = 21}
    }
    local src, dst, h, hColor, templateWindowSize, searchWindowSize = cv.argcheck(t, argRules)
    
    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end
    
    assert(templateWindowSize % 2 == 1)
    assert(searchWindowSize % 2 == 1)

    return cv.unwrap_tensors(
        C.fastNlMeansDenoisingColored(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), h, hColor, templateWindowSize, searchWindowSize))
end

struct TensorWrapper fastNlMeansDenoisingMulti1(struct TensorArray srcImgs, struct TensorWrapper dst,
                            int imgToDenoiseIndex, int temporalWindowSize, float h,
                            int templateWindowSize, int searchWindowSize);

struct TensorWrapper fastNlMeansDenoisingMulti2(struct TensorArray srcImgs, struct TensorWrapper dst,
                            int imgToDenoiseIndex, int temporalWindowSize, struct TensorWrapper h,
                            int templateWindowSize, int searchWindowSize, int normType);

function cv.fastNlMeansDenoisingMulti(t)
    local argRules = {
        {"srcImgs"},
        {"dst", default = nil},
        {"imgToDenoiseIndex"},
        {"temporalWindowSize"},
        {"h", default = nil},
        {"templateWindowSize", default = 7},
        {"searchWindowSize", default = 21},
        {"normType", default = cv.NORM_L2}
    }
    local srcImgs, dst, imgToDenoiseIndex, temporalWindowSize, h, templateWindowSize, searchWindowSize, normType = cv.argcheck(t, argRules)

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

    -- h is a single number
    if type(t.h) == "number" or t.h == nil then
        h = t.h or 3
        return cv.unwrap_tensors(
            C.fastNlMeansDenoisingMulti1(
                cv.wrap_tensors(srcImgs), cv.wrap_tensor(dst), imgToDenoiseIndex, temporalWindowSize,
                h, templateWindowSize, searchWindowSize))
    end

    -- h is a vector
    if type(h) == "table" then
        h = torch.FloatTensor(h)
    end

    return cv.unwrap_tensors(
        C.fastNlMeansDenoisingMulti2(
            cv.wrap_tensors(srcImgs), cv.wrap_tensor(dst), imgToDenoiseIndex, temporalWindowSize,
            cv.wrap_tensor(h), templateWindowSize, searchWindowSize, normType))
end

function cv.fastNlMeansDenoisingColoredMulti(t)
    local argRules = {
        {"srcImgs"},
        {"dst", default = nil},
        {"imgToDenoiseIndex"},
        {"temporalWindowSize"},
        {"h", default = 3},
        {"hColor", default = 3},
        {"templateWindowSize", default = 7},
        {"searchWindowSize", default = 21}
    }
    local srcImgs, dst, imgToDenoiseIndex, temporalWindowSize, h, hColor, templateWindowSize, searchWindowSize = cv.argcheck(t, argRules)

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

    return cv.unwrap_tensors(
        C.fastNlMeansDenoisingColoredMulti(
            cv.wrap_tensors(srcImgs), cv.wrap_tensor(dst), imgToDenoiseIndex, temporalWindowSize,
            h, hColor, templateWindowSize, searchWindowSize))
end

function cv.denoise_TVL1(t)
    local argRules = {
        {"observations"},
        {"result", default = nil},
        {"lambda", default = 1.0},
        {"niters", default = 30}
    }
    local observations, result, lambda, niters = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.denoise_TVL1(
            cv.wrap_tensors(observations), cv.wrap_tensor(result), lambda, niters))
end

function cv.decolor(t)
    local argRules = {
        {"src"},
        {"grayscale", default = nil},
        {"color_boost"}
    }
    local src, grayscale, color_boost = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.decolor(
            cv.wrap_tensor(src), cv.wrap_tensor(grayscale), cv.wrap_tensor(color_boost)))
end

function cv.seamlessClone(t)
    local argRules = {
        {"src"},
        {"dst"},
        {"mask"},
        {"p", operator = cv.Point},
        {"blend", default = nil},
        {"flags"}
    }
    local src, dst, mask, p, blend, flags = cv.argcheck(t, argRules)

    if blend then
        assert(blend:type() == dst:type() and dst:isSameSizeAs(blend))
    end

    return cv.unwrap_tensors(
        C.seamlessClone(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), cv.wrap_tensor(mask),
            p, cv.wrap_tensor(blend), flags))
end

function cv.colorChange(t)
    local argRules = {
        {"src"},
        {"mask"},
        {"dst", default = nil},
        {"red_mul", default = 1.0},
        {"green_mul", default = 1.0},
        {"blue_mul", default = 1.0}
    }
    local src, mask, dst, red_mul, green_mul, blue_mul = cv.argcheck(t, argRules)

    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.colorChange(
            cv.wrap_tensor(src), cv.wrap_tensor(mask), cv.wrap_tensor(dst),
            red_mul, green_mul, blue_mul))
end

function cv.illuminationChange(t)
    local argRules = {
        {"src"},
        {"mask"},
        {"dst", default = nil},
        {"alpha", default = 0.2},
        {"beta", default = 0.4}
    }
    local src, mask, dst, alpha, beta = cv.argcheck(t, argRules)

    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.illuminationChange(
            cv.wrap_tensor(src), cv.wrap_tensor(mask), cv.wrap_tensor(dst),
            alpha, beta))
end

function cv.textureFlattening(t)
    local argRules = {
        {"src"},
        {"mask"},
        {"dst", default = nil},
        {"low_threshold", default = 30},
        {"high_threshold", default = 45},
        {"kernel_size", default = 3}
    }
    local src, mask, dst, low_threshold, high_threshold, kernel_size = cv.argcheck(t, argRules)

    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.textureFlattening(
            cv.wrap_tensor(src), cv.wrap_tensor(mask), cv.wrap_tensor(dst),
            low_threshold, high_threshold, kernel_size))
end

function cv.edgePreservingFilter(t)
    local argRules = {
        {"src"},
        {"dst", default = nil},
        {"flags", default = 1},
        {"sigma_s", default = 60},
        {"sigma_r", default = 0.4}
    }
    local src, dst, flags, sigma_s, sigma_r = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.edgePreservingFilter(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), flags, sigma_s, sigma_r))
end

function cv.detailEnhance(t)
    local argRules = {
        {"src"},
        {"dst", default = nil},
        {"sigma_s", default = 10},
        {"sigma_r", default = 0.15}
    }
    local src, dst, sigma_s, sigma_r = cv.argcheck(t, argRules)
    
    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.detailEnhance(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), sigma_s, sigma_r))
end

function cv.pencilSketch(t)
    local argRules = {
        {"src"},
        {"dst1", default = nil},
        {"dst2", default = nil},
        {"sigma_s", default = 60},
        {"sigma_r", default = 0.07},
        {"shade_factor", default = 0.02}
    }
    local src, dst1, dst2, sigma_s, sigma_r, shade_factor = cv.argcheck(t, argRules)

    if dst2 then
        assert(dst2:type() == src:type() and src:isSameSizeAs(dst2))
    end

    return cv.unwrap_tensors(
        C.pencilSketch(
            cv.wrap_tensor(src), cv.wrap_tensor(dst1), cv.wrap_tensor(dst2),
            sigma_s, sigma_r, shade_factor))
end

function cv.stylization(t)
    local argRules = {
        {"src"},
        {"dst", default = nil},
        {"sigma_s", default = 60},
        {"sigma_r", default = 0.45}
    }
    local src, dst, sigma_s, sigma_r = cv.argcheck(t, argRules)
    
    if dst then
        assert(dst:type() == src:type() and src:isSameSizeAs(dst))
    end

    return cv.unwrap_tensors(
        C.stylization(
            cv.wrap_tensor(src), cv.wrap_tensor(dst), sigma_s, sigma_r))
end

-- Tonemap
do
    local Tonemap = torch.class('cv.Tonemap', 'cv.Algorithm')

    function Tonemap:__init()
        self.ptr = ffi.gc(C.Tonemap_ctor(), C.Algorithm_dtor)
    end

    function Tonemap:process(t)
        local argRules = {
            {"src"},
            {"dst", default = nil}
        }
        local src, dst = cv.argcheck(t, argRules)
    
        return C.Tonemap_process(self.ptr, cv.wrap_tensors(src), cv.wrap_tensor(dst));
    end

    function Tonemap:getGamma()
        return C.Tonemap_getGamma(self.ptr);
    end

    function setGamma(t)
        local argRules = {
            {"gamma"}
        }

        local gamma = cv.argcheck(t, argRules)
        C.Tonmap_setGamma(self.ptr, gamma)
    end
end