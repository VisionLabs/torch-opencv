local cv = require 'cv._env'

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
]]

local C = ffi.load(cv.libPath('photo'))

function cv.inpaint(t)
    local argRules = {
        {"src", required = true},
        {"inpaintMask", required = true},
        {"dst", default = nil},
        {"inpaintRadius", required = true},
        {"flags", required = true}
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
        {"src", required = true},
        {"dst", default = nil},
        {"h", default = nil},
        {"templateWindowSize", default = 7},
        {"searchWindowSize", default = 21},
        {"normType", default = cv.NORM_L2}
    }
    local src, dst, h, templateWindowSize, searchWindowSize, h, normType = cv.argcheck(t, argRules)

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
        {"src", required = true},
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

function cv.fastNlMeansDenoisingMulti(t)
    local argRules = {
        {"srcImgs", required = true},
        {"dst", default = nil},
        {"imgToDenoiseIndex", required = true},
        {"temporalWindowSize", required = true},
        {"h", default = nil},
        {"templateWindowSize", default = 7},
        {"searchWindowSize", default = 21},
        {"normType", default = cv.NORM_L2}
    }
    local srcImgs, dst, imgToDenoiseIndex, temporalWindowSize, h, templateWindowSize, searchWindowSize, h, normType = cv.argcheck(t, argRules)

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
    if type(h) == "number" or h == nil then
        h = h or 3
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
        {"srcImgs", required = true},
        {"dst", default = nil},
        {"imgToDenoiseIndex", required = true},
        {"temporalWindowSize", required = true},
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
        {"observations", required = true},
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
        {"src", required = true},
        {"grayscale", default = nil},
        {"color_boost", required = true}
    }
    local src, grayscale, color_boost = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(
        C.decolor(
            cv.wrap_tensor(src), cv.wrap_tensor(grayscale), cv.wrap_tensor(color_boost)))
end

function cv.seamlessClone(t)
    local argRules = {
        {"src", required = true},
        {"dst", required = true},
        {"mask", required = true},
        {"p", required = true, operator = cv.Point},
        {"blend", default = nil},
        {"flags", required = true}
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
        {"src", required = true},
        {"mask", required = true},
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
        {"src", required = true},
        {"mask", required = true},
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
        {"src", required = true},
        {"mask", required = true},
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
        {"src", required = true},
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
        {"src", required = true},
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
        {"src", required = true},
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
        {"src", required = true},
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


--- ***************** Classes *****************
require 'cv.Classes'

local Classes = ffi.load(cv.libPath('Classes'))

ffi.cdef[[
struct PtrWrapper Tonemap_ctor(float gamma);

struct TensorWrapper Tonemap_process(struct PtrWrapper ptr, struct TensorArray src, struct TensorWrapper dst);

float Tonemap_getGamma(struct PtrWrapper ptr);

void Tonemap_setGamma(struct PtrWrapper ptr, float gamma);

struct PtrWrapper TonemapDrago_ctor(float gamma, float saturation, float bias);

float TonemapDrago_getSaturation(struct PtrWrapper ptr);

void TonemapDrago_setSaturation(struct PtrWrapper ptr, float saturation);

float TonemapDrago_getBias(struct PtrWrapper ptr);

void TonemapDrago_setBias(struct PtrWrapper ptr, float bias);

struct PtrWrapper TonemapDurand_ctor(float gamma, float contrast, float saturation, float sigma_space, float sigma_color);

float TonemapDurand_getSaturation(struct PtrWrapper ptr);

void TonemapDurand_setSaturation(struct PtrWrapper ptr, float Saturation);

float TonemapDurand_getContrast(struct PtrWrapper ptr);

void TonemapDurand_setContrast(struct PtrWrapper ptr, float contrast);

float TonemapDurand_getSigmaSpace(struct PtrWrapper ptr);

void TonemapDurand_setSigmaSpace(struct PtrWrapper ptr, float sigma_space);

float TonemapDurand_getSigmaColor(struct PtrWrapper ptr);

void TonemapDurand_setSigmaColor(struct PtrWrapper ptr, float sigma_color);

struct PtrWrapper TonemapReinhard_ctor(float gamma, float intensity, float light_adapt, float color_adapt);

float TonemapReinhard_getIntensity(struct PtrWrapper ptr);

void TonemapReinhard_setIntensity(struct PtrWrapper ptr, float intensity);

float TonemapReinhard_getLightAdaptation(struct PtrWrapper ptr);

void TonemapReinhard_setLightAdaptation(struct PtrWrapper ptr, float light_adapt);

float TonemapReinhard_getColorAdaptation(struct PtrWrapper ptr);

void TonemapReinhard_setColorAdaptation(struct PtrWrapper ptr, float color_adapt);

struct PtrWrapper TonemapMantiuk_ctor(float gamma, float scale, float saturation);

float TonemapMantiuk_getScale(struct PtrWrapper ptr);

void TonemapMantiuk_setScale(struct PtrWrapper ptr, float scale);

float TonemapMantiuk_getSaturation(struct PtrWrapper ptr);

void TonemapMantiuk_setSaturation(struct PtrWrapper ptr, float saturation);

struct TensorArray AlignExposures_process(struct PtrWrapper ptr, struct TensorArray src, struct TensorArray dst,
                            struct TensorWrapper times, struct TensorWrapper response);

struct PtrWrapper AlignMTB_ctor(int max_bits, int exclude_range, bool cut);

struct TensorArray AlignMTB_process1(struct PtrWrapper ptr, struct TensorArray src, struct TensorArray dst,
                            struct TensorWrapper times, struct TensorWrapper response);

struct TensorArray AlignMTB_process2(struct PtrWrapper ptr, struct TensorArray src, struct TensorArray dst);

struct PointWrapper AlignMTB_calculateShift(struct PtrWrapper ptr, struct TensorWrapper img0, struct TensorWrapper img1);

struct TensorWrapper AlignMTB_shiftMat(struct PtrWrapper ptr, struct TensorWrapper src,
                            struct TensorWrapper dst, struct PointWrapper shift);

void AlignMTB_computeBitmaps(struct PtrWrapper ptr, struct TensorWrapper img, struct TensorWrapper tb, struct TensorWrapper eb);

int AlignMTB_getMaxBits(struct PtrWrapper ptr);

void AlignMTB_setMaxBits(struct PtrWrapper ptr, int max_bits);

int AlignMTB_getExcludeRange(struct PtrWrapper ptr);

void AlignMTB_setExcludeRange(struct PtrWrapper ptr, int exclude_range);

int AlignMTB_getCut(struct PtrWrapper ptr);

void AlignMTB_setCut(struct PtrWrapper ptr, bool cut);

struct TensorWrapper CalibrateCRF_process(struct PtrWrapper ptr, struct TensorArray src, struct TensorWrapper dst,
                            struct TensorWrapper times);

struct PtrWrapper CalibrateDebevec_ctor(int samples, float lambda, bool random);

float CalibrateDebevec_getLambda(struct PtrWrapper ptr);

void CalibrateDebevec_setLambda(struct PtrWrapper ptr, float lambda);

int CalibrateDebevec_getSamples(struct PtrWrapper ptr);

void CalibrateDebevec_setSamples(struct PtrWrapper ptr, int samples);

bool CalibrateDebevec_getRandom(struct PtrWrapper ptr);

void CalibrateDebevec_setRandom(struct PtrWrapper ptr, bool random);

struct PtrWrapper CalibrateRobertson_ctor(int max_iter, float threshold);

int CalibrateRobertson_getMaxIter(struct PtrWrapper ptr);

void CalibrateRobertson_setMaxIter(struct PtrWrapper ptr, int max_iter);

float CalibrateRobertson_getThreshold(struct PtrWrapper ptr);

void CalibrateRobertson_setThreshold(struct PtrWrapper ptr, float threshold);

struct TensorWrapper CalibrateRobertson_getRadiance(struct PtrWrapper ptr);

struct TensorWrapper MergeExposures_process(struct PtrWrapper ptr, struct TensorArray src, struct TensorWrapper dst,
                            struct TensorWrapper times, struct TensorWrapper response);

struct PtrWrapper MergeDebevec_ctor();

struct TensorWrapper MergeDebevec_process1(struct PtrWrapper ptr, struct TensorArray src, struct TensorWrapper dst,
                            struct TensorWrapper times, struct TensorWrapper response);

struct TensorWrapper MergeDebevec_process2(struct PtrWrapper ptr, struct TensorArray src, struct TensorWrapper dst,
                            struct TensorWrapper times);

struct PtrWrapper MergeMertens_ctor(float contrast_weight, float saturation_weight, float exposure_weight);

struct TensorWrapper MergeMertens_process1(struct PtrWrapper ptr, struct TensorArray src, struct TensorWrapper dst,
                            struct TensorWrapper times, struct TensorWrapper response);

struct TensorWrapper MergeMertens_process2(struct PtrWrapper ptr, struct TensorArray src, struct TensorWrapper dst);

float MergeMertens_getContrastWeight(struct PtrWrapper ptr);

void MergeMertens_setContrastWeight(struct PtrWrapper ptr, float contrast_weight);

float MergeMertens_getSaturationWeight(struct PtrWrapper ptr);

void MergeMertens_setSaturationWeight(struct PtrWrapper ptr, float saturation_weight);

float MergeMertens_getExposureWeight(struct PtrWrapper ptr);

void MergeMertens_setExposureWeight(struct PtrWrapper ptr, float exposure_weight);

struct PtrWrapper MergeRobertson_ctor();

struct TensorWrapper MergeRobertson_process1(struct PtrWrapper ptr, struct TensorArray src, struct TensorWrapper dst,
                            struct TensorWrapper times, struct TensorWrapper response);

struct TensorWrapper MergeRobertson_process2(struct PtrWrapper ptr, struct TensorArray src, struct TensorWrapper dst,
                            struct TensorWrapper times);
]]

-- Tonemap

do
    local Tonemap = torch.class('cv.Tonemap', 'cv.Algorithm', cv)

    function Tonemap:__init(t)
        local argRules = {
            {"gamma", default = 1.0}
        }
        local gamma = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.Tonemap_ctor(gamma), Classes.Algorithm_dtor)
    end

    function Tonemap:process(t)
        local argRules = {
            {"src", required = true},
            {"dst", default = nil}
        }
        local src, dst = cv.argcheck(t, argRules)

        return C.Tonemap_process(self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(dst));
    end

    function Tonemap:getGamma()
        return C.Tonemap_getGamma(self.ptr);
    end

    function Tonemap:setGamma(t)
        local argRules = {
            {"gamma", required = true}
        }
        local gamma = cv.argcheck(t, argRules)

        C.Tonemap_setGamma(self.ptr, gamma)
    end 
end

-- TonemapDrago

do
    local TonemapDrago = torch.class('cv.TonemapDrago', 'cv.Tonemap', cv)

    function TonemapDrago:__init(t)
        local argRules = {
            {"gamma", default = 1.0},
            {"saturation", default = 1.0},
            {"bias", default = 0.85}
        }
        local gamma, saturation, bias = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.TonemapDrago_ctor(gamma, saturation, bias), Classes.Algorithm_dtor)
    end

    function TonemapDrago:getSaturation()
        return C.TonemapDrago_getSaturation(self.ptr);
    end

    function TonemapDrago:setSaturation(t)
        local argRules = {
            {"saturation", required = true}
        }
        local saturation = cv.argcheck(t, argRules)

        C.TonemapDrago_setSaturation(self.ptr, saturation)
    end

    function TonemapDrago:getBias()
        return C.TonemapDrago_getBias(self.ptr);
    end

    function TonemapDrago:setBias(t)
        local argRules = {
            {"bias", required = true}
        }
        local bias = cv.argcheck(t, argRules)

        C.TonemapDrago_setBias(self.ptr, bias)
    end
end

-- TonemapDurand

do
    local TonemapDurand = torch.class('cv.TonemapDurand', 'cv.Tonemap', cv);

    function TonemapDurand:__init(t)
        local argRules = {
            {"gamma", default = 1.0},
            {"contrast", default = 4.0},
            {"saturation", default = 1.0},
            {"sigma_space", default = 2.0},
            {"sigma_color", default = 2.0}
        }
        local gamma, contrast, saturation, sigma_space, sigma_color = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.TonemapDurand_ctor(gamma, contrast, saturation, sigma_space, sigma_color), Classes.Algorithm_dtor)
    end

    function TonemapDurand:getSaturation()
        return C.TonemapDurand_getSaturation(self.ptr)
    end

    function TonemapDurand:setSaturation(t)
        local argRules = {
            {"saturation", required = true}
        }
        local saturation = cv.argcheck(t, argRules)

        C.TonemapDurand_setSaturation(self.ptr, saturation)
    end

    function TonemapDurand:getConstant()
        return C.TonemapDurand_getConstant(self.ptr)
    end

    function TonemapDurand:setConstant(t)
        local argRules = {
            {"Constant", required = true}
        }
        local Constant = cv.argcheck(t, argRules)

        C.TonemapDurand_setConstant(self.ptr, Constant)
    end

    function TonemapDurand:getSigmaSpace()
        return C.TonemapDurand_getSigmaSpace(self.ptr)
    end

    function TonemapDurand:setSigmaSpace(t)
        local argRules = {
            {"sigma_space", required = true}
        }
        local sigma_space = cv.argcheck(t, argRules)

        C.TonemapDurand_setSigmaSpace(self.ptr, sigma_space)
    end

    function TonemapDurand:getSigmaColor()
        return C.TonemapDurand_getSigmaColor(self.ptr)
    end

    function TonemapDurand:setSigmaColor(t)
        local argRules = {
            {"sigma_color", required = true}
        }
        local sigma_color = cv.argcheck(t, argRules)

        C.TonemapDurand_setSigmaColor(self.ptr, sigma_color)
    end
end

-- TonemapReinhard

do
    local TonemapReinhard = torch.class('cv.TonemapReinhard', 'cv.Tonemap', cv);

    function TonemapReinhard:__init(t)
        local argRules = {
            {"gamma", default = 1.0},
            {"intensity", default = 0.0},
            {"light_adapt", default = 1.0},
            {"sigma_space", default = 2.0},
            {"sigma_color", default = 2.0}
        }
        local gamma, intensity, light_adapt, color_adapt = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.TonemapReinhard_ctor(gamma, intensity, light_adapt, color_adapt), Classes.Algorithm_dtor)
    end

    function TonemapReinhard:getIntensity()
        return C.TonemapReinhard_getIntensity(self.ptr)
    end

    function TonemapReinhard:setIntensity(t)
        local argRules = {
            {"intensity", required = true}
        }
        local intensity = cv.argcheck(t, argRules)

        C.TonemapReinhard_setIntensity(self.ptr, intensity)
    end

    function TonemapReinhard:getLightAdaptation()
        return C.TonemapReinhard_getLightAdaptation(self.ptr)
    end

    function TonemapReinhard:setLightAdaptation(t)
        local argRules = {
            {"light_adapt", required = true}
        }
        local light_adapt = cv.argcheck(t, argRules)

        C.TonemapReinhard_setLightAdaptation(self.ptr, light_adapt)
    end

    function TonemapReinhard:getColorAdaptation()
        return C.TonemapReinhard_getColorAdaptation(self.ptr)
    end

    function TonemapReinhard:setColorAdaptation(t)
        local argRules = {
            {"color_adapt", required = true}
        }
        local color_adapt = cv.argcheck(t, argRules)

        C.TonemapReinhard_setColorAdaptation(self.ptr, color_adapt)
    end    
end

-- TonemapMantiuk

do
    local TonemapMantiuk = torch.class('cv.TonemapMantiuk', 'cv.Tonemap', cv);

    function TonemapMantiuk:__init(t)
        local argRules = {
            {"gamma", default = 1.0},
            {"scale", default = 0.7},
            {"saturation", default = 1.0}
        }
        local gamma, scale, saturation = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.TonemapMantiuk_ctor(gamma, scale, saturation), Classes.Algorithm_dtor)
    end

    function TonemapMantiuk:getScale()
        return C.TonemapMantiuk_getScale(self.ptr)
    end

    function TonemapMantiuk:setScale(t)
        local argRules = {
            {"scale", required = true}
        }
        local scale = cv.argcheck(t, argRules)

        C.TonemapMantiuk_setScale(self.ptr, scale)
    end

    function TonemapMantiuk:getSaturation()
        return C.TonemapMantiuk_getSaturation(self.ptr)
    end

    function TonemapMantiuk:setSaturation(t)
        local argRules = {
            {"saturation", required = true}
        }
        local saturation = cv.argcheck(t, argRules)

        C.TonemapMantiuk_setSaturation(self.ptr, saturation)
    end
end

-- AlignExposures

do
    local AlignExposures = torch.class('cv.AlignExposures', 'cv.Algorithm', cv)

    function AlignExposures:process(t)
        local argRules = {
            {"src", required = true},
            {"dst", default = nil},
            {"times", required = true},
            {"response", required = true}
        }
        local src, dst, times, response = cv.argcheck(t, argRules)

        if type(times) == "table" then
            times = torch.FloatTensor(times)
        end

        return cv.unwrap_tensors(
                C.AlignExposures_process(
                    self.ptr, cv.wrap_tensors(src), cv.wrap_tensors(dst),
                    cv.wrap_tensor(times), cv.wrap_tensor(response)))
    end
end

-- AlignMTB
do
    local AlignMTB = torch.class('cv.AlignMTB', 'cv.AlignExposures', cv)

    function AlignMTB:__init(t)
        local argRules = {
            {"max_bits", default = 6},
            {"exclude_range", default = 4},
            {"cut", default = true}
        }

        local max_bits, exclude_range, cut = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.AlignMTB_ctor(max_bits, exclude_range, cut), Classes.Algorithm_dtor)
    end

    function AlignMTB:process(t)
        local argRules = {
            {"src", required = true},
            {"dst", default = nil},
            {"times", default = nil},
            {"response", default = nil}
        }
        local src, dst, times, response = cv.argcheck(t, argRules)

        if times == nil then
            return cv.unwrap_tensors(
                C.AlignMTB_process2(self.ptr, cv.wrap_tensors(src), cv.wrap_tensors(dst)))
        end

        if type(times) == "table" then
            times = torch.FloatTensor(times)
        end
        
        return cv.unwrap_tensors(
                C.AlignMTB_process1(
                    self.ptr, cv.wrap_tensors(src), cv.wrap_tensors(dst),
                    cv.wrap_tensor(times), cv.wrap_tensor(response)))
    end

    function AlignMTB:calculateShift(t)
        local argRules = {
            {"img0", required = true},
            {"img1", required = true}
        }

        local img0, img1 = cv.argcheck(t, argRules)
        
        resPoint = C.AlignMTB_calculateShift(self.ptr, cv.wrap_tensor(img0), cv.wrap_tensor(img1))
        return {resPoint.x, resPoint.y}
    end

    function AlignMTB:shiftMat(t)
        local argRules = {
            {"src", required = true},
            {"dst", default = nil},
            {"shift", required = true, operator = cv.Point}
        }

        local src, dst, shift = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(
            C.AlignMTB_shiftMat(self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(dst), shift))
    end

    function AlignMTB:computeBitmaps(t)
        local argRules = {
            {"img", required = true},
            {"tb", required = true},
            {"eb", required = true}
        }

        local img, tb, eb = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(
            C.AlignMTB_computeBitmaps(self.ptr, cv.wrap_tensor(img), cv.wrap_tensor(tb), cv.wrap_tensor(eb)))
    end

    function AlignMTB:getMaxBits()
        return C.AlignMTB_getMaxBits(self.ptr)
    end

    function AlignMTB:setMaxBits(t)
        local argRules = {
            {"max_bits", required = true}
        }
        local max_bits = cv.argcheck(t, argRules)

        C.AlignMTB_setMaxBits(self.ptr, max_bits)
    end

    function AlignMTB:getExcludeRange()
        return C.AlignMTB_getExcludeRange(self.ptr)
    end

    function AlignMTB:setExcludeRange(t)
        local argRules = {
            {"exclude_range", required = true}
        }
        local exclude_range = cv.argcheck(t, argRules)

        C.AlignMTB_setExcludeRange(self.ptr, exclude_range)
    end

    function AlignMTB:getCut()
        return C.AlignMTB_getCut(self.ptr)
    end

    function AlignMTB:setCut(t)
        local argRules = {
            {"cut", required = true}
        }
        local cut = cv.argcheck(t, argRules)
        
        C.AlignMTB_setCut(self.ptr, cut)
    end
end

-- CalibrateCRF

do
    local CalibrateCRF = torch.class('cv.CalibrateCRF', 'cv.Algorithm', cv)

    function CalibrateCRF:process(t)
        local argRules = {
            {"src", required = true},
            {"dst", default = nil},
            {"times", required = true}
        }

        local src, dst, times = cv.argcheck(t, argRules)

        if type(times) == "table" then
            times = torch.FloatTensor(times)
        end

        return cv.unwrap_tensors(
                C.CalibrateCRF_process(
                    self.ptr, cv.wrap_tensors(src), cv.wrap_tensor(dst), cv.wrap_tensor(times)))
    end
end

-- CalibrateDebevec

do
    local CalibrateDebevec = torch.class('cv.CalibrateDebevec', 'cv.CalibrateCRF', cv)

    function CalibrateDebevec:__init(t)
        local argRules = {
            {"samples", default = 70},
            {"lambda", default = 10.0},
            {"random", default = false}
        }

        local samples, lambda, random = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.CalibrateDebevec_ctor(samples, lambda, random), Classes.Algorithm_dtor)
    end

    function CalibrateDebevec:getLambda()
        return C.CalibrateDebevec_getLambda(self.ptr)
    end

    function CalibrateDebevec:setLambda(t)
        local argRules = {
            {"lambda", required = true}
        }
        local lambda = cv.argcheck(t, argRules)

        C.CalibrateDebevec_setLambda(self.ptr, lambda)
    end

    function CalibrateDebevec:getSamples()
        return C.CalibrateDebevec_getSamples(self.ptr)
    end

    function CalibrateDebevec:setSamples(t)
        local argRules = {
            {"samples", required = true}
        }
        local samples = cv.argcheck(t, argRules)

        C.CalibrateDebevec_setSamples(self.ptr, samples)
    end

    function CalibrateDebevec:getRandom()
        return C.CalibrateDebevec_getRandom(self.ptr)
    end

    function CalibrateDebevec:setRandom(t)
        local argRules = {
            {"random", required = true}
        }
        local random = cv.argcheck(t, argRules)

        C.CalibrateDebevec_setRandom(self.ptr, random)
    end
end

-- CalibrateRobertson

do
    local CalibrateRobertson = torch.class('cv.CalibrateRobertson', 'cv.CalibrateCRF', cv)

    function CalibrateRobertson:__init(t)
        local argRules = {
            {"max_iter", default = 30},
            {"threshold", default = 0.01}
        }
        
        local max_iter, threshold = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.CalibrateRobertson_ctor(max_iter, threshold), Classes.Algorithm_dtor)
    end

    function CalibrateRobertson:getMaxIter()
        return C.CalibrateRobertson_getMaxIter(self.ptr)
    end

    function CalibrateRobertson:setMaxIter(t)
        local argRules = {
            {"max_iter", required = true}
        }
        local max_iter = cv.argcheck(t, argRules)

        C.CalibrateRobertson_setMaxIter(self.ptr, max_iter)
    end

    function CalibrateRobertson:getThreshold()
        return C.CalibrateRobertson_getThreshold(self.ptr)
    end

    function CalibrateRobertson:setThreshold(t)
        local argRules = {
            {"threshold", required = true}
        }
        local threshold = cv.argcheck(t, argRules)

        C.CalibrateRobertson_setThreshold(self.ptr, threshold)
    end

    function CalibrateRobertson:getRadiance()
        return cv.unwrap_tensors(C.CalibrateRobertson_getRadiance(self.ptr))
    end
end

-- MergeExposures

do
    local MergeExposures = torch.class('cv.MergeExposures', 'cv.Algorithm', cv)

    function MergeExposures:process(t)
        local argRules = {
            {"src", required = true},
            {"dst", default = nil},
            {"times", required = true},
            {"response", required = true}
        }
        local src, dst, times, response = cv.argcheck(t, argRules)

        if type(times) == "table" then
            times = torch.FloatTensor(times)
        end

        return cv.unwrap_tensors(
                C.MergeExposures_process(
                    self.ptr, cv.wrap_tensors(src), cv.wrap_tensor(dst),
                    cv.wrap_tensor(times), cv.wrap_tensor(response)))
    end
end

-- MergeDebevec

do
    local MergeDebevec = torch.class('cv.MergeDebevec', 'cv.MergeExposures', cv)

    function MergeDebevec:__init(t)
        self.ptr = ffi.gc(C.MergeDebevec_ctor(), Classes.Algorithm_dtor)
    end

    function MergeDebevec:process(t)
        local argRules = {
            {"src", required = true},
            {"dst", default = nil},
            {"times", required = true},
            {"response", default = nil}
        }
        local src, dst, times, response = cv.argcheck(t, argRules)

        if type(times) == "table" then
            times = torch.FloatTensor(times)
        end
        
        if response == nil then
            return cv.unwrap_tensors(
                C.MergeDebevec_process2(self.ptr, cv.wrap_tensors(src), cv.wrap_tensor(dst), cv.wrap_tensor(times)))
        end

        return cv.unwrap_tensors(
                C.MergeDebevec_process1(
                    self.ptr, cv.wrap_tensors(src), cv.wrap_tensor(dst),
                    cv.wrap_tensor(times), cv.wrap_tensor(response)))
    end
end

-- MergeMertens

do
    local MergeMertens = torch.class('cv.MergeMertens', 'cv.MergeExposures', cv)

    function MergeMertens:__init(t)
        local argRules = {
            {"contrast_weight", default = 1.0},
            {"saturation_weight", default = 1.0},
            {"exposure_weight", default = 0.0}
        }
        local contrast_weight, saturation_weight, exposure_weight = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.MergeMertens_ctor(contrast_weight, saturation_weight, exposure_weight), Classes.Algorithm_dtor)
    end

    function MergeMertens:process(t)
        local argRules = {
            {"src", required = true},
            {"dst", default = nil},
            {"times", default = nil},
            {"response", default = nil}
        }
        local src, dst, times, response = cv.argcheck(t, argRules)

        if times == nil then
            return cv.unwrap_tensors(
                C.MergeMertens_process2(self.ptr, cv.wrap_tensors(src), cv.wrap_tensor(dst)))
        end

        if type(times) == "table" then
            times = torch.FloatTensor(times)
        end
        
        return cv.unwrap_tensors(
                C.MergeMertens_process1(
                    self.ptr, cv.wrap_tensors(src), cv.wrap_tensor(dst),
                    cv.wrap_tensor(times), cv.wrap_tensor(response)))
    end

    function MergeMertens:getContrastWeight()
        return C.MergeMertens_getContrastWeight(self.ptr)
    end

    function MergeMertens:setContrastWeight(t)
        local argRules = {
            {"contrast_weight", required = true}
        }
        local contrast_weight = cv.argcheck(t, argRules)

        C.MergeMertens_setContrastWeight(self.ptr, contrast_weight)
    end

    function MergeMertens:getSaturationWeight()
        return C.MergeMertens_getSaturationWeight(self.ptr)
    end

    function MergeMertens:setSaturationWeight(t)
        local argRules = {
            {"saturation_weight", required = true}
        }
        local saturation_weight = cv.argcheck(t, argRules)

        C.MergeMertens_setSaturationWeight(self.ptr, saturation_weight)
    end

    function MergeMertens:getExposureWeight()
        return C.MergeMertens_getExposureWeight(self.ptr)
    end

    function MergeMertens:setExposureWeight(t)
        local argRules = {
            {"exposure_weight", required = true}
        }
        local exposure_weight = cv.argcheck(t, argRules)

        C.MergeMertens_setExposureWeight(self.ptr, exposure_weight)
    end
end

-- MergeRobertson

do
    local MergeRobertson = torch.class('cv.MergeRobertson', 'cv.MergeExposures', cv)

    function MergeRobertson:__init(t)
        self.ptr = ffi.gc(C.MergeRobertson_ctor(), Classes.Algorithm_dtor)
    end

    function MergeRobertson:process(t)
        local argRules = {
            {"src", required = true},
            {"dst", default = nil},
            {"times", required = true},
            {"response", default = nil}
        }
        local src, dst, times, response = cv.argcheck(t, argRules)

        if type(times) == "table" then
            times = torch.FloatTensor(times)
        end
        
        if response == nil then
            return cv.unwrap_tensors(
                C.MergeRobertson_process2(self.ptr, cv.wrap_tensors(src), cv.wrap_tensor(dst), cv.wrap_tensor(times)))
        end

        return cv.unwrap_tensors(
                C.MergeRobertson_process1(
                    self.ptr, cv.wrap_tensors(src), cv.wrap_tensor(dst),
                    cv.wrap_tensor(times), cv.wrap_tensor(response)))
    end
end

return cv
