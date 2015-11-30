local cv = require 'cv._env'

local ffi = require 'ffi'

local C = ffi.load(cv.libPath('features2d'))

--- ***************** Classes *****************

require 'cv.Classes'

local Classes = ffi.load(cv.libPath('Classes'))

ffi.cdef[[
struct KeyPointWrapper {
    struct Point2fWrapper pt;
    float size, angle, response;
    int octave, class_id;
};

struct KeyPointArray {
    struct KeyPointWrapper *data;
    int size;
};

struct TensorPlusKeyPointArray {
    struct TensorWrapper tensor;
    struct KeyPointArray keypoints;
};

struct KeyPointMat {
    struct KeyPointWrapper **data;
    int size1, size2;
};

struct PtrWrapper KeyPointsFilter_ctor();

void KeyPointsFilter_dtor(struct PtrWrapper ptr);

struct KeyPointArray KeyPointsFilter_runByImageBorder(struct KeyPointArray keypoints,
                        struct SizeWrapper imageSize, int borderSize);

struct KeyPointArray KeyPointsFilter_runByKeypointSize(struct KeyPointArray keypoints, float minSize,
                        float maxSize);

struct KeyPointArray KeyPointsFilter_runByPixelsMask(struct KeyPointArray keypoints,
                        struct TensorWrapper mask);

struct KeyPointArray KeyPointsFilter_removeDuplicated(struct KeyPointArray keypoints);

struct KeyPointArray KeyPointsFilter_retainBest(struct KeyPointArray keypoints, int npoints);

struct KeyPointArray Feature2D_detect(struct PtrWrapper ptr, struct TensorWrapper image,
                        struct TensorWrapper mask);

struct TensorPlusKeyPointArray Feature2D_compute(struct PtrWrapper ptr, struct TensorWrapper image,
                        struct KeyPointArray keypoints, struct TensorWrapper descriptors);

struct KeyPointArray Feature2D_detectAndCompute(struct PtrWrapper ptr, struct TensorWrapper image,
                        struct TensorWrapper mask, struct TensorWrapper descriptors, 
                        bool useProvidedKeypoints);

int Feature2D_descriptorSize(struct PtrWrapper ptr);

int Feature2D_descriptorType(struct PtrWrapper ptr);

int Feature2D_defaultNorm(struct PtrWrapper ptr);

bool Feature2D_empty(struct PtrWrapper ptr);

struct PtrWrapper BRISK_ctor(int thresh, int octaves, float patternScale);

struct PtrWrapper BRISK_ctor2(struct TensorWrapper radiusList, struct TensorWrapper numberList,
                        float dMax, float dMin, struct TensorWrapper indexChange);

struct PtrWrapper ORB_ctor(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel,
                        int WTA_K, int scoreType, int patchSize, int fastThreshold);

void ORB_setMaxFeatures(struct PtrWrapper ptr, int maxFeatures);

int ORB_getMaxFeatures(struct PtrWrapper ptr);

void ORB_setScaleFactor(struct PtrWrapper ptr, int scaleFactor);

double ORB_getScaleFactor(struct PtrWrapper ptr);

void ORB_setNLevels(struct PtrWrapper ptr, int nlevels);

int ORB_getNLevels(struct PtrWrapper ptr);

void ORB_setEdgeThreshold(struct PtrWrapper ptr, int edgeThreshold);

int ORB_getEdgeThreshold(struct PtrWrapper ptr);

void ORB_setFirstLevel(struct PtrWrapper ptr, int firstLevel);

int ORB_getFirstLevel(struct PtrWrapper ptr);

void ORB_setWTA_K(struct PtrWrapper ptr, int wta_k);

int ORB_getWTA_K(struct PtrWrapper ptr);

void ORB_setScoreType(struct PtrWrapper ptr, int scoreType);

int ORB_getScoreType(struct PtrWrapper ptr);

void ORB_setPatchSize(struct PtrWrapper ptr, int patchSize);

int ORB_getPatchSize(struct PtrWrapper ptr);

void ORB_setFastThreshold(struct PtrWrapper ptr, int fastThreshold);

int ORB_getFastThreshold(struct PtrWrapper ptr);

struct PtrWrapper MSER_ctor(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity,
                        int _max_evolution, double _area_threshold, double _min_margin, int _edge_blur_size);

struct TensorArray MSER_detectRegions(struct PtrWrapper ptr,
        struct TensorWrapper image, struct TensorWrapper bboxes);

void MSER_setDelta(struct PtrWrapper ptr, int delta);

int MSER_getDelta(struct PtrWrapper ptr);

void MSER_setMinArea(struct PtrWrapper ptr, int minArea);

int MSER_getMinArea(struct PtrWrapper ptr);

void MSER_setMaxArea(struct PtrWrapper ptr, int MaxArea);

int MSER_getMaxArea(struct PtrWrapper ptr);




struct KeyPointArray AGAST(
        struct TensorWrapper image, int threshold, bool nonmaxSuppression);


void BOWTrainer_dtor(struct PtrWrapper ptr);

void BOWTrainer_add(struct PtrWrapper ptr, struct TensorWrapper descriptors);

struct TensorArray BOWTrainer_getDescriptors(struct PtrWrapper ptr);

int BOWTrainer_descriptorsCount(struct PtrWrapper ptr);

void BOWTrainer_clear(struct PtrWrapper ptr);

struct TensorWrapper BOWTrainer_cluster(struct PtrWrapper ptr);

struct TensorWrapper BOWTrainer_cluster_descriptors(struct PtrWrapper ptr, struct TensorWrapper descriptors);

struct PtrWrapper BOWKMeansTrainer_ctor(
        int clusterCount, struct TermCriteriaWrapper termcrit,
        int attempts, int flags);

struct PtrWrapper BOWImgDescriptorExtractor_ctor(
        struct PtrWrapper dextractor, struct PtrWrapper dmatcher);

void BOWImgDescriptorExtractor_dtor(struct PtrWrapper ptr);

void BOWImgDescriptorExtractor_setVocabulary(
        struct PtrWrapper ptr, struct TensorWrapper vocabulary);

struct TensorWrapper getVocabulary(struct PtrWrapper ptr);

struct TensorWrapper compute(
        struct PtrWrapper ptr, struct TensorWrapper image,
        struct KeyPointArray keypoints, struct TensorWrapper imgDescriptor);

int descriptorSize(struct PtrWrapper ptr);

int descriptorType(struct PtrWrapper ptr);

]]

function cv.KeyPoint(...)
    return ffi.new('struct KeyPointWrapper', ...)
end

-- KeyPointsFilter

do
    local KeyPointsFilter = torch.class('cv.KeyPointsFilter', cv)

    function KeyPointsFilter:__init()
        self.ptr = ffi.gc(C.KeyPointsFilter_ctor(), C.KeyPointsFilter_dtor)
    end

    function KeyPointsFilter:runByImageBorder(t)
        local argRules = {
            {"keypoints", required = true},
            {"imageSize", required = true, operator = cv.Size},
            {"borderSize", required = true}
        }
        local keypoints, imageSize, borderSize = cv.argcheck(t, argRules)
        
        return cv.gcarray(
            C.KeyPointsFilter_runByImageBorder(keypoints, imageSize, borderSize))
    end

    function KeyPointsFilter:runByKeypointSize(t)
        local argRules = {
            {"keypoints", required = true},
            {"minSize", required = true},
            {"maxSize", default = cv.FLT_MAX}
        }
        local keypoints, minSize, maxSize = cv.argcheck(t, argRules)

        return cv.gcarray(
            C.KeyPointsFilter_runByKeypointSize(keypoints, imageSize, borderSize))
    end

    function KeyPointsFilter:runByPixelsMask(t)
        local argRules = {
            {"keypoints", required = true},
            {"mask", required = true}
        }
        local keypoints, mask = cv.argcheck(t, argRules)

        return cv.gcarray(
            C.KeyPointsFilter_runByPixelsMask(keypoints, cv.wrap_tensor(mask)))
    end

    function KeyPointsFilter:removeDuplicated(t)
        local argRules = {
            {"keypoints", required = true}
        }
        local keypoints = cv.argcheck(t, argRules)

        return cv.gcarray(
            C.KeyPointsFilter_removeDuplicated(keypoints))
    end

    function KeyPointsFilter:retainBest(t)
        local argRules = {
            {"keypoints", required = true},
            {"npoints", required = true}
        }
        local keypoints, npoints = cv.argcheck(t, argRules)

        return cv.gcarray(
            C.KeyPointsFilter_retainBest(keypoints, npoints))
    end
end

-- Feature2D

do
    local Feature2D = torch.class('cv.Feature2D', 'cv.Algorithm', cv)

    function Feature2D:__init()
        self.ptr = ffi.gc(C.Feature2D_ctor(), Classes.Algorithm_dtor)
    end

    function Feature2D:detect(t)
        local argRules = {
            {"image", required = true},
            {"mask", default = nil}
        }
        local image, mask = cv.argcheck(t, argRules)

        return C.Feature2D_detect(self.ptr, cv.wrap_tensor(image), cv.wrap_tensor(mask))
    end

    function Feature2D:compute(t)
        local argRules = {
            {"image", required = true},
            {"keypoints", required = true},
            {"descriptors", default = nil}
        }
        local image, keypoints, descriptors = cv.argcheck(t, argRules)

        local result = C.Feature2D_compute(
            self.ptr, cv.wrap_tensor(image), keypoints, cv.wrap_tensor(descriptors))
        return result.keypoints, cv.unwrap_tensors(result.tensor)
    end

    function Feature2D:detectAndCompute(t)
        local argRules = {
            {"image", required = true},
            {"mask", default = nil},
            {"descriptors", default = nil},
            {"useProvidedKeypoints", default = false}
        }
        local image, mask, descriptors, useProvidedKeypoints = cv.argcheck(t, argRules)

        local result = C.Feature2D_detectAndCompute(self.ptr, cv.wrap_tensors(image), cv.wrap_tensors(mask),
                    cv.wrap_tensors(descriptors), useProvidedKeypoints)
        return result.keypoints, cv.unwrap_tensors(result.tensor)
    end

    function Feature2D:descriptorSize()
        return C.Feature2D_descriptorSize(self.ptr)
    end

    function Feature2D:descriptorType()
        return C.Feature2D_descriptorType(self.ptr)
    end

    function Feature2D:defaultNorm()
        return C.Feature2D_defaultNorm(self.ptr)
    end

    function Feature2D:empty()
        return C.Feature2D_empty(self.ptr)
    end
end

-- BRISK

do
    local BRISK = torch.class('cv.BRISK', 'cv.Feature2D', cv)

    function BRISK:__init(t)
        if not (t.radiusList or t[1]) or type(t[1]) == "number" then
            local argRules = {
                {"thresh", default = 30},
                {"octaves", default = 3},
                {"patternScale", default = 1.0}
            }
            local thresh, octaves, patternScale = cv.argcheck(t, argRules)

            self.ptr =  ffi.gc(C.BRISK_ctor(thresh, octaves, patternScale), Classes.Algorithm_dtor)
        else
            local argRules = {
                {"radiusList", required = true},
                {"numberList", required = true},
                {"dMax", default = 5.85},
                {"dMin", default = 8.2},
                {"indexChange", default = nil}
            }
            local radiusList, numberList, dMax, dMin, indexChange = cv.argcheck(t, argRules)

            if type(radiusList) == "table" then
                radiusList = torch.FloatTensor(radiusList)
            end

            if type(numberList) == "table" then
                numberList = torch.IntTensor(numberList)
            end

            if type(indexChange) == "table" then
                indexChange = torch.IntTensor(indexChange)
            end

            self.ptr =  ffi.gc(C.BRISK_ctor2(cv.wrap_tensor(radiusList), cv.wrap_tensor(numberList),
                                    dMax, dMin, cv.wrap_tensor(indexChange)), Classes.Algorithm_dtor)
        end
    end
end

-- ORB

do
    local ORB = torch.class('cv.ORB', 'cv.Feature2D', cv)

    function ORB:__init(t)
        local argRules = {
            {"nfeatures", default = 500},
            {"scaleFactor", default = 1.2},
            {"nlevels", default = 8},
            {"edgeThreshold", default = 31},
            {"firstLevel", default = 0},
            {"WTA_K", default = 2},
            {"scoreType", default = cv.ORB_HARRIS_SCORE},
            {"patchSize", default = 31},
            {"fastThreshold", default = 20}
        }
        local nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, 
            scoreType, patchSize, fastThreshold = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.ORB_ctor(nfeatures, scaleFactor, nlevels, edgeThreshold,
                                firstLevel, WTA_K, scoreType, patchSize, fastThreshold),
                                Classes.Algorithm_dtor)
    end

    function ORB:setMaxFeatures(t)
        local argRules = {
            {"maxFeatures", required = true}
        }
        local maxFeatures = cv.argcheck(t, argRules)

        C.ORB_setMaxFeatures(self.ptr, maxFeatures)
    end

    function ORB:getMaxFeatures()
        return C.ORB_getMaxFeatures(self.ptr)
    end

    function ORB:setScaleFactor(t)
        local argRules = {
            {"scaleFactor", required = true}
        }
        local scaleFactor = cv.argcheck(t, argRules)

        C.ORB_setScaleFactor(self.ptr, scaleFactor)
    end

    function ORB:getScaleFactor()
        return C.ORB_getScaleFactor(self.ptr)
    end

    function ORB:setNLevels(t)
        local argRules = {
            {"nlevels", required = true}
        }
        local nlevels = cv.argcheck(t, argRules)

        C.ORB_setNLevels(self.ptr, nlevels)
    end

    function ORB:getNLevels()
        return C.ORB_getNLevels(self.ptr)
    end

    function ORB:setEdgeThreshold(t)
        local argRules = {
            {"edgeThreshold", required = true}
        }
        local edgeThreshold = cv.argcheck(t, argRules)

        C.ORB_setEdgeThreshold(self.ptr, edgeThreshold)
    end

    function ORB:getEdgeThreshold()
        return C.ORB_getEdgeThreshold(self.ptr)
    end

    function ORB:setFirstLevel(t)
        local argRules = {
            {"firstLevel", required = true}
        }
        local firstLevel = cv.argcheck(t, argRules)

        C.ORB_setFirstLevel(self.ptr, firstLevel)
    end

    function ORB:getFirstLevel()
        return C.ORB_getFirstLevel(self.ptr)
    end

    function ORB:setWTA_K(t)
        local argRules = {
            {"wta_k", required = true}
        }
        local wta_k = cv.argcheck(t, argRules)

        C.ORB_setWTA_K(self.ptr, wta_k)
    end

    function ORB:getWTA_K()
        return C.ORB_getWTA_K(self.ptr)
    end

    function ORB:setScoreType(t)
        local argRules = {
            {"scoreType", required = true}
        }
        local scoreType = cv.argcheck(t, argRules)

        C.ORB_setScoreType(self.ptr, scoreType)
    end

    function ORB:getScoreType()
        return C.ORB_getScoreType(self.ptr)
    end

    function ORB:setPatchSize(t)
        local argRules = {
            {"patchSize", required = true}
        }
        local patchSize = cv.argcheck(t, argRules)

        C.ORB_setPatchSize(self.ptr, patchSize)
    end

    function ORB:getPatchSize()
        return C.ORB_getPatchSize(self.ptr)
    end

    function ORB:setFastThreshold(t)
        local argRules = {
            {"fastThreshold", required = true}
        }
        local fastThreshold = cv.argcheck(t, argRules)

        C.ORB_setFastThreshold(self.ptr, fastThreshold)
    end

    function ORB:getFastThreshold()
        return C.ORB_getFastThreshold(self.ptr)
    end
end

-- MSER

do 
    local MSER = torch.class('cv.MSER', 'cv.Feature2D', cv)

    function MSER:__init(t)
        local argRules = {
            {"_delta", default = 5},
            {"_min_area", default = 60},
            {"_max_area", default = 14400},
            {"_max_variation", default = 0.25},
            {"_min_diversity", default = 0.2},
            {"_max_evolution", default = 200},
            {"_area_threshold", default = 1.01},
            {"_min_margin", default = 0.003},
            {"_edge_blur_size", default = 5}   
        }
        local _delta, _min_area, _max_area, _max_variation, _min_diversity, _max_evolution,
                _area_threshold, _min_margin, _edge_blur_size = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.MSER_ctor(_delta, _min_area, _max_area, _max_variation, _min_diversity, _max_evolution,
                                _area_threshold, _min_margin, _edge_blur_size),
                                Classes.Algorithm_dtor)    
    end

    function MSER:detectRegions(t)
        local argRules = {
            {"image", required = true},
            {"bboxes", default = nil}
        }
        local image, bboxes = cv.argcheck(t, argRules)

        if bboxes ~= nil then
            assert(torch.isTensor(bboxes) and cv.tensorType(bboxes) == cv.CV_32S)
            assert(bboxes:size()[2] == 4)
        end

        local result = C.MSER_detectRegions(self.ptr, 
            cv.wrap_tensor(image), cv.wrap_tensor(bboxes))

        result = cv.unwrap_tensors(result, true)
        local msers = result[#result]
        result[#result] = nil

        return msers, result
    end

    function MSER:setDelta(t)
        local argRules = {
            {"delta", required = true}
        }
        local delta = cv.argcheck(t, argRules)

        C.MSER_setDelta(self.ptr, delta)
    end

    function MSER:getDelta()
        return C.MSER_getDelta(self.ptr)
    end

    function MSER:setMinArea(t)
        local argRules = {
            {"minArea", required = true}
        }
        local minArea = cv.argcheck(t, argRules)

        C.MSER_setMinArea(self.ptr, minArea)
    end

    function MSER:getMinArea()
        return C.MSER_getMinArea(self.ptr)
    end

    function MSER:setMaxArea(t)
        local argRules = {
            {"MaxArea", required = true}
        }
        local MaxArea = cv.argcheck(t, argRules)

        C.MSER_setMaxArea(self.ptr, MaxArea)
    end

    function MSER:getMaxArea()
        return C.MSER_getMaxArea(self.ptr)
    end
end


function cv.FAST(t)
    local argRules = {
        {"image", required = true},
        {"threshold", required = true},
        {"nonmaxSuppression", default = true},
        {"_type", default = nil}
    }
    local image, threshold, nonmaxSuppression, _type = cv.argcheck(t, argRules)

    if _type then
        return C.FAST_type(cv.wrap_tensor(image), threshold, nonmaxSuppression, type)
    else
        return C.FAST(cv.wrap_tensor(image), threshold, nonmaxSuppression)
    end
end

function cv.AGAST(t)
    local argRules = {
        {"image", required = true},
        {"threshold", required = true},
        {"nonmaxSuppression", default = true}
    }
    local image, threshold, nonmaxSuppression = cv.argcheck(t, argRules)

    return C.AGAST(cv.wrap_tensor(image), threshold, nonmaxSuppression)
end

-- FastFeatureDetector

do
    local FastFeatureDetector = torch.class('cv.FastFeatureDetector', 'cv.Feature2D', cv)

    function FastFeatureDetector:__init(t)
        local argRules = {
            {"threshold", default = 10},
            {"nonmaxSuppression", default = true},
            {"_type", default = cv.FAST_FEATURE_DETECTOR_TYPE_9_16}
        }
        local threshold, nonmaxSuppression, _type = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(
            C.FastFeatureDetector_ctor(threshold, nonmaxSuppression, _type),
            C.Algorithm_dtor)
    end

    function FastFeatureDetector:setThreshold(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FastFeatureDetector_setThreshold(self.ptr, val)
    end

    function FastFeatureDetector:getThreshold()
        return C.FastFeatureDetector_getThreshold(self.ptr)
    end

    function FastFeatureDetector:setNonmaxSuppression(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FastFeatureDetector_setNonmaxSuppression(self.ptr, val)
    end

    function FastFeatureDetector:getNonmaxSuppression()
        return C.FastFeatureDetector_getNonmaxSuppression(self.ptr)
    end

    function FastFeatureDetector:setType(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.FastFeatureDetector_setType(self.ptr, val)
    end

    function FastFeatureDetector:getType()
        return C.FastFeatureDetector_getType(self.ptr)
    end
end

-- AgastFeatureDetector

do
    local AgastFeatureDetector = torch.class('cv.AgastFeatureDetector', 'cv.Feature2D', cv)

    function AgastFeatureDetector:__init(t)
        local argRules = {
            {"threshold", default = 10},
            {"nonmaxSuppression", default = true},
            {"_type", default = cv.AGAST_FEATURE_DETECTOR_OAST_9_16}
        }
        local threshold, nonmaxSuppression, _type = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(
            C.AgastFeatureDetector_ctor(threshold, nonmaxSuppression, _type),
            C.Algorithm_dtor)
    end

    function AgastFeatureDetector:setThreshold(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.AgastFeatureDetector_setThreshold(self.ptr, val)
    end

    function AgastFeatureDetector:getThreshold()
        return C.AgastFeatureDetector_getThreshold(self.ptr)
    end

    function AgastFeatureDetector:setNonmaxSuppression(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.AgastFeatureDetector_setNonmaxSuppression(self.ptr, val)
    end

    function AgastFeatureDetector:getNonmaxSuppression()
        return C.AgastFeatureDetector_getNonmaxSuppression(self.ptr)
    end

    function AgastFeatureDetector:setType(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.AgastFeatureDetector_setType(self.ptr, val)
    end

    function AgastFeatureDetector:getType()
        return C.AgastFeatureDetector_getType(self.ptr)
    end
end

-- GFTTDetector

do
    local GFTTDetector = torch.class('cv.GFTTDetector', 'cv.Feature2D', cv)

    function GFTTDetector:__init(t)
        local argRules = {
            {"maxCorners", default = 1000},
            {"qualityLevel", default = 0.01},
            {"minDistance", default = 1},
            {"blockSize", default = 3},
            {"useHarrisDetector", default = false},
            {"k", default = 0.04}
        }
        local maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, k = 
            cv.argcheck(t, argRules)
        self.ptr = ffi.gc(
            C.GFTTDetector_ctor(maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, k),
            Classes.Algorithm_dtor)
    end

    function GFTTDetector:setMaxFeatures(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.GFTTDetector_setMaxFeatures(self.ptr, val)
    end

    function GFTTDetector:getMaxFeatures()
        return C.GFTTDetector_getMaxFeatures(self.ptr)
    end

    function GFTTDetector:setQualityLevel(t)

        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.GFTTDetector_setQualityLevel(self.ptr, val)
    end

    function GFTTDetector:getQualityLevel()
        return C.GFTTDetector_getQualityLevel(self.ptr)
    end

    function GFTTDetector:setMinDistance(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.GFTTDetector_setMinDistance(self.ptr, val)
    end

    function GFTTDetector:getMinDistance()
        return C.GFTTDetector_getMinDistance(self.ptr)
    end

    function GFTTDetector:setBlockSize(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.GFTTDetector_setBlockSize(self.ptr, val)
    end

    function GFTTDetector:getBlockSize()
        return C.GFTTDetector_getBlockSize(self.ptr)
    end

    function GFTTDetector:setHarrisDetector(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.GFTTDetector_setHarrisDetector(self.ptr, val)
    end

    function GFTTDetector:getHarrisDetector()
        return C.GFTTDetector_getHarrisDetector(self.ptr)
    end

    function GFTTDetector:setK(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.GFTTDetector_setK(self.ptr, val)
    end

    function GFTTDetector:getK()
        return C.GFTTDetector_getK(self.ptr)
    end
end

-- SimpleBlobDetector

function cv.SimpleBlobDetector_Params(t)
    local argRules = {
        {"params", default = {}}
    }
    local params = cv.argcheck(t, argRules)

    local result = C.SimpleBlobDetector_Params_default()
    for k, v in pairs(params) do
        result[k] = v
    end
end


do
    local SimpleBlobDetector = torch.class('cv.SimpleBlobDetector', 'cv.Feature2D', cv)

    function SimpleBlobDetector:__init(t)
        local argRules = {
            {"params", default = nil}
        }
        local params = cv.argcheck(t, argRules)

        params = params or cv.SimpleBlobDetector_Params{}
        self.ptr = C.SimpleBlobDetector_ctor(params)
    end
end

-- KAZE

do
    local KAZE = torch.class('cv.KAZE', 'cv.Feature2D', cv)

    function KAZE:__init(t)
        local argRules = {
            {"extended", default = false},
            {"upright", default = false},
            {"threshold", default = 0.001},
            {"nOctaves", default = 4},
            {"nOctaveLayers", default = 4},
            {"diffusivity", default = cv.KAZE_DIFF_PM_G2}
        }
        local extended, upright, threshold, nOctaves, nOctaveLayers, diffusivity = 
            cv.argcheck(t, argRules)

        self.ptr = ffi.gc(
            C.KAZE_ctor(extended, upright, threshold, nOctaves, nOctaveLayers, diffusivity),
            C.Algorithm_dtor)
    end

    function KAZE:setExtended(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.KAZE_setExtended(self.ptr, val)
    end

    function KAZE:getExtended()
        return C.KAZE_getExtended(self.ptr)
    end

    function KAZE:setUpright(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.KAZE_setUpright(self.ptr, val)
    end

    function KAZE:getUpright()
        return C.KAZE_getUpright(self.ptr)
    end

    function KAZE:setThreshold(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.KAZE_setThreshold(self.ptr, val)
    end

    function KAZE:getThreshold()
        return C.KAZE_getThreshold(self.ptr)
    end

    function KAZE:setNOctaves(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.KAZE_setNOctaves(self.ptr, val)
    end

    function KAZE:getNOctaves()
        return C.KAZE_getNOctaves(self.ptr)
    end

    function KAZE:setNOctaveLayers(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.KAZE_setNOctaveLayers(self.ptr, val)
    end

    function KAZE:getNOctaveLayers()
        return C.KAZE_getNOctaveLayers(self.ptr)
    end

    function KAZE:setDiffusivity(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.KAZE_setDiffusivity(self.ptr, val)
    end

    function KAZE:getDiffusivity()
        return C.KAZE_getDiffusivity(self.ptr)
    end
end

-- AKAZE

do
    local AKAZE = torch.class('cv.AKAZE', 'cv.Feature2D', cv)

    function AKAZE:__init(t)
        local argRules = {
            {"descriptor_type", default = cv.AKAZE_DESCRIPTOR_MLDB},
            {"descriptor_size", default = 0},
            {"descriptor_channels", default = 3},
            {"threshold", default = 0.001},
            {"nOctaves", default = 4},
            {"nOctaveLayers", default = 4},
            {"diffusivity", default = cv.KAZE_DIFF_PM_G2}
        }
        local descriptor_type, descriptor_size, descriptor_channels, 
            threshold, nOctaves, nOctaveLayers, diffusivity = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(
            C.AKAZE_ctor(descriptor_type, descriptor_size, descriptor_channels, 
                threshold, nOctaves, nOctaveLayers, diffusivity),
            C.Algorithm_dtor)
    end

    function AKAZE:setDescriptorType(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.AKAZE_setDescriptorType(self.ptr, val)
    end

    function AKAZE:getDescriptorType()
        return C.AKAZE_getDescriptorType(self.ptr)
    end

    function AKAZE:setDescriptorSize(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.AKAZE_setDescriptorSize(self.ptr, val)
    end

    function AKAZE:getDescriptorSize()
        return C.AKAZE_getDescriptorSize(self.ptr)
    end

    function AKAZE:setDescriptorChannels(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.AKAZE_setDescriptorChannels(self.ptr, val)
    end

    function AKAZE:getDescriptorChannels()
        return C.AKAZE_getDescriptorChannels(self.ptr)
    end

    function AKAZE:setThreshold(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.AKAZE_setThreshold(self.ptr, val)
    end

    function AKAZE:getThreshold()
        return C.AKAZE_getThreshold(self.ptr)
    end

    function AKAZE:setNOctaves(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.AKAZE_setNOctaves(self.ptr, val)
    end

    function AKAZE:getNOctaves()
        return C.AKAZE_getNOctaves(self.ptr)
    end

    function AKAZE:setNOctaveLayers(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.AKAZE_setNOctaveLayers(self.ptr, val)
    end

    function AKAZE:getNOctaveLayers()
        return C.AKAZE_getNOctaveLayers(self.ptr)
    end

    function AKAZE:setDiffusivity(t)
        local argRules = {
            {"val", required = true}
        }
        local val = cv.argcheck(t, argRules)
        
        C.AKAZE_setDiffusivity(self.ptr, val)
    end

    function AKAZE:getDiffusivity()
        return C.AKAZE_getDiffusivity(self.ptr)
    end
end

-- BOWTrainer

do
    local BOWTrainer = torch.class('cv.BOWTrainer', cv)

    function BOWTrainer:add(t)
        local argRules = {
            {"descriptors", required = true}
        }
        local descriptors = cv.argcheck(t, argRules)

        C.BOWTrainer_add(self.ptr, cv.wrap_tensor(descriptors))
    end

    function BOWTrainer:getDescriptors(t)
        return cv.unwrap_tensors(C.BOWTrainer_getDescriptors(self.ptr))
    end

    function BOWTrainer:descriptorsCount(t)
        return C.BOWTrainer_descriptorsCount(self.ptr)
    end

    function BOWTrainer:clear(t)
        C.BOWTrainer_clear(self.ptr)
    end

    function BOWTrainer:cluster(t)
        if t[1] or t.descriptors then
            local argRules = {
                {"descriptors", required = true}
            }
            local descriptors = cv.argcheck(t, argRules)

            return cv.unwrap_tensors(C.BOWTrainer_cluster_descriptors(
                self.ptr, cv.wrap_tensor(descriptors)))
        else
            return cv.unwrap_tensors(C.BOWTrainer_cluster(self.ptr))
        end
    end
end

-- 

-- BOWKMeansTrainer

do
    local BOWKMeansTrainer = torch.class('cv.BOWKMeansTrainer', 'cv.BOWTrainer', cv)

    function BOWKMeansTrainer:__init(t)
        local argRules = {
            {"clusterCount", required = true},
            {"termcrit", default = 0, operator = cv.TermCriteria},
            {"attempts", default = 3},
            {"flags", default = cv.KMEANS_PP_CENTERS}
        }
        local clusterCount, termcrit, attempts, flags = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(
            C.BOWKMeansTrainer_ctor(clusterCount, termcrit, attempts, flags),
            C.BOWTrainer_dtor)
    end
end

-- BOWImgDescriptorExtractor

do
    local BOWImgDescriptorExtractor = torch.class('cv.BOWImgDescriptorExtractor', cv)

    function BOWImgDescriptorExtractor:__init(t)
        local argRules = {
            {"dextractor", required = true},
            {"dmatcher", required = true}
        }
        local dextractor, dmatcher = cv.argcheck(t, argRules)

        self.dextractor = dextractor
        self.dmatcher = dmatcher

        self.ptr = ffi.gc(
            C.BOWImgDescriptorExtractor_ctor(dextractor.ptr, dmatcher.ptr),
            C.BOWImgDescriptorExtractor_dtor)
    end

    function BOWImgDescriptorExtractor:setVocabulary(t)
        local argRules = {
            {"vocabulary", required = true}
        }
        local vocabulary = cv.argcheck(t, argRules)

        C.BOWImgDescriptorExtractor_setVocabulary(self.ptr, cv.wrap_tensor(vocabulary))
    end

    function BOWImgDescriptorExtractor:getVocabulary(t)
        return cv.unwrap_tensors(C.BOWImgDescriptorExtractor_getVocabulary(self.ptr))
    end

    function BOWImgDescriptorExtractor:compute(t)
        local argRules = {
            {"image", required = true},
            {"keypoints", required = true},
            {"imgDescriptor", default = nil}
        }
        local image, keypoints, imgDescriptor = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.BOWImgDescriptorExtractor_compute(
            self.ptr, cv.wrap_tensor(image), keypoints, cv.wrap_tensor(imgDescriptor)))
    end

    function BOWImgDescriptorExtractor:descriptorSize()
        return C.BOWImgDescriptorExtractor_descriptorSize(self.ptr)
    end

    function BOWImgDescriptorExtractor:descriptorType()
        return C.BOWImgDescriptorExtractor_descriptorType(self.ptr)
    end
end


return cv