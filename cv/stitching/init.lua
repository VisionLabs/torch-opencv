local cv = require 'cv._env'
local ffi = require 'ffi'

ffi.cdef[[
struct RectPlusBool detail_overlapRoi(
	struct PointWrapper tl1, struct PointWrapper tl2,
	struct SizeWrapper sz1, struct SizeWrapper sz2);

struct RectWrapper detail_resultRoi(
	struct PointArray corners,
	struct SizeArray sizes);

struct RectWrapper detail_resultRoiIntersection(
	struct PointArray corners,
	struct SizeArray sizes);

struct PointWrapper detail_resultTl(
	struct PointArray corners);

void detail_selectRandomSubset(
	int count, int size);

int detail_stitchingLogLevel()
]]

local C = ffi.load(cv.libPath('stitching'))

detail = {}
cv.detail = detail;

function cv.detail.overlapRoi(t)
    local argRules = {
        {"tl1", required = true},
        {"tl2", required = true},
        {"sz1", required = true},
        {"sz2", required = true}}
    local tl1, tl2, sz1, sz2 = cv.argcheck(t, argRules)
    local result = C.detail_overlapRoi(tl1, tl2, sz1, sz2)
    return result.val, result.rect
end

function cv.detail.resultRoi(t)
    local argRules = {
        {"corners", required = true},
        {"sizes", required = true}}
    local corners, sizes = cv.argcheck(corners, sizes)
    return C.detail_resultRoi(corners, sizes)
end

function cv.detail.resultRoiIntersection(t)
    local argRules = {
        {"corners", required = true},
        {"sizes", required = true}}
    local corners, sizes = cv.argcheck(corners, sizes)
    return C.detail_resultRoiIntersection(corners, sizes)
end

function cv.detail.resultTl(t)
    local argRules = {
        {"corners", required = true}}
    local corners = cv.argcheck(t, argRules)
    return C.detail_resultTl(corners)
end

function cv.detail.selectRandomSubset(t)
    local argRules = {
        {"count", required = true},
        {"size", required = true}}
    local count, size = cv.argcheck(t, argRules)
    return cv.gcarray(C.detail_selectRandomSubset(count, size))
end

function cv.detail.stitchingLogLevel(t)
    C.detail.stitchingLogLevel()
end

--- ***************** Classes *****************

require 'cv.Classes'

local Classes = ffi.load(cv.libPath('Classes'))

ffi.cdef[[
struct PtrWrapper CameraParams_ctor();

struct PtrWrapper CameraParams_ctor2(
	struct CameraParamsPtr other);

void CameraParams_dtor(
	struct PtrWrapper ptr);

struct TensorWrapper CameraParams_K(
	struct PtrWrapper ptr);

struct PtrWrapper DisjointSets_ctor(
	int elem_count);

void DisjointSets_dtor(
	struct PtrWrapper ptr);

void DisjointSets_createOneElemSets(
	struct PtrWrapper ptr, int elem_count);

int DisjointSets_findSetByElem(
	struct PtrWrapper ptr, int elem);

int DisjointSets_mergeSets(
	struct PtrWrapper ptr,
	int set1, int set2);

struct PtrWrapper Graph_ctor(
	int num_vertices);

void Graph_dtor(
	struct PtrWrapper ptr)

void Graph_addEdge(
	struct PtrWrapper ptr, int from, int to, float weight);

void Graph_create(
	struct PtrWrapper ptr, int num_vertices);

int Graph_numVertices(
	struct PtrWrapper ptr);

struct PtrWrapper GraphEdge_ctor(
	int from, int to, float weight);

void GraphEdge_dtor(
	struct PtrWrapper ptr);

struct PtrWrapper Timelapser_ctor(
	int type)

void Timelapser_dtor(
	struct PtrWrapper ptr);

void Timelapser_initialize(
	struct PtrWrapper ptr, struct PointArray corners,
	struct SizeArray sizes);

void Timelapser_process(
	struct PtrWrapper ptr, struct TensorWrapper img,
	struct TensorWrapper mask, struct PointWrapper tl);

void TimelapserCrop_initialize(
	struct PtrWrapper ptr, struct PointArray corners,
	struct SizeArray sizes);

void FeaturesFinder_dtor(
	struct PtrWrapper ptr)

void FeaturesFinder_collectGarbage(
	struct PtrWrapper ptr);

struct PtrWrapper FeaturesFinder_call(
	struct PtrWrapper ptr, struct TensorWrapper image);

struct PtrWrapper FeaturesFinder_call2(
	struct PtrWrapper ptr, struct TensorWrapper image,
	struct RectArray);

struct PtrWrapper ImageFeatures_ctor();

struct PtrWrapper ImageFeatures_dtor(
	struct PtrWrapper ptr);

struct PtrWrapper BestOf2NearestMatcher_ctor(
    bool try_use_gpu, float match_conf,
    int num_matches_thresh1, int num_matches_thresh2);

void BestOf2NearestMatcher_collectGarbage(
    struct PtrWrapper ptr);

void FeaturesMatcher_dtor(
    struct PtrWrapper ptr);

void FeaturesMatcher_collectGarbage(
    struct PtrWrapper ptr)

struct PtrWrapper FeaturesMatcher_call(
        struct PtrWrapper ptr, struct PtrWrapper features1,
        struct PtrWrapper features2);

bool FeaturesMatcher_isThreadSafe(
        struct PtrWrapper ptr);

struct MatchesInfoPtr MatchesInfo_ctor();

struct MatchesInfoPtr MatchesInfo_ctor2(
        struct PtrWrapper other);

struct PtrWrapper BestOf2NearestRangeMatcher_ctor(
		int range_width, bool try_use_gpu, float match_conf,
		int num_matches_thresh1, int num_matches_thresh2);

struct OrbFeaturesFinderPtr OrbFeaturesFinder_ctor(
        struct SizeWrapper _grid_size, int nfeatures, float scaleFactor, int nlevels);

struct SurfFeaturesFinderPtr SurfFeaturesFinder_ctor(
        double hess_thresh, int num_octaves, int num_layers, int num_octaves_descr, int num_layers_descr);
]]

--CameraParams

--TODO
do
    local CameraParams = torch.class('cv.CameraParams', cv);

    function CameraParams:__init(t)
        local argRules = {
            {"other", default = nil}}
        local other = cv.argcheck(t, argRules)
        if other then 
            self.ptr = ffi.gc(C.CameraParams_ctor(), C.CameraParams_dtor)
        else
            self.ptr = ffi.gc(C.CameraParams_ctor2(other.ptr), C.CameraParams_dtor)
        end
    end

    function CameraParams:K()
        return cv.unwrap_tensor(C.CameraParams_K(self.ptr))
    end
end

--DisjointSets

do
    local DisjointSets = torch.class('cv.DisjointSets', cv)

    function DisjointSets:__init(t)
        local argRules = {
            {"elem_count", default = 0}}
        local elem_count = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.DisjointSets_ctor(elem_count), C.DisjointSets_dtor)
    end

    function DisjointSets:createOneElemSets(t)
        local argRules = {
            {"elem_count", default = 0}}
        local elem_count = cv.argcheck(t, argRules)
        C.DisjointSets_createOneElemSets(self.ptr, elem_count)
    end

    function DisjointSets:findSetByElem(t)
        local argRules = {
            {"elem", required = true}}
        local elem = cv.argcheck(t, argRules)
        return C.findSetByElem(self.ptr, elem)
    end

    function DisjointSets:mergeSets(t)
        local argRules = {
            {"set1", required = true},
            {"set2", required = true}}
        local set1, set2 = cv.argcheck(t, argRules)
        C.DisjointSets_mergeSets(self.ptr, set1, set2)
    end
end

--Graph Class

do
    local Graph = torch.class('cv.Graph', cv)

    function Graph:__init(t)
        local argRules = {
            {"num_vertices", default = 0}}
        local num_vertices = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.Graph_ctor(num_vertices), C.Graph_dtor)
    end

    function Graph:addEdge(t)
        local argRules = {
            {"from", required = true},
            {"to", required = true},
            {"weight", required = true}}
        local from, to, weight = cv.argcheck(t, argRules)
        C.Graph_addEdge(self.ptr, from, to, weight)
    end

    function Graph:create(t)
        local argRules = {
            {"num_vertices", required = true}}
        local num_vertices = cv.argcheck(t, argRules)
        C.Graph_create(self.ptr, num_vertices)
    end

    function Graph:numVertices(t)
        return C.Graph_numVertices(self.ptr)
    end
end

--GraphEdge

do
    local GraphEdge = torch.class('cv.GraphEdge', cv)

    function GraphEdge:__init(t)
        local argRules = {
            {"from", required = true},
            {"to", required = true},
            {"weight", required = true}}
        local from, to, weight = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.GraphEdge_ctor(from, to, weight), C.GraphEdge_dtor)
    end

    function GraphEdge:__lt(t)
        local argRules = {
            {"other", required = true}}
        local other = cv.argcheck(t, argRules)
        return self.weight < other.weight
    end
end

--Timelapser

do
    local Timelapser = torch.class('cv.Timelapser', cv)

    function Timelapser:__init()
        local argRules = {
            {"type", required = true}}
        local type_arg = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.Timelapser_ctor(type_arg), C.Timelapser_dtor)
    end
			
    function Timelapser:initialize(t)
        local argRules = {
            {"corners", required = true},
            {"sizes", required = true}}
        local corners, sizes = cv.argcheck(t, argRules)
        C.Timelapser_initialize(self.ptr, corners, sizes)
    end

    function Timelapser:process(t)
        local argRules = {
            {"img", required = true},
            {"mask", required = true},
            {"tl", reqquired = true}}
        local img, mask, tl = cv.argcheck(t, argRules)
        C.Timelapser_process(self.ptr, cv.wrap_tensor(img), cv.wrap_tensor(mask), tl)
    end

end

--TimelapserCrop

do
    local TimelapserCrop = torch.class('cv.TimelapserCrop', 'cv.Timelapser', cv)

--TODO need to add constructor & destructor C.Timelapser_dtor 

    function TimelapserCrop:initialize(t)
        local argRules = {
            {"corners", required = true},
            {"sizes", required = true}}
        local corners, sizes = cv.argcheck(t, argRules)
        C.TimelapserCrop_initialize(self.ptr, corners, sizes)
    end
end


--*******************Features Finding and Images Matching*********************


--MatchesInfo

do
    local MatchesInfo = torch.class('cv.MatchesInfo', cv)

    function MatchesInfo:_init(t)
        local argRules = {
            {"other", default = nil} }
        local other = cv.argcheck(t, argRules)
        if other then
            self.ptr = ffi.gc(C.MatchesInfo_ctor2(other.ptr), C.MatchesInfo_dtor);
        else
            self.ptr = ffi.gc(C.MatchesInfo_ctor(), C.MatchesInfo_dtor);
        end
    end

    --TODO need to add operator=
end

--FeaturesFinder

do
    local FeaturesFinder = torch.class('cv.FeaturesFinder', cv)

    function FeaturesFinder:collectGarbage()
        C.FeaturesFinder_collectGarbage(self.ptr)
    end

    function FeaturesFinder:__call()
        local argRules = {
            {"image", required = true},
            {"rois", default = nil}}
        local image, rois = cv.argcheck(t, argRules)

        local imgFeat = cv.ImageFeatures{}

        if rois then
            local retval =  ffi.gc(C.FeaturesFinder_call2(self.ptr, cv.wrap_tensor(image), rois),
                                   C.FeaturesFinder_dtor)
            imgFeat.ptr = retval
            return imgFeat
        else
            local retval = ffi.gc(C.FeaturesFinder_call(self.ptr, cv.wrap_tensor(image)),
                                  C.FeaturesFinder_dtor)
            imgFeat.ptr = retval
            return imgFeat
        end
    end
end

--OrbFeaturesFinder

do
    local OrbFeaturesFinder = torch.class('cv.OrbFeaturesFinder', 'cv.FeaturesFinder', cv)

    function OrbFeaturesFinder:__init(t)
        local argRules = {
            {"_grid_size", default = cv.Size(3,1), operator = cv.Size},
            {"nfeatures", default = 1500},
            {"scaleFactor", default = 1.3},
            {"nlevels", default = 5}}
        local _grid_size, nfeatures, scaleFactor, nlevels = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.OrbFeaturesFinder(_grid_size, nfeatures, scaleFactor, nlevels),
                          C.FeaturesFinder_dtor)
    end


end

--SurfFeaturesFinder

do
    local SurfFeaturesFinder = torch.class('cv.SurfFeaturesFinder', 'cv.FeaturesFinder', cv)

    function SurfFeaturesFinder:__init(t)
        local argRules = {
            {"hess_thresh", default = 300},
            {"num_octaves", default = 3},
            {"num_layers", default = 4},
            {"num_octaves_descr", default = 3},
            {"num_layers_descr", default = 4} }
        local hess_thresh, num_octaves, num_layers, num_octaves_descr, num_layers_descr = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.SurfFeaturesFinder_ctor(hess_thresh, num_octaves, num_layers, num_octaves_descr, num_layers_descr),
                          C.FeaturesFinder_dtor)
    end
end

--ImageFeatures

do
    local ImageFeatures = torch.class('cv.ImageFeatures', cv)

    function ImageFeatures:__init()
        self.ptr = ffi.gc(C.ImageFeatures_ctor(), C.ImageFeatures_dtor)
    end
end

--FeaturesMatcher

do
    local FeaturesMatcher = torch.class('cv.FeaturesMatcher', cv)

    function FeaturesMatcher:collectGarbage()
        C.FeaturesMatcher_collectGarbage(self.ptr);
    end

    function FeaturesMatcher_isThreadSafe()
        return C.FeaturesMatcher_isThreadSafe(self.ptr);
    end

    function FeaturesMatcher:__call(t)
        local argRules = {
            {"features1", required = true},
            {"features2", required = true} }
        local features1, features2 = cv.argcheck(t, argRules)
        local result =  ffi.gc(C.FeaturesMatcher_call(features1.ptr, features2.ptr),
                               C.MatchesInfo_dtor)
        local retval = cv.MatchesInfo{}
        retval.ptr = result
        return retval;
    end
end

--BestOf2NearestMatcher

do
    local BestOf2NearestMatcher = torch.class('cv.BestOf2NearestMatcher', 'cv.FeaturesMatcher', cv)

    function BestOf2NearestMatcher:__init(t)
        local argRules = {
            {"try_use_gpu", default = false},
            {"match_conf", default = 0.3},
            {"num_matches_thresh1", default = 6},
            {"num_matches_thresh2", default = 6} }
        local try_use_gpu, match_conf, num_matches_thresh1, num_matches_thresh2 = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(
                C.BestOf2NearestMatcher_ctor(try_use_gpu, match_conf, num_matches_thresh1, num_matches_thresh2),
                C.FeaturesMatcher_dtor)
    end

    function BestOf2NearestMatcher:collectGarbage()
        C.BestOf2NearestMatcher_collectGarbage(self.ptr)
    end
end

--BestOf2NearestRangeMatcher

do
    local BestOf2NearestRangeMatcher = torch.class('cv.BestOf2NearestRangeMatcher', 'cv.BestOf2NearestMatcher', cv)

    function BestOf2NearestRangeMatcher:__init(t)
        local argRules = {
            {"range_width", default = 5},
            {"try_use_gpu", default = false},
            {"match_conf", default = 0.3},
            {"num_matches_thresh1", default = 6},
            {"num_matches_thresh2", default = 6} }
        local range_width, try_use_gpu, match_conf,
              num_matches_thresh1, num_matches_thresh2 = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.BestOf2NearestRangeMatcher_ctor(
                                        range_width, try_use_gpu, match_conf,
                                        num_matches_thresh1, num_matches_thresh2),
                          C.FeaturesMatcher_dtor)
    end

--TODO need to make operator()
end

return cv
