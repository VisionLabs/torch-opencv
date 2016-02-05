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
	int count, int size, struct IntArray subset);

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
        {"size", required = true},
        {"subset", required = true}}
    local count, size, subset = cv.argcheck(t, argRules)
    C.detail_selectRandomSubset(count, size, subset)
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

struct TimelapserPtr Timelapser_ctor(
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

void FeaturesFinder_collectGarbage(
	struct PtrWrapper ptr);
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
            self.ptr = ffi.gc(С.CameraParams_ctor2(other.ptr), C.CameraParams_dtor)
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


return cv

--FeaturesFinder

do
    local FeaturesFinder = torch.class('cv.FeaturesFinder', cv)

    function FeaturesFinder:collectGarbage()
        C.FeaturesFinder_collectGarbage(self.ptr)
    end




