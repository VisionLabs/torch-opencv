local cv = require 'cv._env'
local ffi = require 'ffi'
require 'cv.Classes'

ffi.cdef[[
struct RectPlusBool detail_overlapRoi(
	struct PointWrapper tl1, struct PointWrapper tl2,
	struct SizeWrapper sz1, struct SizeWrapper sz2);

struct RectWrapper detail_resultRoi(
	struct PointArray corners,
	struct SizeArray sizes);

struct RectWrapper detail_resultRoi2(
		struct PointArray corners,
		struct TensorArray images);

struct RectWrapper detail_resultRoiIntersection(
	struct PointArray corners,
	struct SizeArray sizes);

struct PointWrapper detail_resultTl(
	struct PointArray corners);

void detail_selectRandomSubset(
	int count, int size);

int detail_stitchingLogLevel();

struct GraphPtrPlusIntArray {
	struct IntArray array;
	struct PtrWrapper graph;
};

struct GraphPtrPlusIntArray detail_findMaxSpanningTree(
        int num_images, struct ClassArray pairwise_matches);

struct IntArray detail_leaveBiggestComponent(
        struct ClassArray features, struct ClassArray pairwise_matches, float conf_threshold);

struct StringWrapper detail_matchesGraphAsString(
        struct StringArray pathes, struct ClassArray pairwise_matches, float conf_threshold);

void detail_waveCorrect(
        struct TensorArray rmats, int kind);

struct TensorPlusBool detail_calibrateRotatingCamera(
        struct TensorArray Hs);

struct DoubleArray detail_estimateFocal(struct ClassArray features, struct ClassArray pairwise_matches);

struct focalsFromHomographyRetval {
	double f0, f1;
	bool f0_ok, f1_ok;
};

struct focalsFromHomographyRetval detail_focalsFromHomography(
        struct TensorWrapper H);

struct TensorArray detail_createLaplacePyr(
		struct TensorWrapper img, int num_levels);

struct TensorArray detail_createLaplacePyrGpu(
		struct TensorWrapper img, int num_levels);

void detail_createWeightMap(
		struct TensorWrapper mask, float sharpness,
		struct TensorWrapper weight);

void detail_normalizeUsingWeightMap(
		struct TensorWrapper weight, struct TensorWrapper src);

void detail_restoreImageFromLaplacePyr(
		struct TensorArray pyr);

void detail_restoreImageFromLaplacePyrGpu(
		struct TensorArray pyr);
]]

local C = ffi.load(cv.libPath('stitching'))

cv.detail = {};

function cv.detail.overlapRoi(t)
    local argRules = {
        {"tl1", required = true, operator = cv.Point},
        {"tl2", required = true, operator = cv.Point},
        {"sz1", required = true, operator = cv.Size},
        {"sz2", required = true, operator = cv.Size}}
    local tl1, tl2, sz1, sz2 = cv.argcheck(t, argRules)
    local result = C.detail_overlapRoi(tl1, tl2, sz1, sz2)
    return result.val, result.rect
end

function cv.detail.resultRoi(t)
    local argRules = {
        {"corners", required = true},
        {"sizes", required = true}}
    local corners, sizes = cv.argcheck(t, argRules)
    return C.detail_resultRoi(corners, sizes)
end

function cv.detail.resultRoi2(t)
    local argRules = {
        {"corners", required = true},
        {"images", required = true}}
    local corners, images = cv.argcheck(t, argRules)
    return C.detail_resultRoi2(corners, cv.wrap_tensors(images))
end

function cv.detail.resultRoiIntersection(t)
    local argRules = {
        {"corners", required = true},
        {"sizes", required = true}}
    local corners, sizes = cv.argcheck(t, argRules)
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

--**********************Rotation Estimation********************************

cv.detail.WAVE_CORRECT_HORIZ = 0
cv.detail.WAVE_CORRECT_VERT = 1

function cv.detail.findMaxSpanningTree(t)
    local argRules = {
        {"num_images", required = true},
        {"pairwise_matches", required = true}}
    local num_images, pairwise_matches = cv.argcheck(t, argRules)
    local array = cv.newArray("Class", pairwise_matches)
    local result = C.detail_findMaxSpanningTree(num_images, array)
    local retval = cv.MatchesInfo
    retval.ptr = result.graph
    return retval, cv.gcarray(result.array)
end

function cv.detail.leaveBiggestComponent(t)
    local argRules = {
        {"features", required = true},
        {"pairwise_matches", required  = true},
        {"conf_threshold", required = true}}
    local features, pairwise_matches, conf_threshold = cv.argcheck(t, argRules)
    return cv.gcarray(
                C.detail_leaveBiggestComponent(
                    cv.newArray("Class", features), cv.newArray("Class", pairwise_matches), conf_threshold))
end

function cv.detail.matchesGraphAsString(t)
    local argRules = {
        {"pathes", required = true},
        {"pairwise_matches", required = true},
        {"conf_threshold", required = true}}
    local pathes, pairwise_matches, conf_threshold = cv.argcheck(t, argRules)
    return cv.unwrap_string(
                C.detail_matchesGraphAsString(
                    cv.newArray("cv.String", pathes), cv.newArray("Class", pairwise_matches), conf_threshold))
end

function cv.detail.waveCorrect(t)
    local argRules = {
        {"rmats", required = true},
        {"kind", required = true} }
    local rmats, kind = cv.argcheck(t, argRules)
    C.detail_waveCorrect(cv.wrap_tensors(rmats), kind)
end

--***********************Autocalibration************************

function cv.detail.calibrateRotatingCamera(t)
    local argRules = {
        {"Hs", required = true}}
    local Hs = cv.argcheck(t, argRules)
    local result = C.detail_calibrateRotatingCamera(cv.wrap_tensors(Hs))
    return result.val, cv.unwrap_tensors(result.tensor)
end

function cv.detail.estimateFocal(t)
    local argRules = {
        {"features", required = true},
        {"pairwise_matches", required = true} }
    local features, pairwise_matches = cv.argcheck(t, argRules)
    return cv.gcarray(
                C.detail_estimateFocal(
                    cv.newArray("Class", features), cv.newArray("Class", pairwise_matches)))
end

function cv.detail.focalsFromHomography(t)
    local argRules = {
        {"H", required = true}}
    local H = cv.argcheck(t, argRules)
    local result = C.detail_focalsFromHomography(cv.wrap_tensor(H))
    return result.f0, result.f1, result.f0_ok, result.f1_ok
end


--*******************************Image Blenders**********************

cv.detail_WAVE_CORRECT_HORIZ = 0
cv.detail_WAVE_CORRECT_VERT = 1

function cv.detail.createLaplacePyr(t)
    local argRules = {
        {"img", required = true},
        {"num_levels", required = true} }
    local img, num_levels = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
                C.detail_createLaplacePyr(cv.wrap_tensor(img), num_levels))
end

function cv.detail.createLaplacePyrGpu(t)
    local argRules = {
        {"img", required = true},
        {"num_levels", required = true} }
    local img, num_levels = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
        C.detail_createLaplacePyrGpu(cv.wrap_tensor(img), num_levels))
end

function cv.detail.createWeightMap(t)
    local argRules = {
        {"mask", required = true},
        {"sharpness", required = true},
        {"weight", required = true} }
    local mask, sharpness, weight = cv.argcheck(t, argRules)
    C.detail_createWeightMap(cv.wrap_tensor(mask), sharpness, cv.wrap_tensor(weight))
end

function cv.detail.normalizeUsingWeightMap(t)
    local argRules = {
        {"weight", required = true},
        {"src", required = true} }
    local weight, src = cv.argcheck(t, argRules)
    C.detail_normalizeUsingWeightMap(cv.wrap_tensor(weight), cv.wrap_tensor(src))
end

function cv.detail.restoreImageFromLaplacePyr(t)
    local argRules = {
        {"pyr", required = true}}
    local pyr = cv.argcheck(t, argRules)
    C.detail_restoreImageFromLaplacePyr(cv.wrap_tensors(pyr))
end

function cv.detail.restoreImageFromLaplacePyrGpu(t)
    local argRules = {
        {"pyr", required = true}}
    local pyr = cv.argcheck(t, argRules)
    C.detail_restoreImageFromLaplacePyrGpu(cv.wrap_tensors(pyr))
end

--- ***************** Classes *****************

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
	struct PtrWrapper ptr);

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
	int type);

void Timelapser_dtor(
	struct PtrWrapper ptr);

struct TensorWrapper Timelapser_getDst(
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
	struct PtrWrapper ptr);

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
    struct PtrWrapper ptr);

struct PtrWrapper FeaturesMatcher_call(
        struct PtrWrapper ptr, struct PtrWrapper features1,
        struct PtrWrapper features2);

bool FeaturesMatcher_isThreadSafe(
        struct PtrWrapper ptr);

struct PtrWrapper MatchesInfo_ctor();

struct PtrWrapper MatchesInfo_ctor2(
        struct PtrWrapper other);

void MatchesInfo_dtor(
        struct PtrWrapper other);

struct PtrWrapper BestOf2NearestRangeMatcher_ctor(
		int range_width, bool try_use_gpu, float match_conf,
		int num_matches_thresh1, int num_matches_thresh2);

void BestOf2NearestRangeMatcher_call(
        struct PtrWrapper ptr, struct ClassArray features,
        struct ClassArray pairwise_matches, struct TensorWrapper mask);

struct PtrWrapper OrbFeaturesFinder_ctor(
        struct SizeWrapper _grid_size, int nfeatures, float scaleFactor, int nlevels);

struct PtrWrapper SurfFeaturesFinder_ctor(
        double hess_thresh, int num_octaves, int num_layers, int num_octaves_descr, int num_layers_descr);

void Estimator_dtor(
        struct EstimatorPtr ptr);

struct BoolPlusClassArray {
	bool val;
	struct ClassArray array;
};

struct BoolPlusClassArray Estimator_call(
        struct PtrWrapper ptr, struct ClassArray features, struct ClassArray pairwise_matches);

struct PtrWrapper HomographyBasedEstimator_ctor(
        bool is_focals_estimated);

double BundleAdjusterBase_confThresh(
        struct PtrWrapper ptr);

struct TensorWrapper BundleAdjusterBase_refinementMask(
        struct PtrWrapper ptr);

void BundleAdjusterBase_setConfThresh(
       struct PtrWrapper ptr, double conf_thresh);

void BundleAdjusterBase_setRefinementMask(
        struct PtrWrapper ptr, struct TensorWrapper mask);

void BundleAdjusterBase_setTermCriteria(
       struct PtrWrapper ptr, struct TermCriteriaWrapper term_criteria);

struct TermCriteriaWrapper BundleAdjusterBase_termCriteria(
        struct PtrWrapper ptr);

struct PtrWrapper BundleAdjusterRay_ctor();

struct PtrWrapper BundleAdjusterReproj_ctor();

struct PtrWrapper ProjectorBase_ctor();

void ProjectorBase_dtor(
		struct PtrWrapper ptr);

void ProjectorBase_setCameraParams(
        struct PtrWrapper ptr, struct TensorWrapper K,
        struct TensorWrapper R, struct TensorWrapper T);

struct FloatArray CompressedRectilinearPortraitProjector_mapBackward(
        struct PtrWrapper ptr, float u, float v);

struct FloatArray CompressedRectilinearPortraitProjector_mapForward(
        struct PtrWrapper ptr, float x, float y);

struct PtrWrapper CompressedRectilinearPortraitProjector_ctor();

void CompressedRectilinearPortraitProjector_dtor(
        struct PtrWrapper ptr);

struct PtrWrapper CompressedRectilinearProjector_ctor();

void CompressedRectilinearProjector_dtor(
		struct PtrWrapper ptr);

struct FloatArray CompressedRectilinearProjector_mapBackward(
		struct PtrWrapper ptr, float u, float v);

struct FloatArray CompressedRectilinearProjector_mapForward(
		struct PtrWrapper ptr, float x, float y);

struct PtrWrapper CylindricalPortraitProjector_ctor();

void CylindricalPortraitProjector_dtor(
		struct PtrWrapper ptr);

struct FloatArray CylindricalPortraitProjector_mapBackward(
		struct PtrWrapper ptr, float u, float v);

struct FloatArray CylindricalPortraitProjector_mapForward(
		struct PtrWrapper ptr, float x, float y);

struct PtrWrapper CylindricalProjector_ctor();

void CylindricalProjector_dtor(
		struct PtrWrapper ptr);

struct FloatArray CylindricalProjector_mapBackward(
		struct PtrWrapper ptr, float u, float v);

struct FloatArray CylindricalProjector_mapForward(
		struct PtrWrapper ptr, float x, float y);

struct PtrWrapper FisheyeProjector_ctor();

void FisheyeProjector_dtor(
		struct PtrWrapper ptr);

struct FloatArray FisheyeProjector_mapBackward(
		struct PtrWrapper ptr, float u, float v);

struct FloatArray FisheyeProjector_mapForward(
		struct PtrWrapper ptr, float x, float y);

struct PtrWrapper MercatorProjector_ctor();

void MercatorProjector_dtor(
		struct PtrWrapper ptr);

struct FloatArray MercatorProjector_mapBackward(
		struct PtrWrapper ptr, float u, float v);

struct FloatArray MercatorProjector_mapForward(
		struct PtrWrapper ptr, float x, float y);

struct PtrWrapper PaniniPortraitProjector_ctor();

void PaniniPortraitProjector_dtor(
		struct PtrWrapper ptr);

struct FloatArray PaniniPortraitProjector_mapBackward(
		struct PtrWrapper ptr, float u, float v);

struct FloatArray PaniniPortraitProjector_mapForward(
		struct PtrWrapper ptr, float x, float y);

struct PtrWrapper PaniniProjector_ctor();

void PaniniProjector_dtor(
		struct PtrWrapper ptr);

struct FloatArray PaniniProjector_mapBackward(
		struct PtrWrapper ptr, float u, float v);

struct FloatArray PaniniProjector_mapForward(
		struct PtrWrapper ptr, float x, float y);

struct PtrWrapper PlanePortraitProjector_ctor();

void PlanePortraitProjector_dtor(
		struct PtrWrapper ptr);

struct FloatArray PlanePortraitProjector_mapBackward(
		struct PtrWrapper ptr, float u, float v);

struct FloatArray PlanePortraitProjector_mapForward(
		struct PtrWrapper ptr, float x, float y);

struct PtrWrapper PlaneProjector_ctor();

void PlaneProjector_dtor(
		struct PtrWrapper ptr);

struct FloatArray PlaneProjector_mapBackward(
		struct PtrWrapper ptr, float u, float v);

struct FloatArray PlaneProjector_mapForward(
		struct PtrWrapper ptr, float x, float y);

struct PtrWrapper SphericalPortraitProjector_ctor();

void SphericalPortraitProjector_dtor(
		struct PtrWrapper ptr);

struct FloatArray SphericalPortraitProjector_mapBackward(
		struct PtrWrapper ptr, float u, float v);

struct FloatArray SphericalPortraitProjector_mapForward(
		struct PtrWrapper ptr, float x, float y);

struct PtrWrapper SphericalProjector_ctor();

void SphericalProjector_dtor(
		struct PtrWrapper ptr);

struct FloatArray SphericalProjector_mapBackward(
		struct PtrWrapper ptr, float u, float v);

struct FloatArray SphericalProjector_mapForward(
		struct PtrWrapper ptr, float x, float y);

struct PtrWrapper StereographicProjector_ctor();

void StereographicProjector_dtor(
		struct PtrWrapper ptr);

struct FloatArray StereographicProjector_mapBackward(
		struct PtrWrapper ptr, float u, float v);

struct FloatArray StereographicProjector_mapForward(
		struct PtrWrapper ptr, float x, float y);

struct PtrWrapper TransverseMercatorProjector_ctor();

void TransverseMercatorProjector_dtor(
		struct PtrWrapper ptr);

struct FloatArray TransverseMercatorProjector_mapBackward(
		struct PtrWrapper ptr, float u, float v);

struct FloatArray TransverseMercatorProjector_mapForward(
		struct PtrWrapper ptr, float x, float y);

void RotationWarper_dtor(
		struct PtrWrapper ptr);

struct TensorArrayPlusRect RotationWarper_buildMaps(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

float RotationWarper_getScale(
		struct PtrWrapper ptr);

void RotationWarper_setScale(
		struct PtrWrapper ptr, float val);

struct TensorPlusPoint RotationWarper_warp(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

struct TensorWrapper RotationWarper_warpBackward(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

struct Point2fWrapper RotationWarper_warpPoint(
		struct PtrWrapper ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

struct RectWrapper RotationWarper_warpRoi(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

struct PtrWrapper RotationWarperBase_CompressedRectilinearPortraitProjector_ctor();

struct TensorArrayPlusRect RotationWarperBase_CompressedRectilinearPortraitProjector_buildMaps(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

float RotationWarperBase_CompressedRectilinearPortraitProjector_getScale(
		struct PtrWrapper ptr);

void RotationWarperBase_CompressedRectilinearPortraitProjector_setScale(
		struct PtrWrapper ptr, float val);

struct TensorPlusPoint RotationWarperBase_CompressedRectilinearPortraitProjector_warp(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

struct TensorWrapper RotationWarperBase_CompressedRectilinearPortraitProjector_warpBackward(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

struct Point2fWrapper RotationWarperBase_CompressedRectilinearPortraitProjector_warpPoint(
		struct PtrWrapper ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

struct RectWrapper RotationWarperBase_CompressedRectilinearPortraitProjector_warpRoi(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

struct PtrWrapper RotationWarperBase_CompressedRectilinearProjector_ctor();

struct TensorArrayPlusRect RotationWarperBase_CompressedRectilinearProjector_buildMaps(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

float RotationWarperBase_CompressedRectilinearProjector_getScale(
		struct PtrWrapper ptr);

void RotationWarperBase_CompressedRectilinearProjector_setScale(
		struct PtrWrapper ptr, float val);

struct TensorPlusPoint RotationWarperBase_CompressedRectilinearProjector_warp(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

struct TensorWrapper RotationWarperBase_CompressedRectilinearProjector_warpBackward(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

struct Point2fWrapper RotationWarperBase_CompressedRectilinearProjector_warpPoint(
		struct PtrWrapper ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

struct RectWrapper RotationWarperBase_CompressedRectilinearProjector_warpRoi(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

struct PtrWrapper RotationWarperBase_CylindricalPortraitProjector_ctor();

struct TensorArrayPlusRect RotationWarperBase_CylindricalPortraitProjector_buildMaps(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

float RotationWarperBase_CylindricalPortraitProjector_getScale(
		struct PtrWrapper ptr);

void RotationWarperBase_CylindricalPortraitProjector_setScale(
		struct PtrWrapper ptr, float val);

struct TensorPlusPoint RotationWarperBase_CylindricalPortraitProjector_warp(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

struct TensorWrapper RotationWarperBase_CylindricalPortraitProjector_warpBackward(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

struct Point2fWrapper RotationWarperBase_CylindricalPortraitProjector_warpPoint(
		struct PtrWrapper ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

struct RectWrapper RotationWarperBase_CylindricalPortraitProjector_warpRoi(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

struct PtrWrapper RotationWarperBase_CylindricalProjector_ctor();

struct PtrWrapper RotationWarperBase_CylindricalProjector_buildMaps(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

float RotationWarperBase_CylindricalProjector_getScale(
		struct PtrWrapper ptr);

void RotationWarperBase_CylindricalProjector_setScale(
		struct PtrWrapper ptr, float val);

struct TensorPlusPoint RotationWarperBase_CylindricalProjector_warp(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

struct TensorWrapper RotationWarperBase_CylindricalProjector_warpBackward(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

struct Point2fWrapper RotationWarperBase_CylindricalProjector_warpPoint(
		struct PtrWrapper ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

struct RectWrapper RotationWarperBase_CylindricalProjector_warpRoi(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

struct PtrWrapper RotationWarperBase_FisheyeProjector_ctor();

struct TensorArrayPlusRect RotationWarperBase_FisheyeProjector_buildMaps(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

float RotationWarperBase_FisheyeProjector_getScale(
		struct PtrWrapper ptr);

void RotationWarperBase_FisheyeProjector_setScale(
		struct PtrWrapper ptr, float val);

struct TensorPlusPoint RotationWarperBase_FisheyeProjector_warp(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

struct TensorWrapper RotationWarperBase_FisheyeProjector_warpBackward(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

struct Point2fWrapper RotationWarperBase_FisheyeProjector_warpPoint(
		struct PtrWrapper ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

struct RectWrapper RotationWarperBase_FisheyeProjector_warpRoi(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);


struct PtrWrapper RotationWarperBase_MercatorProjector_ctor();

struct TensorArrayPlusRect RotationWarperBase_MercatorProjector_buildMaps(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

float RotationWarperBase_MercatorProjector_getScale(
		struct PtrWrapper ptr);

void RotationWarperBase_MercatorProjector_setScale(
		struct PtrWrapper ptr, float val);

struct TensorPlusPoint RotationWarperBase_MercatorProjector_warp(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

struct TensorWrapper RotationWarperBase_MercatorProjector_warpBackward(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

struct Point2fWrapper RotationWarperBase_MercatorProjector_warpPoint(
		struct PtrWrapper ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

struct RectWrapper RotationWarperBase_MercatorProjector_warpRoi(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

struct PtrWrapper RotationWarperBase_PaniniPortraitProjector_ctor();

struct TensorArrayPlusRect RotationWarperBase_PaniniPortraitProjector_buildMaps(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

float RotationWarperBase_PaniniPortraitProjector_getScale(
		struct PtrWrapper ptr);

void RotationWarperBase_PaniniPortraitProjector_setScale(
		struct PtrWrapper ptr, float val);

struct TensorPlusPoint RotationWarperBase_PaniniPortraitProjector_warp(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

struct TensorWrapper RotationWarperBase_PaniniPortraitProjector_warpBackward(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

struct Point2fWrapper RotationWarperBase_PaniniPortraitProjector_warpPoint(
		struct PtrWrapper ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

struct RectWrapper RotationWarperBase_PaniniPortraitProjector_warpRoi(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

struct PtrWrapper RotationWarperBase_PaniniProjector_ctor();

struct TensorArrayPlusRect RotationWarperBase_PaniniProjector_buildMaps(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

float RotationWarperBase_PaniniProjector_getScale(
		struct PtrWrapper ptr);

void RotationWarperBase_PaniniProjector_setScale(
		struct PtrWrapper ptr, float val);

struct TensorPlusPoint RotationWarperBase_PaniniProjector_warp(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

struct TensorWrapper RotationWarperBase_PaniniProjector_warpBackward(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

struct Point2fWrapper RotationWarperBase_PaniniProjector_warpPoint(
		struct PtrWrapper ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

struct RectWrapper RotationWarperBase_PaniniProjector_warpRoi(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

struct PtrWrapper RotationWarperBase_PlanePortraitProjector_ctor();

struct TensorArrayPlusRect RotationWarperBase_PlanePortraitProjector_buildMaps(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

float RotationWarperBase_PlanePortraitProjector_getScale(
		struct PtrWrapper ptr);

void RotationWarperBase_PlanePortraitProjector_setScale(
		struct PtrWrapper ptr, float val);

struct TensorPlusPoint RotationWarperBase_PlanePortraitProjector_warp(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

struct TensorWrapper RotationWarperBase_PlanePortraitProjector_warpBackward(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

struct Point2fWrapper RotationWarperBase_PlanePortraitProjector_warpPoint(
		struct PtrWrapper ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

struct RectWrapper RotationWarperBase_PlanePortraitProjector_warpRoi(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

struct PtrWrapper RotationWarperBase_PlaneProjector_ctor();

struct TensorArrayPlusRect RotationWarperBase_PlaneProjector_buildMaps(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

float RotationWarperBase_PlaneProjector_getScale(
		struct PtrWrapper ptr);

void RotationWarperBase_PlaneProjector_setScale(
		struct PtrWrapper ptr, float val);

struct TensorPlusPoint RotationWarperBase_PlaneProjector_warp(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

struct TensorWrapper RotationWarperBase_PlaneProjector_warpBackward(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

struct Point2fWrapper RotationWarperBase_PlaneProjector_warpPoint(
		struct PtrWrapper ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

struct RectWrapper RotationWarperBase_PlaneProjector_warpRoi(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

struct PtrWrapper RotationWarperBase_SphericalPortraitProjector_ctor();

struct TensorArrayPlusRect RotationWarperBase_SphericalPortraitProjector_buildMaps(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

float RotationWarperBase_SphericalPortraitProjector_getScale(
		struct PtrWrapper ptr);

void RotationWarperBase_SphericalPortraitProjector_setScale(
		struct PtrWrapper ptr, float val);

struct TensorPlusPoint RotationWarperBase_SphericalPortraitProjector_warp(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

struct TensorWrapper RotationWarperBase_SphericalPortraitProjector_warpBackward(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

struct Point2fWrapper RotationWarperBase_SphericalPortraitProjector_warpPoint(
		struct PtrWrapper ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

struct RectWrapper RotationWarperBase_SphericalPortraitProjector_warpRoi(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

struct PtrWrapper RotationWarperBase_SphericalProjector_ctor();

struct TensorArrayPlusRect RotationWarperBase_SphericalProjector_buildMaps(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

float RotationWarperBase_SphericalProjector_getScale(
		struct PtrWrapper ptr);

void RotationWarperBase_SphericalProjector_setScale(
		struct PtrWrapper ptr, float val);

struct TensorPlusPoint RotationWarperBase_SphericalProjector_warp(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

struct TensorWrapper RotationWarperBase_SphericalProjector_warpBackward(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

struct Point2fWrapper RotationWarperBase_SphericalProjector_warpPoint(
		struct PtrWrapper ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

struct RectWrapper RotationWarperBase_SphericalProjector_warpRoi(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

struct PtrWrapper RotationWarperBase_StereographicProjector_ctor();

struct TensorArrayPlusRect RotationWarperBase_StereographicProjector_buildMaps(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

float RotationWarperBase_StereographicProjector_getScale(
		struct PtrWrapper ptr);

void RotationWarperBase_StereographicProjector_setScale(
		struct PtrWrapper ptr, float val);

struct TensorPlusPoint RotationWarperBase_StereographicProjector_warp(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

struct TensorWrapper RotationWarperBase_StereographicProjector_warpBackward(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

struct Point2fWrapper RotationWarperBase_StereographicProjector_warpPoint(
		struct PtrWrapper ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

struct RectWrapper RotationWarperBase_StereographicProjector_warpRoi(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

struct PtrWrapper RotationWarperBase_TransverseMercatorProjector_ctor();

struct TensorArrayPlusRect RotationWarperBase_TransverseMercatorProjector_buildMaps(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

float RotationWarperBase_TransverseMercatorProjector_getScale(
		struct PtrWrapper ptr);

void RotationWarperBase_TransverseMercatorProjector_setScale(
		struct PtrWrapper ptr, float val);

struct TensorPlusPoint RotationWarperBase_TransverseMercatorProjector_warp(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

struct TensorWrapper RotationWarperBase_TransverseMercatorProjector_warpBackward(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

struct Point2fWrapper RotationWarperBase_TransverseMercatorProjector_warpPoint(
		struct PtrWrapper ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

struct RectWrapper RotationWarperBase_TransverseMercatorProjector_warpRoi(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

void WarperCreator_dtor(
		struct PtrWrapper ptr);

struct RotationWarperPtr WarperCreator_create(
		struct PtrWrapper ptr, float scale);

struct PtrWrapper CompressedRectilinearPortraitWarper_ctor(
        float A, float B);

struct PtrWrapper CompressedRectilinearPortraitWarper_create(
		struct PtrWrapper ptr, float scale);

struct PtrWrapper CompressedRectilinearWarper_ctor(
		float A, float B);

struct PtrWrapper CompressedRectilinearWarper_create(
		struct PtrWrapper ptr, float scale);

struct PtrWrapper CylindricalWarper_ctor();

struct PtrWrapper CylindricalWarper_create(
		struct PtrWrapper ptr, float scale);

struct PtrWrapper FisheyeWarper_ctor();

struct PtrWrapper FisheyeWarper_create(
		struct PtrWrapper ptr, float scale);

struct PtrWrapper MercatorWarper_ctor();

struct PtrWrapper MercatorWarper_create(
		struct PtrWrapper ptr, float scale);

struct PtrWrapper PaniniPortraitWarper_ctor(
		float A, float B);

struct PtrWrapper PaniniPortraitWarper_create(
		struct PtrWrapper ptr, float scale);

struct PtrWrapper PaniniWarper_ctor(
		float A, float B);

struct PtrWrapper PaniniWarper_create(
		struct PtrWrapper ptr, float scale);

struct PtrWrapper PlaneWarper_ctor();

struct PtrWrapper PlaneWarper_create(
		struct PtrWrapper ptr, float scale);

struct PtrWrapper SphericalWarper_ctor();

struct PtrWrapper SphericalWarper_create(
		struct PtrWrapper ptr, float scale);

struct PtrWrapper StereographicWarper_ctor();

struct PtrWrapper StereographicWarper_create(
		struct PtrWrapper ptr, float scale);

struct PtrWrapper TransverseMercatorWarper_ctor();

struct PtrWrapper TransverseMercatorWarper_create(
		struct PtrWrapper ptr, float scale);

struct PtrWrapper detail_CompressedRectilinearPortraitWarper_ctor(
		float scale, float A, float B);

struct PtrWrapper detail_CompressedRectilinearWarper_ctor(
		float scale, float A, float B);

struct PtrWrapper detail_CylindricalPortraitWarper_ctor(
		float scale);

struct PtrWrapper detail_CylindricalWarper_ctor(
		float scale);

struct TensorArrayPlusRect detail_CylindricalWarper_buildMaps(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

struct TensorPlusPoint detail_CylindricalWarper_warp(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct TensorWrapper dst);

struct PtrWrapper detail_CylindricalWarperGpu_ctor(
		float scale);

struct TensorArrayPlusRect detail_CylindricalWarperGpu_buildMaps(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

struct TensorPlusPoint detail_CylindricalWarperGpu_warp(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct TensorWrapper dst);

struct PtrWrapper detail_FisheyeWarper_ctor(
		float scale);

struct PtrWrapper detail_MercatorWarper_ctor(
		float scale);

struct PtrWrapper detail_PaniniPortraitWarper_ctor(
        float scale, float A, float B);

struct PtrWrapper detail_PaniniWarper_ctor(
        float scale, float A, float B);

struct PtrWrapper detail_PlanePortraitWarper_ctor(
		float scale);

struct PtrWrapper detail_PlaneWarper_ctor(
        float scale);

struct TensorArrayPlusRect detail_PlaneWarper_buildMaps(
        struct PtrWrapper ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper T, struct TensorWrapper xmap,
        struct TensorWrapper ymap);

struct TensorArrayPlusRect detail_PlaneWarper_buildMaps2(
        struct PtrWrapper ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap);

struct TensorPlusPoint detail_PlaneWarper_warp(
        struct PtrWrapper ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper T, int interp_mode,
        int border_mode, struct TensorWrapper dst);

struct TensorPlusPoint detail_PlaneWarper_warp2(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct TensorWrapper dst);

struct Point2fWrapper detail_PlaneWarper_warpPoint(
        struct PtrWrapper ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R, struct TensorWrapper T);

struct Point2fWrapper detail_PlaneWarper_warpPoint2(
        struct PtrWrapper ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R);

struct RectWrapper detail_PlaneWarper_warpRoi2(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

struct RectWrapper detail_PlaneWarper_warpRoi(
		struct PtrWrapper ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper T);

struct PtrWrapper detail_SphericalPortraitWarper_ctor(
		float scale);

struct PtrWrapper detail_SphericalWarper_ctor(
		float scale);

struct TensorArrayPlusRect detail_SphericalWarper_buildMaps(
        struct PtrWrapper ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap);

struct TensorPlusPoint detail_SphericalWarper_warp(
        struct PtrWrapper ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
        int border_mode, struct TensorWrapper dst);

struct PtrWrapper detail_SphericalWarperGpu_ctor(
        float scale);

struct TensorArrayPlusRect detail_SphericalWarperGpu_buildMaps(
		struct PtrWrapper, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

struct TensorPlusPoint detail_SphericalWarperGpu_warp(
		struct PtrWrapper ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

struct PtrWrapper detail_StereographicWarper_ctor(
		float scale);

struct PtrWrapper detail_TransverseMercatorWarper_ctor(
        float scale);

void SeamFinder_dtor(
		struct PtrWrapper ptr);

void SeamFinder_find(
         struct TensorArray src);

struct PtrWrapper DpSeamFinder_ctor(int costFunc);

int DpSeamFinder_costFunction(
        struct PtrWrapper ptr);

void DpSeamFinder_find(
        struct PtrWrapper ptr, struct TensorArray src,
        struct PointArray corners, struct TensorArray masks);

void DpSeamFinder_setCostFunction(
        struct PtrWrapper ptr, int val);

struct PtrWrapper GraphCutSeamFinder_ctor(
		int cost_type, float terminal_cost, float bad_region_penalty);

void GraphCutSeamFinder_dtor(
        struct PtrWrapper ptr);

void GraphCutSeamFinder_find(
		struct PtrWrapper ptr, struct TensorArray src,
		struct PointArray corners, struct TensorArray masks);

struct PtrWrapper NoSeamFinder_ctor();

void NoSeamFinder_find(
        struct PtrWrapper ptr, struct TensorArray src,
        struct PointArray corners, struct TensorArray masks);

void PairwiseSeamFinder_find(
		struct PtrWrapper ptr, struct TensorArray src,
		struct PointArray corners, struct TensorArray masks);

struct PtrWrapper VoronoiSeamFinder_ctor();

void VoronoiSeamFinder_find(
		struct PtrWrapper ptr, struct TensorArray src,
		struct PointArray corners, struct TensorArray masks);

void VoronoiSeamFinder_find2(
        struct PtrWrapper ptr, struct SizeArray size,
        struct PointArray corners, struct TensorArray masks);

void ExposureCompensator_dtor(
		struct PtrWrapper ptr);

void  ExposureCompensator_apply(
		struct PtrWrapper ptr, int index, struct PointWrapper corner,
		struct TensorWrapper image, struct TensorWrapper mask);

struct PtrWrapper ExposureCompensator_ctor(
        int type);

void ExposureCompensator_feed(
		struct PtrWrapper ptr, struct PointArray corners,
		struct TensorArray images, struct TensorArray masks);

struct PtrWrapper BlocksGainCompensator_ctor(
		int bl_width, int bl_height);

void  BlocksGainCompensator_apply(
        struct PtrWrapper ptr, int index, struct PointWrapper corner,
        struct TensorWrapper image, struct TensorWrapper mask);

void BlocksGainCompensator_feed(
		struct PtrWrapper ptr, struct PointArray corners,
		struct TensorArray images, struct TensorArray mat, struct UCharArray chr);

struct PtrWrapper GainCompensator_ctor();

void  GainCompensator_apply(
		struct PtrWrapper ptr, int index, struct PointWrapper corner,
		struct TensorWrapper image, struct TensorWrapper mask);

void GainCompensator_feed(
        struct PtrWrapper ptr, struct PointArray corners,
        struct TensorArray images, struct TensorArray mat, struct UCharArray chr);

struct DoubleArray GainCompensator_gains(
		struct PtrWrapper ptr);

struct PtrWrapper NoExposureCompensator_ctor();

void NoExposureCompensator_apply(
		struct PtrWrapper ptr, int index, struct PointWrapper corner,
		struct TensorWrapper image, struct TensorWrapper mask);

void NoExposureCompensator_feed(
		struct PtrWrapper ptr, struct PointArray corners,
		struct TensorArray images, struct TensorArray mat, struct UCharArray chr);

struct PtrWrapper Blender_ctor(
		int type, bool try_gpu);

void Blender_dtor(
		struct PtrWrapper ptr);

void Blender_blend(
		struct PtrWrapper ptr, struct TensorWrapper dst,
		struct TensorWrapper dst_mask);

void Blender_feed(
		struct PtrWrapper ptr, struct TensorWrapper img,
		struct TensorWrapper mask, struct PointWrapper tl);

void Blender_prepare2(
        struct PtrWrapper ptr, struct PointArray corners,
        struct SizeArray sizes);

void Blender_prepare(
		struct PtrWrapper ptr, struct RectWrapper dst_roi);

struct PtrWrapper FeatherBlender_ctor(
		float sharpness);

void FeatherBlender_blend(
		struct PtrWrapper ptr, struct TensorWrapper dst,
		struct TensorWrapper dst_mask);

struct RectWrapper FeatherBlender_createWeightMaps(
		struct PtrWrapper ptr, struct TensorArray masks,
		struct PointArray corners, struct TensorArray weight_maps);

void FeatherBlender_feed(
		struct PtrWrapper ptr, struct TensorWrapper img,
		struct TensorWrapper mask, struct PointWrapper tl);

void FeatherBlender_prepare(
		struct PtrWrapper ptr, struct RectWrapper dst_roi);

void FeatherBlender_setSharpness(
		struct PtrWrapper ptr, float val);

float FeatherBlender_sharpness(
		struct PtrWrapper ptr);

struct PtrWrapper MultiBandBlender_ctor(
		int try_gpu, int num_bands, int weight_type);

void MultiBandBlender_blend(
		struct PtrWrapper ptr, struct TensorWrapper dst,
		struct TensorWrapper dst_mask);

void MultiBandBlender_feed(
		struct PtrWrapper ptr, struct TensorWrapper img,
		struct TensorWrapper mask, struct PointWrapper tl);

int MultiBandBlender_numBands(
		struct PtrWrapper ptr);

void MultiBandBlender_prepare(
		struct PtrWrapper ptr, struct RectWrapper dst_roi);

void MultiBandBlender_setNumBands(
		struct PtrWrapper ptr, int val);

struct PtrWrapper Stitcher_ctor(
    bool try_use_gpu);

void Stitcher_dtor(
		struct PtrWrapper ptr);

struct PtrWrapper Stitcher_blender(
		struct PtrWrapper ptr);

struct PtrWrapper Stitcher_bundleAdjuster(
		struct PtrWrapper ptr);

struct ClassArray Stitcher_cameras(
		struct PtrWrapper ptr);

struct IntArray Stitcher_component(
		struct PtrWrapper ptr);

struct TensorPlusInt Stitcher_composePanorama(
		struct PtrWrapper ptr);

struct TensorPlusInt Stitcher_composePanorama2(
		struct PtrWrapper ptr, struct TensorArray images);

double Stitcher_compositingResol(
		struct PtrWrapper ptr);

struct StitcherPtr Stitcher_createDefault(
		struct PtrWrapper ptr, bool try_use_gpu);

int Stitcher_estimateTransform(
		struct PtrWrapper ptr, struct TensorArray images);

struct PtrWrapper Stitcher_exposureCompensator(
		struct PtrWrapper ptr);

struct PtrWrapper Stitcher_featuresFinder(
		struct PtrWrapper ptr);

struct PtrWrapper Stitcher_featuresMatcher(
		struct PtrWrapper ptr);

struct TensorWrapper Stitcher_matchingMask(
		struct PtrWrapper ptr);

double Stitcher_panoConfidenceThresh(
		struct PtrWrapper ptr);

double Stitcher_registrationResol(
		struct PtrWrapper ptr);

double Stitcher_seamEstimationResol(
		struct PtrWrapper ptr);

struct PtrWrapper Stitcher_seamFinder(
		struct PtrWrapper ptr);

void Stitcher_setBlender(
		struct PtrWrapper ptr, struct PtrWrapper b);

void Stitcher_setBundleAdjuster(
		struct PtrWrapper ptr, struct PtrWrapper bundle_adjuster);

void Stitcher_setCompositingResol(
		struct PtrWrapper ptr, double resol_mpx);

void Stitcher_setExposureCompensator(
		struct PtrWrapper ptr, struct PtrWrapper exposure_comp);

void Stitcher_setFeaturesFinder(
		struct PtrWrapper ptr, struct PtrWrapper features_finder);

void Stitcher_setFeaturesMatcher(
		struct PtrWrapper ptr, struct PtrWrapper features_matcher);

void Stitcher_setMatchingMask(
		struct PtrWrapper ptr, struct TensorWrapper mask);

void Stitcher_setPanoConfidenceThresh(
		struct PtrWrapper ptr, double conf_thresh);

void Stitcher_setRegistrationResol(
		struct PtrWrapper ptr, double resol_mpx);

void Stitcher_setSeamEstimationResol(
		struct PtrWrapper ptr, double resol_mpx);

void Stitcher_setSeamFinder(
		struct PtrWrapper ptr, struct PtrWrapper seam_finder);

void Stitcher_setWarper(
		struct PtrWrapper ptr, struct PtrWrapper creator);

void Stitcher_setWaveCorrection(
		struct PtrWrapper ptr, bool flag);

void Stitcher_setWaveCorrectKind(
		struct PtrWrapper ptr, int kind);

struct TensorPlusInt Stitcher_stitch(
		struct PtrWrapper ptr, struct TensorArray images);

struct PtrWrapper Stitcher_warper(
		struct PtrWrapper ptr);

bool Stitcher_waveCorrection(
		struct PtrWrapper ptr);

int Stitcher_waveCorrectKind(
		struct PtrWrapper ptr);

double Stitcher_workScale(
		struct PtrWrapper ptr);
]]

--CameraParams

do
    local CameraParams = torch.class('cv.CameraParams', cv);

    function CameraParams:__init(t)
        local argRules = {
            {"other", default = nil}}
        local other = cv.argcheck(t, argRules)
        if other then
            self.ptr = ffi.gc(C.CameraParams_ctor2(other.ptr), C.CameraParams_dtor)
        else
            self.ptr = ffi.gc(C.CameraParams_ctor(), C.CameraParams_dtor)
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

    function Timelapser:__init(t)
        local argRules = {
            {"type", required = true}}
        local type_arg = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.Timelapser_ctor(type_arg), C.Timelapser_dtor)
    end

    function Timelapser:getDst()
        return cv.unwrap_tensors(C.Timelapser_getDst(self.ptr))
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

    function MatchesInfo:__init(t)
        local argRules = {
            {"other", default = nil} }
        local other = cv.argcheck(t, argRules)
        if other then
            self.ptr = ffi.gc(C.MatchesInfo_ctor2(other.ptr), C.MatchesInfo_dtor);
        else
            self.ptr = ffi.gc(C.MatchesInfo_ctor(), C.MatchesInfo_dtor);
        end
    end
end

--FeaturesFinder

do
    local FeaturesFinder = torch.class('cv.FeaturesFinder', cv)

    function FeaturesFinder:collectGarbage()
        C.FeaturesFinder_collectGarbage(self.ptr)
    end

    function FeaturesFinder:__call(t)
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
        self.ptr = ffi.gc(C.SurfFeaturesFinder_ctor(hess_thresh, num_octaves, num_layers,
                                                    num_octaves_descr, num_layers_descr),
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

    function FeaturesMatcher:isThreadSafe()
        return C.FeaturesMatcher_isThreadSafe(self.ptr);
    end

    function FeaturesMatcher:__call(t)
        local argRules = {
            {"features1", required = true},
            {"features2", required = true} }
        local features1, features2 = cv.argcheck(t, argRules)
        local result =  ffi.gc(C.FeaturesMatcher_call(self.ptr, features1.ptr, features2.ptr),
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

    function BestOf2NearestRangeMatcher:__call(t)
        local argRules = {
            {"features", required = true},
            {"pairwise_matches", required = true},
            {"mask", default = nil} }
        local features, pairwise_matches, mask = cv.argcheck(t, argRules)
        C.BestOf2NearestRangeMatcher_call(
            self.ptr, cv.newArray("Class", features), cv.newArray("Class", pairwise_matches), cv.wrap_tensor(mask))
    end
end

--Estimator

do
    local Estimator = torch.class('cv.Estimator', cv)

    function Estimator:__call(t)
        local argRules = {
            {"features", required = true},
            {"pairwise_matches", required = true} }
        local features, pairwise_matches = cv.argcheck(t, argRules)
        local result = C.Estimator_call(self.ptr, cv.newArray("Class", features), cv.newArray("Class", pairwise_matches))
        return result.val, cv.unwrap_class("CameraParams", result.array)
    end
end

--HomographyBasedEstimator

do
    local HomographyBasedEstimator = torch.class('cv.HomographyBasedEstimator', 'cv.Estimator', cv)

    function HomographyBasedEstimator:__init(t)
        local argRules = {
            {"is_focals_estimated", default = false} }
        local is_focals_estimated = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.HomographyBasedEstimator(is_focals_estimated), C.Estimator_dtor)
    end
end

--BundleAdjusterBase

do
    local BundleAdjusterBase = torch.class("cv.BundleAdjusterBase", 'cv.Estimator', cv)

    function BundleAdjusterBase:confThresh(t)
        return C.BundleAdjusterBase_confThresh(self.ptr)
    end

    function BundleAdjusterBase:refinementMask(t)
        return cv.unwrap_tensors(C.BundleAdjusterBase_refinementMask());
    end

    function BundleAdjusterBase:setConfThresh(t)
        local argRules = {
            {"conf_thresh", required = true} }
        local conf_thresh = cv.argcheck(t, argRules)
        C.BundleAdjusterBase_setConfThresh(self.ptr, conf_thresh)
    end

    function BundleAdjusterBase:setRefinementMask(t)
        local argRules ={
            {"mask", required= true} }
        local mask = cv.argcheck(t, argRules)
        C.BundleAdjusterBase_setRefinementMask(cv.wrap_temnsor(mask))
    end

    function BundleAdjusterBase:setTermCriteria(t)
        local argRules = {
            {"term_criteria", required = true}}
        local term_criteria = cv.argcheck(t, argRules)
        C.BundleAdjusterBase_setTermCriteria(self.ptr, term_criteria)
    end

    function BundleAdjusterBase:termCriteria(t)
        return C.BundleAdjusterBase_termCriteria(self.ptr)
    end
end

--BundleAdjusterRay

do
    local BundleAdjusterRay = torch.class('cv.BundleAdjusterRay', "cv.BundleAdjusterBase", cv)

    function BundleAdjusterRay:__init(t)
        self.ptr = ffi.gc(C.BundleAdjusterRay_ctor(), C.Estimator_dtor)
    end
end

--BundleAdjusterReproj

do
    local BundleAdjusterReproj = torch.class('cv.BundleAdjusterReproj', "cv.BundleAdjusterBase", cv)

    function BundleAdjusterReproj:__init(t)
        self.ptr = ffi.gc(C.BundleAdjusterReproj_ctor(), C.Estimator_dtor)
    end
end


--***********************Images Warping***********************


--ProjectorBase
do
    local ProjectorBase = torch.class('cv.ProjectorBase', cv)

    function ProjectorBase:__init(t)
        self.ptr = ffi.gc(C.ProjectorBase_ctor(), C.ProjectorBase_dtor)
    end

    function ProjectorBase:setCameraParams(t)
        local argRules = {
            {"K", default = torch.FloatTensor(3,3):eye(3)},
            {"R", default = torch.FloatTensor(3,3):eye(3)},
            {"T", default = torch.FloatTensor(3,1):zero()}}
        local K, R, T = cv.argcheck(t, argRules)
        C.ProjectorBase_setCameraParams(
                self.ptr, cv.wrap_tensor(K), cv.wrap_tensor(R), cv.wrap_tensor(T))
    end
end

--CompressedRectilinearPortraitProjector

do
    local CompressedRectilinearPortraitProjector =
                torch.class('cv.CompressedRectilinearPortraitProjector', 'cv.ProjectorBase', cv)

    function CompressedRectilinearPortraitProjector:__init(t)
        self.ptr = ffi.gc(C.CompressedRectilinearPortraitProjector_ctor(), C.CompressedRectilinearPortraitProjector_dtor)
    end

    function CompressedRectilinearPortraitProjector:mapBackward(t)
        local argRules = {
            {"u", required = true},
            {"v", required = true} }
        local u, v = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.CompressedRectilinearPortraitProjector_mapBackward(self.ptr, u, v))
        return  result.data[0], result.data[1]
    end

    function CompressedRectilinearPortraitProjector:mapForward(t)
        local argRules = {
            {"x", required = true},
            {"y", required = true} }
        local x, y = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.CompressedRectilinearPortraitProjector_mapForward(self.ptr, x, y))
        return  result.data[0], result.data[1]
    end
end

--CompressedRectilinearProjector

do
    local CompressedRectilinearProjector = torch.class('cv.CompressedRectilinearProjector', 'cv.ProjectorBase', cv)

    function CompressedRectilinearProjector:__init(t)
        self.ptr = ffi.gc(C.CompressedRectilinearProjector_ctor(), C.CompressedRectilinearProjector_dtor)
    end

    function CompressedRectilinearProjector:mapBackward(t)
        local argRules = {
            {"u", required = true},
            {"v", required = true} }
        local u, v = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.CompressedRectilinearProjector_mapBackward(self.ptr, u, v))
        return  result.data[0], result.data[1]
    end

    function CompressedRectilinearProjector:mapForward(t)
        local argRules = {
            {"x", required = true},
            {"y", required = true} }
        local x, y = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.CompressedRectilinearProjector_mapForward(self.ptr, x, y))
        return  result.data[0], result.data[1]
    end
end

--CylindricalPortraitProjector

do
    local CylindricalPortraitProjector = torch.class('cv.CylindricalPortraitProjector', 'cv.ProjectorBase', cv)

    function CylindricalPortraitProjector:__init(t)
        self.ptr = ffi.gc(C.CylindricalPortraitProjector_ctor(), C.CylindricalPortraitProjector_dtor)
    end

    function CylindricalPortraitProjector:mapBackward(t)
        local argRules = {
            {"u", required = true},
            {"v", required = true} }
        local u, v = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.CylindricalPortraitProjector_mapBackward(self.ptr, u, v))
        return  result.data[0], result.data[1]
    end

    function CylindricalPortraitProjector:mapForward(t)
        local argRules = {
            {"x", required = true},
            {"y", required = true} }
        local x, y = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.CylindricalPortraitProjector_mapForward(self.ptr, x, y))
        return  result.data[0], result.data[1]
    end
end

--CylindricalProjector

do
    local CylindricalProjector = torch.class('cv.CylindricalProjector', 'cv.ProjectorBase', cv)

    function CylindricalProjector:__init(t)
        self.ptr = ffi.gc(C.CylindricalProjector_ctor(), C.CylindricalProjector_dtor)
    end

    function CylindricalProjector:mapBackward(t)
        local argRules = {
            {"u", required = true},
            {"v", required = true} }
        local u, v = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.CylindricalProjector_mapBackward(self.ptr, u, v))
        return  result.data[0], result.data[1]
    end

    function CylindricalProjector:mapForward(t)
        local argRules = {
            {"x", required = true},
            {"y", required = true} }
        local x, y = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.CylindricalProjector_mapForward(self.ptr, x, y))
        return  result.data[0], result.data[1]
    end
end

--FisheyeProjector

do
    local FisheyeProjector = torch.class('cv.FisheyeProjector', 'cv.ProjectorBase', cv)

    function FisheyeProjector:__init(t)
        self.ptr = ffi.gc(C.FisheyeProjector_ctor(), C.FisheyeProjector_dtor)
    end

    function FisheyeProjector:mapBackward(t)
        local argRules = {
            {"u", required = true},
            {"v", required = true} }
        local u, v = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.FisheyeProjector_mapBackward(self.ptr, u, v))
        return  result.data[0], result.data[1]
    end

    function FisheyeProjector:mapForward(t)
        local argRules = {
            {"x", required = true},
            {"y", required = true} }
        local x, y = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.FisheyeProjector_mapForward(self.ptr, x, y))
        return  result.data[0], result.data[1]
    end
end

--MercatorProjector

do
    local MercatorProjector = torch.class('cv.MercatorProjector', 'cv.ProjectorBase', cv)

    function MercatorProjector:__init(t)
        self.ptr = ffi.gc(C.MercatorProjector_ctor(), C.MercatorProjector_dtor)
    end

    function MercatorProjector:mapBackward(t)
        local argRules = {
            {"u", required = true},
            {"v", required = true} }
        local u, v = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.MercatorProjector_mapBackward(self.ptr, u, v))
        return  result.data[0], result.data[1]
    end

    function MercatorProjector:mapForward(t)
        local argRules = {
            {"x", required = true},
            {"y", required = true} }
        local x, y = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.MercatorProjector_mapForward(self.ptr, x, y))
        return  result.data[0], result.data[1]
    end
end

--PaniniPortraitProjector

do
    local PaniniPortraitProjector = torch.class('cv.PaniniPortraitProjector', 'cv.ProjectorBase', cv)

    function PaniniPortraitProjector:__init(t)
        self.ptr = ffi.gc(C.PaniniPortraitProjector_ctor(), C.PaniniPortraitProjector_dtor)
    end

    function PaniniPortraitProjector:mapBackward(t)
        local argRules = {
            {"u", required = true},
            {"v", required = true}}
        local u, v = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.PaniniPortraitProjector_mapBackward(self.ptr, u, v))
        return  result.data[0], result.data[1]
    end

    function PaniniPortraitProjector:mapForward(t)
        local argRules = {
            {"x", required = true},
            {"y", required = true}}
        local x, y = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.PaniniPortraitProjector_mapForward(self.ptr, x, y))
        return  result.data[0], result.data[1]
    end
end

--PaniniProjector

do
    local PaniniProjector = torch.class('cv.PaniniProjector', 'cv.ProjectorBase', cv)

    function PaniniProjector:__init(t)
        self.ptr = ffi.gc(C.PaniniProjector_ctor(), C.PaniniProjector_dtor)
    end

    function PaniniProjector:mapBackward(t)
        local argRules = {
            {"u", required = true},
            {"v", required = true}}
        local u, v = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.PaniniProjector_mapBackward(self.ptr, u, v))
        return  result.data[0], result.data[1]
    end

    function PaniniProjector:mapForward(t)
        local argRules = {
            {"x", required = true},
            {"y", required = true}}
        local x, y = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.PaniniProjector_mapForward(self.ptr, x, y))
        return  result.data[0], result.data[1]
    end
end

--PlanePortraitProjector

do
    local PlanePortraitProjector = torch.class('cv.PlanePortraitProjector', 'cv.ProjectorBase', cv)

    function PlanePortraitProjector:__init(t)
        self.ptr = ffi.gc(C.PlanePortraitProjector_ctor(), C.PlanePortraitProjector_dtor)
    end

    function PlanePortraitProjector:mapBackward(t)
        local argRules = {
            {"u", required = true},
            {"v", required = true}}
        local u, v = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.PlanePortraitProjector_mapBackward(self.ptr, u, v))
        return  result.data[0], result.data[1]
    end

    function PlanePortraitProjector:mapForward(t)
        local argRules = {
            {"x", required = true},
            {"y", required = true} }
        local x, y = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.PlanePortraitProjector_mapForward(self.ptr, x, y))
        return  result.data[0], result.data[1]
    end
end

--PlaneProjector

do
    local PlaneProjector = torch.class('cv.PlaneProjector', 'cv.ProjectorBase', cv)

    function PlaneProjector:__init(t)
        self.ptr = ffi.gc(C.PlaneProjector_ctor(), C.PlaneProjector_dtor)
    end

    function PlaneProjector:mapBackward(t)
        local argRules = {
            {"u", required = true},
            {"v", required = true}}
        local u, v = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.PlaneProjector_mapBackward(self.ptr, u, v))
        return  result.data[0], result.data[1]
    end

    function PlaneProjector:mapForward(t)
        local argRules = {
            {"x", required = true},
            {"y", required = true}}
        local x, y = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.PlaneProjector_mapForward(self.ptr, x, y))
        return  result.data[0], result.data[1]
    end
end

--SphericalPortraitProjector

do
    local SphericalPortraitProjector = torch.class('cv.SphericalPortraitProjector', 'cv.ProjectorBase', cv)

    function SphericalPortraitProjector:__init(t)
        self.ptr = ffi.gc(C.SphericalPortraitProjector_ctor(), C.SphericalPortraitProjector_dtor)
    end

    function SphericalPortraitProjector:mapBackward(t)
        local argRules = {
            {"u", required = true},
            {"v", required = true}}
        local u, v = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.SphericalPortraitProjector_mapBackward(self.ptr, u, v))
        return  result.data[0], result.data[1]
    end

    function SphericalPortraitProjector:mapForward(t)
        local argRules = {
            {"x", required = true},
            {"y", required = true}}
        local x, y = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.SphericalPortraitProjector_mapForward(self.ptr, x, y))
        return  result.data[0], result.data[1]
    end
end

--SphericalProjector

do
    local SphericalProjector = torch.class('cv.SphericalProjector', 'cv.ProjectorBase', cv)

    function SphericalProjector:__init(t)
        self.ptr = ffi.gc(C.SphericalProjector_ctor(), C.SphericalProjector_dtor)
    end

    function SphericalProjector:mapBackward(t)
        local argRules = {
            {"u", required = true},
            {"v", required = true}}
        local u, v = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.SphericalProjector_mapBackward(self.ptr, u, v))
        return  result.data[0], result.data[1]
    end

    function SphericalProjector:mapForward(t)
        local argRules = {
            {"x", required = true},
            {"y", required = true}}
        local x, y = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.SphericalProjector_mapForward(self.ptr, x, y))
        return  result.data[0], result.data[1]
    end
end

--StereographicProjector

do
    local StereographicProjector = torch.class('cv.StereographicProjector', 'cv.ProjectorBase', cv)

    function StereographicProjector:__init(t)
        self.ptr = ffi.gc(C.StereographicProjector_ctor(), C.StereographicProjector_dtor)
    end

    function StereographicProjector:mapBackward(t)
        local argRules = {
            {"u", required = true},
            {"v", required = true}}
        local u, v = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.StereographicProjector_mapBackward(self.ptr, u, v))
        return  result.data[0], result.data[1]
    end

    function StereographicProjector:mapForward(t)
        local argRules = {
            {"x", required = true},
            {"y", required = true}}
        local x, y = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.StereographicProjector_mapForward(self.ptr, x, y))
        return  result.data[0], result.data[1]
    end
end

--TransverseMercatorProjector

do
    local TransverseMercatorProjector = torch.class('cv.TransverseMercatorProjector', 'cv.ProjectorBase', cv)

    function TransverseMercatorProjector:__init(t)
        self.ptr = ffi.gc(C.TransverseMercatorProjector_ctor(), C.TransverseMercatorProjector_dtor)
    end

    function TransverseMercatorProjector:mapBackward(t)
        local argRules = {
            {"u", required = true},
            {"v", required = true}}
        local u, v = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.TransverseMercatorProjector_mapBackward(self.ptr, u, v))
        return  result.data[0], result.data[1]
    end

    function TransverseMercatorProjector:mapForward(t)
        local argRules = {
            {"x", required = true},
            {"y", required = true}}
        local x, y = cv.argcheck(t, argRules)
        local result = cv.gcarray(C.TransverseMercatorProjector_mapForward(self.ptr, x, y))
        return  result.data[0], result.data[1]
    end
end

--RotationWarper

do
    local RotationWarper = torch.class('cv.RotationWarper', cv)

    function RotationWarper:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil}}
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.RotationWarper_buildMaps(
                            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
                            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap))
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function RotationWarper:getScale(t)
        return C.RotationWarper_getScale(self.ptr);
    end

    function RotationWarper:setScale(t)
        local argRules = {
            {"val", required= true} }
        local val = cv.argcheck(t, argRules)
        C.RotationWarper_setScale(self.ptr, val)
    end

    function RotationWarper:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil}}
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.RotationWarper_warp(
                            self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
                            interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end

    function RotationWarper:warpBackward(t)
        local argRules = {
            {"src", required = true},
            {"K", required = K},
            {"R", required = R},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil} }
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        return cv.unwrap_tensors(
                    C.RotationWarper_warpBackward(self.ptr, cv.wrap_tensor(K), cv.wrap_tensor(R),
                                                  interp_mode, border_mode, cv.wrap_tensor(dst)))
    end

    function RotationWarper:warpPoint(t)
        local argRules = {
            {"pt", required = true, operator = cv.Point2f},
            {"K", required = true},
            {"R", required = true} }
        local pt, K, R = cv.argcheck(t, argRules)
        return C.RotationWarper_warpPoint(
                    self.ptr, pt, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

    function RotationWarper:warpRoi(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true} }
        local src_size, K, R = cv.argcheck(t, argRules)
        return C.RotationWarper_warpRoi(self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

end

--RotationWarperBase_CompressedRectilinearPortraitProjector

do
    local RotationWarperBase_CompressedRectilinearPortraitProjector =
                torch.class('cv.RotationWarperBase_CompressedRectilinearPortraitProjector', 'cv.RotationWarper', cv)

    function RotationWarperBase_CompressedRectilinearPortraitProjector:__init(t)
        self.ptr = ffi.gc(C.RotationWarperBase_CompressedRectilinearPortraitProjector_ctor(), C.RotationWarper_dtor)
    end

    function RotationWarperBase_CompressedRectilinearPortraitProjector:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil}}
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_CompressedRectilinearPortraitProjector_buildMaps(
            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap))
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function RotationWarperBase_CompressedRectilinearPortraitProjector:getScale(t)
        return C.RotationWarperBase_CompressedRectilinearPortraitProjector_getScale(self.ptr);
    end

    function RotationWarperBase_CompressedRectilinearPortraitProjector:setScale(t)
        local argRules = {
            {"val", required= true} }
        local val = cv.argcheck(t, argRules)
        C.RotationWarperBase_CompressedRectilinearPortraitProjector_setScale(self.ptr, val)
    end

    function RotationWarperBase_CompressedRectilinearPortraitProjector:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil}}
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_CompressedRectilinearPortraitProjector_warp(
            self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
            interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end

    function RotationWarperBase_CompressedRectilinearPortraitProjector:warpBackward(t)
        local argRules = {
            {"src", required = true},
            {"K", required = K},
            {"R", required = R},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil} }
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        return cv.unwrap_tensors(
            C.RotationWarperBase_CompressedRectilinearPortraitProjector_warpBackward(
                        self.ptr, cv.wrap_tensor(K), cv.wrap_tensor(R),
                interp_mode, border_mode, cv.wrap_tensor(dst)))
    end

    function RotationWarperBase_CompressedRectilinearPortraitProjector:warpPoint(t)
        local argRules = {
            {"pt", required = true, operator = cv.Point2f},
            {"K", required = true},
            {"R", required = true} }
        local pt, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_CompressedRectilinearPortraitProjector_warpPoint(
            self.ptr, pt, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

    function RotationWarperBase_CompressedRectilinearPortraitProjector:warpRoi(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true} }
        local src_size, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_CompressedRectilinearPortraitProjector_warpRoi(
                    self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

end

--RotationWarperBase_CompressedRectilinearProjector

do
    local RotationWarperBase_CompressedRectilinearProjector =
    torch.class('cv.RotationWarperBase_CompressedRectilinearProjector', 'cv.RotationWarper', cv)

    function RotationWarperBase_CompressedRectilinearProjector:__init(t)
        self.ptr = ffi.gc(C.RotationWarperBase_CompressedRectilinearProjector_ctor(), C.RotationWarper_dtor)
    end

    function RotationWarperBase_CompressedRectilinearProjector:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil}}
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_CompressedRectilinearProjector_buildMaps(
            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap))
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function RotationWarperBase_CompressedRectilinearProjector:getScale(t)
        return C.RotationWarperBase_CompressedRectilinearProjector_getScale(self.ptr);
    end

    function RotationWarperBase_CompressedRectilinearProjector:setScale(t)
        local argRules = {
            {"val", required= true} }
        local val = cv.argcheck(t, argRules)
        C.RotationWarperBase_CompressedRectilinearProjector_setScale(self.ptr, val)
    end

    function RotationWarperBase_CompressedRectilinearProjector:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil}}
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_CompressedRectilinearProjector_warp(
            self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
            interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end

    function RotationWarperBase_CompressedRectilinearProjector:warpBackward(t)
        local argRules = {
            {"src", required = true},
            {"K", required = K},
            {"R", required = R},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil} }
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        return cv.unwrap_tensors(
            C.RotationWarperBase_CompressedRectilinearProjector_warpBackward(
                        self.ptr, cv.wrap_tensor(K), cv.wrap_tensor(R),
                interp_mode, border_mode, cv.wrap_tensor(dst)))
    end

    function RotationWarperBase_CompressedRectilinearProjector:warpPoint(t)
        local argRules = {
            {"pt", required = true, operator = cv.Point2f},
            {"K", required = true},
            {"R", required = true} }
        local pt, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_CompressedRectilinearProjector_warpPoint(
            self.ptr, pt, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

    function RotationWarperBase_CompressedRectilinearProjector:warpRoi(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true} }
        local src_size, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_CompressedRectilinearProjector_warpRoi(
                    self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

end

--RotationWarperBase_CylindricalPortraitProjector

do
    local RotationWarperBase_CylindricalPortraitProjector =
    torch.class('cv.RotationWarperBase_CylindricalPortraitProjector', 'cv.RotationWarper', cv)

    function RotationWarperBase_CylindricalPortraitProjector:__init(t)
        self.ptr = ffi.gc(C.RotationWarperBase_CylindricalPortraitProjector_ctor(), C.RotationWarper_dtor)
    end

    function RotationWarperBase_CylindricalPortraitProjector:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil}}
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_CylindricalPortraitProjector_buildMaps(
            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap))
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function RotationWarperBase_CylindricalPortraitProjector:getScale(t)
        return C.RotationWarperBase_CylindricalPortraitProjector_getScale(self.ptr);
    end

    function RotationWarperBase_CylindricalPortraitProjector:setScale(t)
        local argRules = {
            {"val", required= true} }
        local val = cv.argcheck(t, argRules)
        C.RotationWarperBase_CylindricalPortraitProjector_setScale(self.ptr, val)
    end

    function RotationWarperBase_CylindricalPortraitProjector:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil}}
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_CylindricalPortraitProjector_warp(
            self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
            interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end

    function RotationWarperBase_CylindricalPortraitProjector:warpBackward(t)
        local argRules = {
            {"src", required = true},
            {"K", required = K},
            {"R", required = R},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil} }
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        return cv.unwrap_tensors(
            C.RotationWarperBase_CylindricalPortraitProjector_warpBackward(self.ptr, cv.wrap_tensor(K), cv.wrap_tensor(R),
                interp_mode, border_mode, cv.wrap_tensor(dst)))
    end

    function RotationWarperBase_CylindricalPortraitProjector:warpPoint(t)
        local argRules = {
            {"pt", required = true, operator = cv.Point2f},
            {"K", required = true},
            {"R", required = true} }
        local pt, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_CylindricalPortraitProjector_warpPoint(
            self.ptr, pt, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

    function RotationWarperBase_CylindricalPortraitProjector:warpRoi(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true} }
        local src_size, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_CylindricalPortraitProjector_warpRoi(
                    self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

end

--RotationWarperBase_CylindricalProjector

do
    local RotationWarperBase_CylindricalProjector =
    torch.class('cv.RotationWarperBase_CylindricalProjector', 'cv.RotationWarper', cv)

    function RotationWarperBase_CylindricalProjector:__init(t)
        self.ptr = ffi.gc(C.RotationWarperBase_CylindricalProjector_ctor(), C.RotationWarper_dtor)
    end

    function RotationWarperBase_CylindricalProjector:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil}}
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_CylindricalProjector_buildMaps(
            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap))
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function RotationWarperBase_CylindricalProjector:getScale(t)
        return C.RotationWarperBase_CylindricalProjector_getScale(self.ptr);
    end

    function RotationWarperBase_CylindricalProjector:setScale(t)
        local argRules = {
            {"val", required= true} }
        local val = cv.argcheck(t, argRules)
        C.RotationWarperBase_CylindricalProjector_setScale(self.ptr, val)
    end

    function RotationWarperBase_CylindricalProjector:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil}}
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_CylindricalProjector_warp(
            self.ptr,cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
            interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end

    function RotationWarperBase_CylindricalProjector:warpBackward(t)
        local argRules = {
            {"src", required = true},
            {"K", required = K},
            {"R", required = R},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil} }
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        return cv.unwrap_tensors(
            C.RotationWarperBase_CylindricalProjector_warpBackward(self.ptr, cv.wrap_tensor(K), cv.wrap_tensor(R),
                interp_mode, border_mode, cv.wrap_tensor(dst)))
    end

    function RotationWarperBase_CylindricalProjector:warpPoint(t)
        local argRules = {
            {"pt", required = true, operator = cv.Point2f},
            {"K", required = true},
            {"R", required = true} }
        local pt, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_CylindricalProjector_warpPoint(
            self.ptr, pt, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

    function RotationWarperBase_CylindricalProjector:warpRoi(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true} }
        local src_size, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_CylindricalProjector_warpRoi(
                    self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end
end

--RotationWarperBase_FisheyeProjector

do
    local RotationWarperBase_FisheyeProjector =
    torch.class('cv.RotationWarperBase_FisheyeProjector', 'cv.RotationWarper', cv)

    function RotationWarperBase_FisheyeProjector:__init(t)
        self.ptr = ffi.gc(C.RotationWarperBase_FisheyeProjector_ctor(), C.RotationWarper_dtor)
    end

    function RotationWarperBase_FisheyeProjector:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil}}
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_FisheyeProjector_buildMaps(
            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap))
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function RotationWarperBase_FisheyeProjector:getScale(t)
        return C.RotationWarperBase_FisheyeProjector_getScale(self.ptr);
    end

    function RotationWarperBase_FisheyeProjector:setScale(t)
        local argRules = {
            {"val", required= true} }
        local val = cv.argcheck(t, argRules)
        C.RotationWarperBase_FisheyeProjector_setScale(self.ptr, val)
    end

    function RotationWarperBase_FisheyeProjector:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil}}
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_FisheyeProjector_warp(
            self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
            interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end

    function RotationWarperBase_FisheyeProjector:warpBackward(t)
        local argRules = {
            {"src", required = true},
            {"K", required = K},
            {"R", required = R},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil} }
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        return cv.unwrap_tensors(
            C.RotationWarperBase_FisheyeProjector_warpBackward(self.ptr, cv.wrap_tensor(K), cv.wrap_tensor(R),
                interp_mode, border_mode, cv.wrap_tensor(dst)))
    end

    function RotationWarperBase_FisheyeProjector:warpPoint(t)
        local argRules = {
            {"pt", required = true, operator = cv.Point2f},
            {"K", required = true},
            {"R", required = true} }
        local pt, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_FisheyeProjector_warpPoint(
            self.ptr, pt, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

    function RotationWarperBase_FisheyeProjector:warpRoi(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true} }
        local src_size, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_FisheyeProjector_warpRoi(
                    self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end
end

--RotationWarperBase_MercatorProjector

do
    local RotationWarperBase_MercatorProjector =
    torch.class('cv.RotationWarperBase_MercatorProjector', 'cv.RotationWarper', cv)

    function RotationWarperBase_MercatorProjector:__init(t)
        self.ptr = ffi.gc(C.RotationWarperBase_MercatorProjector_ctor(), C.RotationWarper_dtor)
    end

    function RotationWarperBase_MercatorProjector:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil}}
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_MercatorProjector_buildMaps(
            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap))
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function RotationWarperBase_MercatorProjector:getScale(t)
        return C.RotationWarperBase_MercatorProjector_getScale(self.ptr);
    end

    function RotationWarperBase_MercatorProjector:setScale(t)
        local argRules = {
            {"val", required= true} }
        local val = cv.argcheck(t, argRules)
        C.RotationWarperBase_MercatorProjector_setScale(self.ptr, val)
    end

    function RotationWarperBase_MercatorProjector:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil}}
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_MercatorProjector_warp(
            self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
            interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end

    function RotationWarperBase_MercatorProjector:warpBackward(t)
        local argRules = {
            {"src", required = true},
            {"K", required = K},
            {"R", required = R},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil} }
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        return cv.unwrap_tensors(
            C.RotationWarperBase_MercatorProjector_warpBackward(self.ptr, cv.wrap_tensor(K), cv.wrap_tensor(R),
                interp_mode, border_mode, cv.wrap_tensor(dst)))
    end

    function RotationWarperBase_MercatorProjector:warpPoint(t)
        local argRules = {
            {"pt", required = true, operator = cv.Point2f},
            {"K", required = true},
            {"R", required = true} }
        local pt, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_MercatorProjector_warpPoint(
            self.ptr, pt, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

    function RotationWarperBase_MercatorProjector:warpRoi(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true} }
        local src_size, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_MercatorProjector_warpRoi(
                    self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end
end

--RotationWarperBase_PaniniPortraitProjector

do
    local RotationWarperBase_PaniniPortraitProjector =
    torch.class('cv.RotationWarperBase_PaniniPortraitProjector', 'cv.RotationWarper', cv)

    function RotationWarperBase_PaniniPortraitProjector:__init(t)
        self.ptr = ffi.gc(C.RotationWarperBase_PaniniPortraitProjector_ctor(), C.RotationWarper_dtor)
    end

    function RotationWarperBase_PaniniPortraitProjector:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil}}
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_PaniniPortraitProjector_buildMaps(
            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap))
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function RotationWarperBase_PaniniPortraitProjector:getScale(t)
        return C.RotationWarperBase_PaniniPortraitProjector_getScale(self.ptr);
    end

    function RotationWarperBase_PaniniPortraitProjector:setScale(t)
        local argRules = {
            {"val", required= true} }
        local val = cv.argcheck(t, argRules)
        C.RotationWarperBase_PaniniPortraitProjector_setScale(self.ptr, val)
    end

    function RotationWarperBase_PaniniPortraitProjector:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil}}
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_PaniniPortraitProjector_warp(
            self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
            interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end

    function RotationWarperBase_PaniniPortraitProjector:warpBackward(t)
        local argRules = {
            {"src", required = true},
            {"K", required = K},
            {"R", required = R},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil} }
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        return cv.unwrap_tensors(
            C.RotationWarperBase_PaniniPortraitProjector_warpBackward(
                self.ptr, cv.wrap_tensor(K), cv.wrap_tensor(R),
                interp_mode, border_mode, cv.wrap_tensor(dst)))
    end

    function RotationWarperBase_PaniniPortraitProjector:warpPoint(t)
        local argRules = {
            {"pt", required = true, operator = cv.Point2f},
            {"K", required = true},
            {"R", required = true} }
        local pt, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_PaniniPortraitProjector_warpPoint(
            self.ptr, pt, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

    function RotationWarperBase_PaniniPortraitProjector:warpRoi(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true} }
        local src_size, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_PaniniPortraitProjector_warpRoi(
                    self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end
end

--RotationWarperBase_PaniniProjector

do
    local RotationWarperBase_PaniniProjector =
    torch.class('cv.RotationWarperBase_PaniniProjector', 'cv.RotationWarper', cv)

    function RotationWarperBase_PaniniProjector:__init(t)
        self.ptr = ffi.gc(C.RotationWarperBase_PaniniProjector_ctor(), C.RotationWarper_dtor)
    end

    function RotationWarperBase_PaniniProjector:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil}}
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_PaniniProjector_buildMaps(
            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap))
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function RotationWarperBase_PaniniProjector:getScale(t)
        return C.RotationWarperBase_PaniniProjector_getScale(self.ptr);
    end

    function RotationWarperBase_PaniniProjector:setScale(t)
        local argRules = {
            {"val", required= true} }
        local val = cv.argcheck(t, argRules)
        C.RotationWarperBase_PaniniProjector_setScale(self.ptr, val)
    end

    function RotationWarperBase_PaniniProjector:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil}}
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_PaniniProjector_warp(
            self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
            interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end

    function RotationWarperBase_PaniniProjector:warpBackward(t)
        local argRules = {
            {"src", required = true},
            {"K", required = K},
            {"R", required = R},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil} }
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        return cv.unwrap_tensors(
            C.RotationWarperBase_PaniniProjector_warpBackward(
                self.ptr, cv.wrap_tensor(K), cv.wrap_tensor(R),
                interp_mode, border_mode, cv.wrap_tensor(dst)))
    end

    function RotationWarperBase_PaniniProjector:warpPoint(t)
        local argRules = {
            {"pt", required = true, operator = cv.Point2f},
            {"K", required = true},
            {"R", required = true} }
        local pt, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_PaniniProjector_warpPoint(
            self.ptr, pt, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

    function RotationWarperBase_PaniniProjector:warpRoi(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true} }
        local src_size, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_PaniniProjector_warpRoi(
                    self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end
end

--RotationWarperBase_PlanePortraitProjector

do
    local RotationWarperBase_PlanePortraitProjector =
    torch.class('cv.RotationWarperBase_PlanePortraitProjector', 'cv.RotationWarper', cv)

    function RotationWarperBase_PlanePortraitProjector:__init(t)
        self.ptr = ffi.gc(C.RotationWarperBase_PlanePortraitProjector_ctor(), C.RotationWarper_dtor)
    end

    function RotationWarperBase_PlanePortraitProjector:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil}}
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_PlanePortraitProjector_buildMaps(
            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap))
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function RotationWarperBase_PlanePortraitProjector:getScale(t)
        return C.RotationWarperBase_PlanePortraitProjector_getScale(self.ptr);
    end

    function RotationWarperBase_PlanePortraitProjector:setScale(t)
        local argRules = {
            {"val", required= true} }
        local val = cv.argcheck(t, argRules)
        C.RotationWarperBase_PlanePortraitProjector_setScale(self.ptr, val)
    end

    function RotationWarperBase_PlanePortraitProjector:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil}}
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_PlanePortraitProjector_warp(
            self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
            interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end

    function RotationWarperBase_PlanePortraitProjector:warpBackward(t)
        local argRules = {
            {"src", required = true},
            {"K", required = K},
            {"R", required = R},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil} }
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        return cv.unwrap_tensors(
            C.RotationWarperBase_PlanePortraitProjector_warpBackward(
                self.ptr, cv.wrap_tensor(K), cv.wrap_tensor(R),
                interp_mode, border_mode, cv.wrap_tensor(dst)))
    end

    function RotationWarperBase_PlanePortraitProjector:warpPoint(t)
        local argRules = {
            {"pt", required = true, operator = cv.Point2f},
            {"K", required = true},
            {"R", required = true} }
        local pt, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_PlanePortraitProjector_warpPoint(
            self.ptr, pt, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

    function RotationWarperBase_PlanePortraitProjector:warpRoi(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true} }
        local src_size, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_PlanePortraitProjector_warpRoi(
                    self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end
end

--RotationWarperBase_PlaneProjector

do
    local RotationWarperBase_PlaneProjector =
    torch.class('cv.RotationWarperBase_PlaneProjector', 'cv.RotationWarper', cv)

    function RotationWarperBase_PlaneProjector:__init(t)
        self.ptr = ffi.gc(C.RotationWarperBase_PlaneProjector_ctor(), C.RotationWarper_dtor)
    end

    function RotationWarperBase_PlaneProjector:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil}}
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_PlaneProjector_buildMaps(
            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap))
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function RotationWarperBase_PlaneProjector:getScale(t)
        return C.RotationWarperBase_PlaneProjector_getScale(self.ptr);
    end

    function RotationWarperBase_PlaneProjector:setScale(t)
        local argRules = {
            {"val", required= true} }
        local val = cv.argcheck(t, argRules)
        C.RotationWarperBase_PlaneProjector_setScale(self.ptr, val)
    end

    function RotationWarperBase_PlaneProjector:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil}}
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_PlaneProjector_warp(
            self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
            interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end

    function RotationWarperBase_PlaneProjector:warpBackward(t)
        local argRules = {
            {"src", required = true},
            {"K", required = K},
            {"R", required = R},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil} }
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        return cv.unwrap_tensors(
            C.RotationWarperBase_PlaneProjector_warpBackward(self.ptr, cv.wrap_tensor(K), cv.wrap_tensor(R),
                interp_mode, border_mode, cv.wrap_tensor(dst)))
    end

    function RotationWarperBase_PlaneProjector:warpPoint(t)
        local argRules = {
            {"pt", required = true, operator = cv.Point2f},
            {"K", required = true},
            {"R", required = true} }
        local pt, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_PlaneProjector_warpPoint(
            self.ptr, pt, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

    function RotationWarperBase_PlaneProjector:warpRoi(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true} }
        local src_size, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_PlaneProjector_warpRoi(
                    self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end
end

--RotationWarperBase_SphericalPortraitProjector

do
    local RotationWarperBase_SphericalPortraitProjector =
    torch.class('cv.RotationWarperBase_SphericalPortraitProjector', 'cv.RotationWarper', cv)

    function RotationWarperBase_SphericalPortraitProjector:__init(t)
        self.ptr = ffi.gc(C.RotationWarperBase_SphericalPortraitProjector_ctor(), C.RotationWarper_dtor)
    end

    function RotationWarperBase_SphericalPortraitProjector:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil}}
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_SphericalPortraitProjector_buildMaps(
            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap))
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function RotationWarperBase_SphericalPortraitProjector:getScale(t)
        return C.RotationWarperBase_SphericalPortraitProjector_getScale(self.ptr);
    end

    function RotationWarperBase_SphericalPortraitProjector:setScale(t)
        local argRules = {
            {"val", required= true} }
        local val = cv.argcheck(t, argRules)
        C.RotationWarperBase_SphericalPortraitProjector_setScale(self.ptr, val)
    end

    function RotationWarperBase_SphericalPortraitProjector:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil}}
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_SphericalPortraitProjector_warp(
            self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
            interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end

    function RotationWarperBase_SphericalPortraitProjector:warpBackward(t)
        local argRules = {
            {"src", required = true},
            {"K", required = K},
            {"R", required = R},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil} }
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        return cv.unwrap_tensors(
            C.RotationWarperBase_SphericalPortraitProjector_warpBackward(
                self.ptr, cv.wrap_tensor(K), cv.wrap_tensor(R),
                interp_mode, border_mode, cv.wrap_tensor(dst)))
    end

    function RotationWarperBase_SphericalPortraitProjector:warpPoint(t)
        local argRules = {
            {"pt", required = true, operator = cv.Point2f},
            {"K", required = true},
            {"R", required = true} }
        local pt, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_SphericalPortraitProjector_warpPoint(
            self.ptr, pt, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

    function RotationWarperBase_SphericalPortraitProjector:warpRoi(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true} }
        local src_size, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_SphericalPortraitProjector_warpRoi(
                    self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end
end

--RotationWarperBase_SphericalProjector

do
    local RotationWarperBase_SphericalProjector =
    torch.class('cv.RotationWarperBase_SphericalProjector', 'cv.RotationWarper', cv)

    function RotationWarperBase_SphericalProjector:__init(t)
        self.ptr = ffi.gc(C.RotationWarperBase_SphericalProjector_ctor(), C.RotationWarper_dtor)
    end

    function RotationWarperBase_SphericalProjector:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil}}
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_SphericalProjector_buildMaps(
            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap))
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function RotationWarperBase_SphericalProjector:getScale(t)
        return C.RotationWarperBase_SphericalProjector_getScale(self.ptr);
    end

    function RotationWarperBase_SphericalProjector:setScale(t)
        local argRules = {
            {"val", required= true} }
        local val = cv.argcheck(t, argRules)
        C.RotationWarperBase_SphericalProjector_setScale(self.ptr, val)
    end

    function RotationWarperBase_SphericalProjector:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil}}
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_SphericalProjector_warp(
            self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
            interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end

    function RotationWarperBase_SphericalProjector:warpBackward(t)
        local argRules = {
            {"src", required = true},
            {"K", required = K},
            {"R", required = R},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil} }
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        return cv.unwrap_tensors(
            C.RotationWarperBase_SphericalProjector_warpBackward(
                self.ptr, cv.wrap_tensor(K), cv.wrap_tensor(R),
                interp_mode, border_mode, cv.wrap_tensor(dst)))
    end

    function RotationWarperBase_SphericalProjector:warpPoint(t)
        local argRules = {
            {"pt", required = true, operator = cv.Point2f},
            {"K", required = true},
            {"R", required = true} }
        local pt, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_SphericalProjector_warpPoint(
            self.ptr, pt, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

    function RotationWarperBase_SphericalProjector:warpRoi(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true} }
        local src_size, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_SphericalProjector_warpRoi(
                    self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end
end

--RotationWarperBase_StereographicProjector

do
    local RotationWarperBase_StereographicProjector =
    torch.class('cv.RotationWarperBase_StereographicProjector', 'cv.RotationWarper', cv)

    function RotationWarperBase_StereographicProjector:__init(t)
        self.ptr = ffi.gc(C.RotationWarperBase_StereographicProjector_ctor(), C.RotationWarper_dtor)
    end

    function RotationWarperBase_StereographicProjector:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil}}
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_StereographicProjector_buildMaps(
            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap))
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function RotationWarperBase_StereographicProjector:getScale(t)
        return C.RotationWarperBase_StereographicProjector_getScale(self.ptr);
    end

    function RotationWarperBase_StereographicProjector:setScale(t)
        local argRules = {
            {"val", required= true} }
        local val = cv.argcheck(t, argRules)
        C.RotationWarperBase_StereographicProjector_setScale(self.ptr, val)
    end

    function RotationWarperBase_StereographicProjector:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil}}
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_StereographicProjector_warp(
            self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
            interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end

    function RotationWarperBase_StereographicProjector:warpBackward(t)
        local argRules = {
            {"src", required = true},
            {"K", required = K},
            {"R", required = R},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil} }
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        return cv.unwrap_tensors(
            C.RotationWarperBase_StereographicProjector_warpBackward(
                self.ptr, cv.wrap_tensor(K), cv.wrap_tensor(R),
                interp_mode, border_mode, cv.wrap_tensor(dst)))
    end

    function RotationWarperBase_StereographicProjector:warpPoint(t)
        local argRules = {
            {"pt", required = true, operator = cv.Point2f},
            {"K", required = true},
            {"R", required = true} }
        local pt, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_StereographicProjector_warpPoint(
            self.ptr, pt, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

    function RotationWarperBase_StereographicProjector:warpRoi(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true} }
        local src_size, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_StereographicProjector_warpRoi(
                    self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end
end

--RotationWarperBase_TransverseMercatorProjector

do
    local RotationWarperBase_TransverseMercatorProjector =
    torch.class('cv.RotationWarperBase_TransverseMercatorProjector', 'cv.RotationWarper', cv)

    function RotationWarperBase_TransverseMercatorProjector:__init(t)
        self.ptr = ffi.gc(C.RotationWarperBase_TransverseMercatorProjector_ctor(), C.RotationWarper_dtor)
    end

    function RotationWarperBase_TransverseMercatorProjector:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil}}
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_TransverseMercatorProjector_buildMaps(
            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap))
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function RotationWarperBase_TransverseMercatorProjector:getScale(t)
        return C.RotationWarperBase_TransverseMercatorProjector_getScale(self.ptr);
    end

    function RotationWarperBase_TransverseMercatorProjector:setScale(t)
        local argRules = {
            {"val", required= true} }
        local val = cv.argcheck(t, argRules)
        C.RotationWarperBase_TransverseMercatorProjector_setScale(self.ptr, val)
    end

    function RotationWarperBase_TransverseMercatorProjector:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil}}
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.RotationWarperBase_TransverseMercatorProjector_warp(
            self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
            interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end

    function RotationWarperBase_TransverseMercatorProjector:warpBackward(t)
        local argRules = {
            {"src", required = true},
            {"K", required = K},
            {"R", required = R},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil} }
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        return cv.unwrap_tensors(
            C.RotationWarperBase_TransverseMercatorProjector_warpBackward(self.ptr, cv.wrap_tensor(K), cv.wrap_tensor(R),
                interp_mode, border_mode, cv.wrap_tensor(dst)))
    end

    function RotationWarperBase_TransverseMercatorProjector:warpPoint(t)
        local argRules = {
            {"pt", required = true, operator = cv.Point2f},
            {"K", required = true},
            {"R", required = true} }
        local pt, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_TransverseMercatorProjector_warpPoint(
            self.ptr, pt, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end

    function RotationWarperBase_TransverseMercatorProjector:warpRoi(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true} }
        local src_size, K, R = cv.argcheck(t, argRules)
        return C.RotationWarperBase_TransverseMercatorProjector_warpRoi(
                    self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R))
    end
end

--WarperCreator

do
    local WarperCreator = torch.class('cv.WarperCreator', cv)

    function WarperCreator:create(t)
        local argRules = {
            {"scale", required = true} }
        local scale = cv.argcheck(t, argRules)
        local retval = torch.factory('cv.RotationWarper')()
        retval.ptr = ffi.gc(C.WarperCreator_create(self.ptr, scale), C.RotationWarper_dtor)
        return retval
    end
end

--CompressedRectilinearPortraitWarper

do
    local CompressedRectilinearPortraitWarper = torch.class('cv.CompressedRectilinearPortraitWarper', 'cv.WarperCreator', cv)

    function CompressedRectilinearPortraitWarper:__init(t)
        local argRules = {
            {"A", default = 1},
            {"B", default = 1} }
        local A, B = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.CompressedRectilinearPortraitWarper_ctor(A, B), C.WarperCreator_dtor)
    end

    function CompressedRectilinearPortraitWarper:create(t)
        local argRules = {
            {"scale", required = true} }
        local scale = cv.argcheck(t, argRules)
        local retval = torch.factory('cv.RotationWarper')()
        retval.ptr = ffi.gc(C.CompressedRectilinearPortraitWarper_create(self.ptr, scale), C.RotationWarper_dtor)
        return retval
    end
end

--CompressedRectilinearWarper

do
    local CompressedRectilinearWarper = torch.class('cv.CompressedRectilinearWarper', 'cv.WarperCreator', cv)

    function CompressedRectilinearWarper:__init(t)
        local argRules = {
            {"A", default = 1},
            {"B", default = 1} }
        local A, B = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.CompressedRectilinearWarper_ctor(A, B), C.WarperCreator_dtor)
    end

    function CompressedRectilinearWarper:create(t)
        local argRules = {
            {"scale", required = true} }
        local scale = cv.argcheck(t, argRules)
        local retval = torch.factory('cv.RotationWarper')()
        retval.ptr = ffi.gc(C.CompressedRectilinearWarper_create(self.ptr, scale), C.RotationWarper_dtor)
        return retval
    end
end

--CylindricalWarper

do
    local CylindricalWarper = torch.class('cv.CylindricalWarper', 'cv.WarperCreator', cv)

    function CylindricalWarper:__init(t)
        self.ptr = ffi.gc(C.CylindricalWarper_ctor(), C.WarperCreator_dtor)
    end

    function CylindricalWarper:create(t)
        local argRules = {
            {"scale", required = true} }
        local scale = cv.argcheck(t, argRules)
        local retval = torch.factory('cv.RotationWarper')()
        retval.ptr = ffi.gc(C.CylindricalWarper_create(self.ptr, scale), C.RotationWarper_dtor)
        return retval
    end
end

--FisheyeWarper

do
    local FisheyeWarper = torch.class('cv.FisheyeWarper', 'cv.WarperCreator', cv)

    function FisheyeWarper:__init(t)
        self.ptr = ffi.gc(C.FisheyeWarper_ctor(), C.WarperCreator_dtor)
    end

    function FisheyeWarper:create(t)
        local argRules = {
            {"scale", required = true} }
        local scale = cv.argcheck(t, argRules)
        local retval = torch.factory('cv.RotationWarper')()
        retval.ptr = ffi.gc(C.FisheyeWarper_create(self.ptr, scale), C.RotationWarper_dtor)
        return retval
    end
end

--MercatorWarper

do
    local MercatorWarper = torch.class('cv.MercatorWarper', 'cv.WarperCreator', cv)

    function MercatorWarper:__init(t)
        self.ptr = ffi.gc(C.MercatorWarper_ctor(), C.WarperCreator_dtor)
    end

    function MercatorWarper:create(t)
        local argRules = {
            {"scale", required = true} }
        local scale = cv.argcheck(t, argRules)
        local retval = torch.factory('cv.RotationWarper')()
        retval.ptr = ffi.gc(C.MercatorWarper_create(self.ptr, scale), C.RotationWarper_dtor)
        return retval
    end
end

--PaniniPortraitWarper

do
    local PaniniPortraitWarper = torch.class('cv.PaniniPortraitWarper', 'cv.WarperCreator', cv)

    function PaniniPortraitWarper:__init(t)
        local argRules = {
            {"A", default = 1},
            {"B", default = 1} }
        local A, B = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.PaniniPortraitWarper_ctor(A, B), C.WarperCreator_dtor)
    end

    function PaniniPortraitWarper:create(t)
        local argRules = {
            {"scale", required = true} }
        local scale = cv.argcheck(t, argRules)
        local retval = torch.factory('cv.RotationWarper')()
        retval.ptr = ffi.gc(C.PaniniPortraitWarper_create(self.ptr, scale), C.RotationWarper_dtor)
        return retval
    end
end

--PaniniWarper

do
    local PaniniWarper = torch.class('cv.PaniniWarper', 'cv.WarperCreator', cv)

    function PaniniWarper:__init(t)
        local argRules = {
            {"A", default = 1},
            {"B", default = 1} }
        local A, B = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.PaniniWarper_ctor(A, B), C.WarperCreator_dtor)
    end

    function PaniniWarper:create(t)
        local argRules = {
            {"scale", required = true} }
        local scale = cv.argcheck(t, argRules)
        local retval = torch.factory('cv.RotationWarper')()
        retval.ptr = ffi.gc(C.PaniniWarper_create(self.ptr, scale), C.RotationWarper_dtor)
        return retval
    end
end

--PlaneWarper

do
    local PlaneWarper = torch.class('cv.PlaneWarper', 'cv.WarperCreator', cv)

    function PlaneWarper:__init(t)
        self.ptr = ffi.gc(C.PlaneWarper_ctor(), C.WarperCreator_dtor)
    end

    function PlaneWarper:create(t)
        local argRules = {
            {"scale", required = true} }
        local scale = cv.argcheck(t, argRules)
        local retval = torch.factory('cv.RotationWarper')()
        retval.ptr = ffi.gc(C.PlaneWarper_create(self.ptr, scale), C.RotationWarper_dtor)
        return retval
    end
end

--SphericalWarper

do
    local SphericalWarper = torch.class('cv.SphericalWarper', 'cv.WarperCreator', cv)

    function SphericalWarper:__init(t)
        self.ptr = ffi.gc(C.SphericalWarper_ctor(), C.WarperCreator_dtor)
    end

    function SphericalWarper:create(t)
        local argRules = {
            {"scale", required = true} }
        local scale = cv.argcheck(t, argRules)
        local retval = torch.factory('cv.RotationWarper')()
        retval.ptr = ffi.gc(C.SphericalWarper_create(self.ptr, scale), C.RotationWarper_dtor)
        return retval
    end
end

--StereographicWarper

do
    local StereographicWarper = torch.class('cv.StereographicWarper', 'cv.WarperCreator', cv)

    function StereographicWarper:__init(t)
        self.ptr = ffi.gc(C.StereographicWarper_ctor(), C.WarperCreator_dtor)
    end

    function StereographicWarper:create(t)
        local argRules = {
            {"scale", required = true} }
        local scale = cv.argcheck(t, argRules)
        local retval = torch.factory('cv.RotationWarper')()
        retval.ptr = ffi.gc(C.StereographicWarper_create(self.ptr, scale), C.RotationWarper_dtor)
        return retval
    end
end

--TransverseMercatorWarper

do
    local TransverseMercatorWarper = torch.class('cv.TransverseMercatorWarper', 'cv.WarperCreator', cv)

    function TransverseMercatorWarper:__init(t)
        self.ptr = ffi.gc(C.TransverseMercatorWarper_ctor(), C.WarperCreator_dtor)
    end

    function TransverseMercatorWarper:create(t)
        local argRules = {
            {"scale", required = true} }
        local scale = cv.argcheck(t, argRules)
        local retval = torch.factory('cv.RotationWarper')()
        retval.ptr = ffi.gc(C.TransverseMercatorWarper_create(self.ptr, scale), C.RotationWarper_dtor)
        return retval
    end
end

--detail_CompressedRectilinearPortraitWarper

do
    local detail_CompressedRectilinearPortraitWarper =
        torch.class('cv.detail_CompressedRectilinearPortraitWarper', 'cv.RotationWarperBase_CompressedRectilinearPortraitProjector', cv)

    function detail_CompressedRectilinearPortraitWarper:__init(t)
        local argRules = {
            {"scale", required = true},
            {"A", default = 1},
            {"B", default = 1} }
        local scale, A, B = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.detail_CompressedRectilinearPortraitWarper_ctor(scale, A, B), C.RotationWarper_dtor)
    end
end

--detail_CompressedRectilinearWarper

do
    local detail_CompressedRectilinearWarper =
    torch.class('cv.detail_CompressedRectilinearWarper', 'cv.RotationWarperBase_CompressedRectilinearProjector', cv)

    function detail_CompressedRectilinearWarper:__init(t)
        local argRules = {
            {"scale", required = true},
            {"A", default = 1},
            {"B", default = 1} }
        local scale, A, B = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.detail_CompressedRectilinearWarper_ctor(scale, A, B), C.RotationWarper_dtor)
    end
end

--detail_CylindricalPortraitWarper

do
    local detail_CylindricalPortraitWarper =
    torch.class('cv.detail_CylindricalPortraitWarper', 'cv.RotationWarperBase_CylindricalPortraitProjector', cv)

    function detail_CylindricalPortraitWarper:__init(t)
        local argRules = {
            {"scale", required = true}}
        local scale = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.detail_CylindricalPortraitWarper_ctor(scale), C.RotationWarper_dtor)
    end
end

--detail_CylindricalWarper

do
    local detail_CylindricalWarper =
    torch.class('cv.detail_CylindricalWarper', 'cv.RotationWarperBase_CylindricalProjector', cv)

    function detail_CylindricalWarper:__init(t)
        local argRules = {
            {"scale", required = true}}
        local scale = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.detail_CylindricalWarper_ctor(scale), C.RotationWarper_dtor)
    end

    function detail_CylindricalWarper:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil} }
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.detail_CylindricalWarper_buildMaps(
                            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
                            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap));
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function detail_CylindricalWarper:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required=  true},
            {"dst", default = nil} }
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.detail_CylindricalWarper_warp(
                                self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
                                interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end
end

--detail_CylindricalWarperGpu

do
    local detail_CylindricalWarperGpu =
    torch.class('cv.detail_CylindricalWarperGpu', 'cv.detail_CylindricalWarper', cv)

    function detail_CylindricalWarperGpu:__init(t)
        local argRules = {
            {"scale", required = true}}
        local scale = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.detail_CylindricalWarperGpu_ctor(scale), C.RotationWarper_dtor)
    end

    function detail_CylindricalWarperGpu:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil} }
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.detail_CylindricalWarperGpu_buildMaps(
            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap));
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function detail_CylindricalWarperGpu:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required=  true},
            {"dst", default = nil} }
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.detail_CylindricalWarperGpu_warp(
            self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
            interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end
end

--detail_FisheyeWarper

do
    local detail_FisheyeWarper =
    torch.class('cv.detail_FisheyeWarper', 'cv.RotationWarperBase_FisheyeProjector', cv)

    function detail_FisheyeWarper:__init(t)
        local argRules = {
            {"scale", required = true}}
        local scale = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.detail_FisheyeWarper_ctor(scale), C.RotationWarper_dtor)
    end
end

--detail_MercatorWarper

do
    local detail_MercatorWarper =
    torch.class('cv.detail_MercatorWarper', 'cv.RotationWarperBase_MercatorProjector', cv)

    function detail_MercatorWarper:__init(t)
        local argRules = {
            {"scale", required = true}}
        local scale = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.detail_MercatorWarper_ctor(scale), C.RotationWarper_dtor)
    end
end

--detail_PaniniPortraitWarper

do
    local detail_PaniniPortraitWarper =
    torch.class('cv.detail_PaniniPortraitWarper', 'cv.RotationWarperBase_PaniniPortraitProjector', cv)

    function detail_PaniniPortraitWarper:__init(t)
        local argRules = {
            {"scale", required = true},
            {"A", required = true},
            {"B", required = true}}
        local scale, A, B = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.detail_PaniniPortraitWarper_ctor(scale, A, B), C.RotationWarper_dtor)
    end
end

--detail_PaniniWarper

do
    local detail_PaniniWarper =
    torch.class('cv.detail_PaniniWarper', 'cv.RotationWarperBase_PaniniProjector', cv)

    function detail_PaniniWarper:__init(t)
        local argRules = {
            {"scale", required = true},
            {"A", required = true},
            {"B", required = true}}
        local scale, A, B = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.detail_PaniniWarper_ctor(scale, A, B), C.RotationWarper_dtor)
    end
end

--detail_PlanePortraitWarper

do
    local detail_PlanePortraitWarper =
    torch.class('cv.detail_PlanePortraitWarper', 'cv.RotationWarperBase_PlanePortraitProjector', cv)

    function detail_PlanePortraitWarper:__init(t)
        local argRules = {
            {"scale", required = true}}
        local scale = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.detail_PlanePortraitWarper_ctor(scale), C.RotationWarper_dtor)
    end
end

--detail_PlaneWarper

do
    local detail_PlaneWarper =
    torch.class('cv.detail_PlaneWarper', 'cv.RotationWarperBase_PlaneProjector', cv)

    function detail_PlaneWarper:__init(t)
        local argRules = {
            {"scale", default = 1}}
        local scale = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.detail_PlaneWarper_ctor(scale), C.RotationWarper_dtor)
    end

    function detail_PlaneWarper:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"T", default  = nil},
            {"xmap", default = nil},
            {"ymap", default = nil} }
        local src_size, K, R, T, xmap, ymap = cv.argcheck(t, argRules)
        if T then
            local result = C.detail_PlaneWarper_buildMaps(
                                self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
                                cv.wrap_tensor(T), cv.wrap_tensor(xmap), cv.wrap_tensor(ymap));
            return result.rect, cv.unwrap_tensors(result.tensors)
        else
            local result = C.detail_PlaneWarper_buildMaps2(
                self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
                cv.wrap_tensor(xmap), cv.wrap_tensor(ymap))
            return result.rect, cv.unwrap_tensors(result.tensors)
        end
    end

    function detail_PlaneWarper:warp(t)

        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"T", default = nil},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil}}
        local src, K, R, T, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        if T then
            local result = C.detail_PlaneWarper_warp(
                                self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
                                cv.wrap_tensor(T), interp_mode, border_mode, cv.wrap_tensor(dst))
            return result.point, cv.unwrap_tensors(result.tensor)
        else
            local result = C.detail_PlaneWarper_warp2(
                self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
                interp_mode, border_mode, cv.wrap_tensor(dst))
            return result.point, cv.unwrap_tensors(result.tensor)
        end
    end

    function detail_PlaneWarper:warpPoint(t)
        local argRules = {
            {"pt", required = true, operator = cv.Point2f},
            {"K", required = true},
            {"R", required = true},
            {"T", default = nil}}
        local pt, K, R, T = cv.argcheck(t, argRules)
        if T then
            return C.detail_PlaneWarper_warpPoint(
                            self.ptr, pt, cv.wrap_tensor(K), cv.wrap_tensor(R),
                            cv.wrap_tensor(T))
        else
            return C.detail_PlaneWarper_warpPoint2(
                            self.ptr, pt, cv.wrap_tensor(K),
                            cv.wrap_tensor(R))
        end
    end

    function detail_PlaneWarper:warpRoi(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"T", default = nil}}
        local src_size, K, R, T = cv.argcheck(t, argRules)
        if T then
            return C.detail_PlaneWarper_warpRoi(self.ptr, src_size, cv.wrap_tensor(K),
                                                cv.wrap_tensor(R), cv.wrap_tensor(T))
        else
            return C.detail_PlaneWarper_warpRoi2(self.ptr, src_size, cv.wrap_tensor(K),
                                                 cv.wrap_tensor(R))
        end
    end
end

--detail_SphericalPortraitWarper

do
    local detail_SphericalPortraitWarper =
    torch.class('cv.detail_SphericalPortraitWarper', 'cv.RotationWarperBase_SphericalPortraitProjector', cv)

    function detail_SphericalPortraitWarper:__init(t)
        local argRules = {
            {"scale", required = true}}
        local scale = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.detail_SphericalPortraitWarper_ctor(scale), C.RotationWarper_dtor)
    end
end

--detail_SphericalWarper

do
    local detail_SphericalWarper =
    torch.class('cv.detail_SphericalWarper', 'cv.RotationWarperBase_SphericalProjector', cv)

    function detail_SphericalWarper:__init(t)
        local argRules = {
            {"scale", required = true}}
        local scale = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.detail_SphericalWarper_ctor(scale), C.RotationWarper_dtor)
    end

    function detail_SphericalWarper:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil}}
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.detail_SphericalWarper_buildMaps(
            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap))
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function detail_SphericalWarper:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil}}
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.detail_SphericalWarper_warp(
            self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
            interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end
end

--detail_SphericalWarperGpu

do
    local detail_SphericalWarperGpu =
    torch.class('cv.detail_SphericalWarperGpu', 'cv.detail_SphericalWarper', cv)

    function detail_SphericalWarperGpu:__init(t)
        local argRules = {
            {"scale", required = true}}
        local scale = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.detail_SphericalWarperGpu_ctor(scale), C.RotationWarper_dtor)
    end

    function detail_SphericalWarperGpu:buildMaps(t)
        local argRules = {
            {"src_size", required = true, operator = cv.Size},
            {"K", required = true},
            {"R", required = true},
            {"xmap", default = nil},
            {"ymap", default = nil}}
        local src_size, K, R, xmap, ymap = cv.argcheck(t, argRules)
        local result = C.detail_SphericalWarperGpu_buildMaps(
            self.ptr, src_size, cv.wrap_tensor(K), cv.wrap_tensor(R),
            cv.wrap_tensor(xmap), cv.wrap_tensor(ymap))
        return result.rect, cv.unwrap_tensors(result.tensors)
    end

    function detail_SphericalWarperGpu:warp(t)
        local argRules = {
            {"src", required = true},
            {"K", required = true},
            {"R", required = true},
            {"interp_mode", required = true},
            {"border_mode", required = true},
            {"dst", default = nil}}
        local src, K, R, interp_mode, border_mode, dst = cv.argcheck(t, argRules)
        local result = C.detail_SphericalWarperGpu_warp(
            self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(K), cv.wrap_tensor(R),
            interp_mode, border_mode, cv.wrap_tensor(dst))
        return result.point, cv.unwrap_tensors(result.tensor)
    end
end

--detail_StereographicWarper

do
    local detail_StereographicWarper =
    torch.class('cv.detail_StereographicWarper', 'cv.RotationWarperBase_StereographicProjector', cv)

    function detail_StereographicWarper:__init(t)
        local argRules = {
            {"scale", required = true}}
        local scale = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.detail_SphericalWarperGpu_ctor(scale), C.RotationWarper_dtor)
    end
end

--detail_TransverseMercatorWarper

do
    local detail_TransverseMercatorWarper =
    torch.class('cv.detail_TransverseMercatorWarper', 'cv.RotationWarperBase_TransverseMercatorProjector', cv)

    function detail_TransverseMercatorWarper:__init(t)
        local argRules = {
            {"scale", required = true}}
        local scale = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.detail_TransverseMercatorWarper_ctor(scale), C.RotationWarper_dtor)
    end
end


--************************Seam Estimation******************************

cv.COLOR = 0
cv.COLOR_GRAD = 1
cv.COST_COLOR = 0
cv.COST_COLOR_GRAD = 1

--SeamFinder

do
    local SeamFinder = torch.class('cv.SeamFinder', cv)

    function SeamFinder:find(t)
        local argRules = {
            {"src", required = true},
            {"corners", required = true},
            {"masks", required = true}}
        local src, corners, masks = cv.argcheck(t, argRules)
        C.SeamFinder_find(self.ptr, cv.wrap_tensors(src), corners, cv.wrap_tensors(masks))
    end
end

--DpSeamFinder

do
    local DpSeamFinder = torch.class('cv.DpSeamFinder', 'cv.SeamFinder', cv)

    function DpSeamFinder:__init(t)
        local argRules = {
            {"costFunc", default = cv.COLOR}}
        local costFunc = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.DpSeamFinder_ctor(costFunc), C.SeamFinder_dtor)
    end

    function DpSeamFinder:costFunction()
        return C.DpSeamFinder_costFunction(self.ptr)
    end

    function DpSeamFinder:find(t)
        local argRules = {
            {"src", required = true},
            {"corners", required = true},
            {"masks", required = true}}
        local src, corners, masks = cv.argcheck(t, argRules)
        C.DpSeamFinder_find(self.ptr, cv.wrap_tensors(src), corners, cv.wrap_tensors(masks))
    end

    function DpSeamFinder:setCostFunction(t)
        local argRules = {
            {"val", required = true} }
        local val = cv.argcheck(t, argRules)
        C.DpSeamFinder_setCostFunction(self.ptr, val)
    end
end

--GraphCutSeamFinder

do
    local GraphCutSeamFinder = torch.class('cv.GraphCutSeamFinder', 'cv.SeamFinder', cv)

    function GraphCutSeamFinder:__init(t)
        local argRules = {
            {"cost_type", default = cv.COST_COLOR_GRAD},
            {"terminal_cost", default = 10000},
            {"bad_region_penalty", default = 1000} }
        local cost_type, terminal_cost, bad_region_penalty = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.GraphCutSeamFinder_ctor(cost_type, terminal_cost, bad_region_penalty),
                          C.GraphCutSeamFinder_dtor)
    end

    function GraphCutSeamFinder:find(t)
        local argRules = {
            {"src", required = true},
            {"corners", required = true},
            {"masks", required = true}}
        local src, corners, masks = cv.argcheck(t, argRules)
        C.GraphCutSeamFinder_find(self.ptr, cv.wrap_tensors(src), corners, cv.wrap_tensors(masks))
    end
end

--NoSeamFinder

do
    local NoSeamFinder = torch.class('cv.NoSeamFinder', 'cv.SeamFinder', cv)

    function NoSeamFinder:__init(t)
        self.ptr = ffi.gc(C.NoSeamFinder_ctor(), C.SeamFinder_dtor)
    end

    function NoSeamFinder:find(t)
        local argRules = {
            {"src", required = true},
            {"corners", required = true},
            {"masks", required = true}}
        local src, corners, masks = cv.argcheck(t, argRules)
        C.NoSeamFinder_find(self.ptr, cv.wrap_tensors(src), corners, cv.wrap_tensors(masks))
    end
end

--PairwiseSeamFinder

do
    local PairwiseSeamFinder = torch.class('cv.PairwiseSeamFinder', 'cv.SeamFinder', cv)

    function PairwiseSeamFinder:find(t)
        local argRules = {
            {"src", required = true},
            {"corners", required = true},
            {"masks", required = true}}
        local src, corners, masks = cv.argcheck(t, argRules)
        C.PairwiseSeamFinder_find(self.ptr, cv.wrap_tensors(src), corners, cv.wrap_tensors(masks))
    end
end

--VoronoiSeamFinder

do
    local VoronoiSeamFinder = torch.class('cv.VoronoiSeamFinder', 'cv.PairwiseSeamFinder', cv)

    function VoronoiSeamFinder:__init(t)
        self.ptr = ffi.gc(C.VoronoiSeamFinder_ctor(), C.SeamFinder_dtor)
    end

    function VoronoiSeamFinder:find(t)
        local argRules = {
            {"src", required = true},
            {"corners", required = true},
            {"masks", required = true}}
        local src, corners, masks = cv.argcheck(t, argRules)
        C.VoronoiSeamFinder_find(self.ptr, cv.wrap_tensors(src), corners, cv.wrap_tensors(masks))
    end

    function VoronoiSeamFinder:find2(t)
        local argRules = {
            {"size", required = true},
            {"corners", required = true},
            {"masks", required = true}}
        local size, corners, masks = cv.argcheck(t, argRules)
        C.VoronoiSeamFinder_find2(self.ptr, size, corners, cv.wrap_tensors(masks))
    end
end

--ExposureCompensator

do
    local ExposureCompensator = torch.class('cv.ExposureCompensator', cv)

    function ExposureCompensator:__init(t)
        local argRules = {
            {"type", required = true} }
        local type = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.ExposureCompensator_ctor(type), C.ExposureCompensator_dtor)
    end

    function ExposureCompensator:apply(t)
        local argRules = {
            {"index", required = true},
            {"corner", required = true, operator = cv.Point},
            {"image", required = true},
            {"mask", required = true} }
        local index, corner, image, mask = cv.argcheck(t, argRules)
        C.ExposureCompensator_apply(self.ptr, index, corner, cv.wrap_tensor(image), cv.wrap_tensor(mask))
    end

    function ExposureCompensator:feed(t)
        local argRules = {
            {"corners", required = true},
            {"images", required= true},
            {"masks", required = true} }
        local corners, images, masks = cv.argcheck(t, argRules)
        C.ExposureCompensator_feed(self.ptr, corners, cv.wrap_tensors(images),
                                   cv.wrap_tensors(masks));
    end
end

--BlocksGainCompensator

do
    local BlocksGainCompensator = torch.class('cv.BlocksGainCompensator', 'cv.ExposureCompensator', cv)

    function BlocksGainCompensator:__init(t)
        local argRules = {
            {"bl_width", default = 32},
            {"bl_height", default = 32} }
        local bl_width, bl_height = cv.argcheck(t, argRules)
        self.ptr = C.BlocksGainCompensator_ctor(bl_width, bl_height)
    end

    function BlocksGainCompensator:apply(t)
        local argRules = {
            {"index", required = true},
            {"corner", required = true, operator = cv.Point},
            {"image", required = true},
            {"mask", required = true} }
        local index, corner, image, mask = cv.argcheck(t, argRules)
        C.BlocksGainCompensator_apply(self.ptr, index, corner, cv.wrap_tensor(image),
                                      cv.wrap_tensor(mask))
    end

    function BlocksGainCompensator:feed(t)
        local argRules = {
            {"corners", required = true},
            {"images", required= true},
            {"mat", required = true},
            {"chr", required = true}}
        local corners, images, mat, chr = cv.argcheck(t, argRules)
        C.BlocksGainCompensator_feed(self.ptr, corners, cv.wrap_tensors(images),
                                     cv.wrap_tensors(mat), chr);
    end
end

--GainCompensator

do
    local GainCompensator = torch.class('cv.GainCompensator', 'cv.ExposureCompensator', cv)

    function GainCompensator:__init()
        self.ptr = C.GainCompensator_ctor()
    end

    function GainCompensator:apply(t)
        local argRules = {
            {"index", required = true},
            {"corner", required = true, operator = cv.Point},
            {"image", required = true},
            {"mask", required = true} }
        local index, corner, image, mask = cv.argcheck(t, argRules)
        C.GainCompensator_apply(self.ptr, index, corner, cv.wrap_tensor(image), cv.wrap_tensor(mask))
    end

    function GainCompensator:feed(t)
        local argRules = {
            {"corners", required = true},
            {"images", required= true},
            {"mat", required = true},
            {"chr", required = true}}
        local corners, images, mat, chr = cv.argcheck(t, argRules)
        C.GainCompensator_feed(self.ptr, corners, cv.wrap_tensors(images),
            cv.wrap_tensors(mat), chr);
    end

    function GainCompensator:gains()
        return C.GainCompensator_gains(self.ptr)
    end
end

--NoExposureCompensator

do
    local NoExposureCompensator = torch.class('cv.NoExposureCompensator', 'cv.ExposureCompensator', cv)

    function NoExposureCompensator:__init()
        self.ptr = C.NoExposureCompensator_ctor()
    end

    function NoExposureCompensator:apply(t)
        local argRules = {
            {"index", required = true},
            {"corner", required = true, operator = cv.Point},
            {"image", required = true},
            {"mask", required = true} }
        local index, corner, image, mask = cv.argcheck(t, argRules)
        C.NoExposureCompensator_apply(self.ptr, index, corner, cv.wrap_tensor(image), cv.wrap_tensor(mask))
    end

    function NoExposureCompensator:feed(t)
        local argRules = {
            {"corners", required = true},
            {"images", required= true},
            {"mat", required = true},
            {"chr", required = true}}
        local corners, images, mat, chr = cv.argcheck(t, argRules)
        C.NoExposureCompensator_feed(self.ptr, corners, cv.wrap_tensors(images),
            cv.wrap_tensors(mat), chr);
    end
end


--*******************************Image Blenders**********************

cv.OK = 0
cv.ERR_NEED_MORE_IMGS = 1
cv.ERR_HOMOGRAPHY_EST_FAIL = 2
cv.ERR_CAMERA_PARAMS_ADJUST_FAIL = 3

--Blender

do
    local Blender = torch.class('cv.Blender', cv)

    function Blender:__init(t)
        local argRules = {
            {"type", required = true},
            {"try_gpu", default = false} }
        local type, try_gpu = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.Blender_ctor(type, try_gpu), C.Blender_dtor)
    end

    function Blender:blend(t)
        local argRules = {
            {"dst", required = true},
            {"dst_mask", required = true} }
        local dst, dst_mat = cv.argcheck(t, argRules)
        C.Blender_blend(self.ptr, cv.wrap_tensor(dst), cv.wrap_tensor(dst_mat))
    end

    function Blender:feed(t)
        local argRules = {
            {"img", required = true},
            {"mask", required = true},
            {"tl", required = true, operator = cv.Point} }
        local img, mask, tl = cv.argcheck(t, argRules)
        C.Blender_feed(self.ptr, cv.wrap_tensor(img), cv.wrap_tensor(mask), tl)
    end

    function Blender:prepare(t)
        local argRules = {
            {"dst_roi", required = true, operator = cv.Rect}}
        local dst_roi = cv.argcheck(t, argRules)
        C.Blender_prepare(self.ptr, dst_roi)
    end

    function Blender:prepare2(t)
        local argRules = {
            {"corners", required = true},
            {"sizes", required = true} }
        local corners, sizes = cv.argcheck(t, argRules)
        C.Blender_prepare2(self.ptr, corners, sizes)
    end
end

--FeatherBlender

do
    local FeatherBlender = torch.class('cv.FeatherBlender', 'cv.Blender', cv)

    function FeatherBlender:__init(t)
        local argRules = {
            {"sharpness", default = 0.02} }
        local sharpness = cv.argcheck(t, argRules)
        self.ptr = C.FeatherBlender_ctor(sharpness)
    end

    function FeatherBlender:blend(t)
        local argRules = {
            {"dst", required = true},
            {"dst_mask", required = true} }
        local dst, dst_mat = cv.argcheck(t, argRules)
        C.FeatherBlender_blend(self.ptr, cv.wrap_tensor(dst), cv.wrap_tensor(dst_mat))
    end

    function FeatherBlender:createWeightMaps(t)
        local argRules ={
            {"masks", required = true},
            {"corners", required = true},
            {"weight_maps", requird = true} }
        local masks, corners, weight_maps = cv.argcheck(t, argRules)

        assert(#masks == #weight_maps)

        return C.FeatherBlender_createWeightMaps(
                    self.ptr, cv.wrap_tensors(masks), corners,
                    cv.wrap_tensors(weight_maps))
    end

    function FeatherBlender:feed(t)
        local argRules = {
            {"img", required = true},
            {"mask", required = true},
            {"tl", required = true, operator = cv.Point} }
        local img, mask, tl = cv.argcheck(t, argRules)
        C.FeatherBlender_feed(self.ptr, cv.wrap_tensor(img), cv.wrap_tensor(mask), tl)
    end

    function FeatherBlender:prepare(t)
        local argRules = {
            {"dst_roi", required = true, operator = cv.Rect}}
        local dst_roi = cv.argcheck(t, argRules)
        C.FeatherBlender_prepare(self.ptr, dst_roi)
    end

    function FeatherBlender:setSharpness(t)
        local argRules = {
            {"val", required = true} }
        local val = cv.argcheck(t, argRules);
        C.FeatherBlender_setSharpness(self.ptr, val)
    end

    function FeatherBlender:sharpness()
        return C.FeatherBlender_sharpness(self.ptr);
    end

end

--MultiBandBlender

do
    local MultiBandBlender = torch.class('cv.MultiBandBlender', 'cv.Blender', cv)

    function MultiBandBlender:__init(t)
        local argRules = {
            {"try_gpu", default = false},
            {"num_bands", default = 5},
            {"weight_type", default = cv.CV_32F}}
        local try_gpu, num_bands, weight_type = cv.argcheck(t, argRules)
        self.ptr = C.MultiBandBlender_ctor(try_gpu, num_bands, weight_type)
    end

    function MultiBandBlender:blend(t)
        local argRules = {
            {"dst", required = true},
            {"dst_mask", required = true} }
        local dst, dst_mat = cv.argcheck(t, argRules)
        C.MultiBandBlender_blend(self.ptr, cv.wrap_tensor(dst), cv.wrap_tensor(dst_mat))
    end

    function MultiBandBlender:feed(t)
        local argRules = {
            {"img", required = true},
            {"mask", required = true},
            {"tl", required = true, operator = cv.Point} }
        local img, mask, tl = cv.argcheck(t, argRules)
        C.MultiBandBlender_feed(self.ptr, cv.wrap_tensor(img), cv.wrap_tensor(mask), tl)
    end

    function MultiBandBlender:numBands()
        return C.MultiBandBlender_numBands(self.ptr)
    end

    function MultiBandBlender:prepare(t)
        local argRules = {
            {"dst_roi", required = true, operator = cv.Rect}}
        local dst_roi = cv.argcheck(t, argRules)
        C.MultiBandBlender_prepare(self.ptr, dst_roi)
    end

    function MultiBandBlender:setNumBands(t)
        local argRules = {
            {"val", required= true} }
        local val = cv.argcheck(t, argRules)
        C.MultiBandBlender_setNumBands(self.ptr, val)
    end
end

--Stitcher

do
    local Stitcher = torch.class('cv.Stitcher', cv)

    function Stitcher:__init(t)
        local argRules = {
            {"try_use_gpu", default = false}}
        local try_use_gpu = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(C.Stitcher_ctor(try_use_gpu), C.Stitcher_dtor)
    end

    function Stitcher:blender()
        local retval = torch.factory('cv.Blender')()
        retval.ptr = ffi.gc(C.Stitcher_blender(self.ptr), C.Blender_dtor)
        return retval
    end

    function Stitcher:bundleAdjuster()
        local retval = torch.factory('cv.BundleAdjusterBase')()
        retval.ptr = ffi.gc(C.Stitcher_bundleAdjuster(self.ptr), C.Estimator_dtor)
        return retval
    end

    function Stitcher:cameras()
        return cv.unwrap_class("CameraParams", C.Stitcher_cameras(self.ptr))
    end

    function Stitcher:component()
        return cv.gcarray(C.Stitcher_component(self.ptr))
    end

    function Stitcher:composePanorama(t)
        local  argRules = {
            {"images", default = nil} }
        local images = cv.argcheck(t, argRules)
        if images then
            local result = C.Stitcher_composePanorama2(self.ptr, cv.wrap_tensors(images))
        else
            local result = C.Stitcher_composePanorama(self.ptr)
        end
        return result.val, cv.unwrap_tensors(result.tensor)
    end

    function Stitcher:compositingResol()
        return C.Stitcher_compositingResol(self.ptr)
    end

    function Stitcher:createDefault(t)
        local argRules = {
            {"try_use_gpu", default = false} }
        local try_use_gpu = cv.argcheck(t, argRules)
        local retval = torch.factory('cv.Stitcher')()
        retval.ptr = ffi.gc(C.Stitcher_createDefault(self.ptr, try_use_gpu), C.Blender_dtor)
        return retval
    end

    function Stitcher:estimateTransform(t)
        local argRules = {
            {"images", required = true} }
        local images = cv.argcheck(t, argRules)
        return C.Stitcher_estimateTransform(self.ptr, cv.wrap_tensors(images))
    end

    function Stitcher:exposureCompensator()
        local retval = torch.factory('cv.ExposureCompensator')()
        retval.ptr = ffi.gc(C.Stitcher_exposureCompensator(self.ptr), C.ExposureCompensator_dtor)
        return retval
    end

    function Stitcher:featuresFinder() --TODO
        local retval = torch.factory('cv.featuresFinder')()
        retval.ptr = ffi.gc(C.Stitcher_featuresFinder(self.ptr), C.FeaturesFinder_dtor)
        return retval
    end

    function Stitcher:featuresMatcher()
        local retval = torch.factory('cv.FeaturesMatcher')()
        retval.ptr = ffi.gc(C.Stitcher_featuresMatcher(self.ptr), C.FeaturesMatcher_dtor)
        return retval
    end

    function Stitcher:matchingMask()
        return cv.unwrap_tensors(C.Stitcher_matchingMask(self.ptr));
    end

    function Stitcher:panoConfidenceThresh()
        return C.Stitcher_panoConfidenceThresh(self.ptr)
    end

    function Stitcher:registrationResol()
        return C.Stitcher_registrationResol(self.ptr)
    end

    function Stitcher:seamEstimationResol()
        return C.Stitcher_seamEstimationResol(self.ptr)
    end

    function Stitcher:setBlender(t)
        local argRules = {
            {"b", required = true} }
        local b = cv.argcheck(t, argRules)
        C.Stitcher_setBlender(self.ptr, b.ptr)
    end

    function Stitcher:setBundleAdjuster(t)
        local argRules = {
            {"bundle_adjuster", required = true} }
        local bundle_adjuster = cv.argcheck(t, argRules)
        C.Stitcher_setBundleAdjuster(self.ptr, bundle_adjuster.ptr)
    end

    function Stitcher:setCompositingResol(t)
        local argRules = {
            {"resol_mpx", required = true} }
        local resol_mpx = cv.argcheck(t, argRules)
        C.Stitcher_setCompositingResol(self.ptr, resol_mpx)
    end

    function Stitcher:setExposureCompensator(t)
        local argRules = {
            {"exposure_comp", required = true} }
        local exposure_comp = cv.argcheck(t, argRules)
        C.Stitcher_setExposureCompensator(self.ptr, exposure_comp.ptr)
    end

    function Stitcher:setFeaturesFinder(t)
        local argRules = {
            {"features_finder", required = true} }
        local features_finder = cv.argcheck()
        C.Stitcher_setFeaturesFinder(self.ptr, features_finder.ptr)
    end

    function Stitcher:setFeaturesMatcher(t)
        local argRules = {
            {"features_matcher", required = true} }
        local features_matcher = cv.argcheck()
        C.Stitcher_setFeaturesMatcher(self.ptr, features_matcher.ptr)
    end

    function Stitcher:setMatchingMask(t)
        local argRules = {
            {"mask", required = true} }
        local mask = cv.argcheck(t, argRules)
        C.Stitcher_setMatchingMask(self.ptr, cv.wrap_tensor(mask))
    end

    function Stitcher:setPanoConfidenceThresh(t)
        local argRules = {
            {"conf_thresh", required = true} }
        local conf_thresh = cv.argcheck(t, argRules)
        C.Stitcher_setPanoConfidenceThresh(self.ptr, conf_thresh)
    end

    function Stitcher:setRegistrationResol(t)
        local argRules = {
            {"resol_mpx", required = true} }
        local resol_mpx = cv.argcheck(t, argRules)
        C.Stitcher_setRegistrationResol(self.ptr, resol_mpx)
    end

    function Stitcher:setSeamEstimationResol(t)
        local argRules = {
            {"resol_mpx", required = true} }
        local resol_mpx = cv.argcheck(t, argRules)
        C.Stitcher_setSeamEstimationResol(self.ptr, resol_mpx)
    end

    function Stitcher:setSeamFinder(t)
        local argRules = {
            {"seam_finder", required = true} }
        local seam_finder = cv.argcheck(t, argRules)
        C.Stitcher_setSeamFinder(self.ptr, seam_finder.ptr)
    end

    function Stitcher:setWarper(t)
        local argRules = {
            {"creator", required = true} }
        local creator = cv.argcheck(t, argRules)
        C.Stitcher_setWarper(self.ptr, creator.ptr)
    end

    function Stitcher:setWaveCorrection(t)
        local argRules = {
            {"flag", required = true} }
        local flag = cv.argcheck(t, argRules)
        C.Stitcher_setWaveCorrection(self.ptr, flag)
    end

    function Stitcher:setWaveCorrectKind(t)
        local argRules = {
            {"kind", required = true} }
        local kind = cv.argcheck(t, argRules)
        C.Stitcher_setWaveCorrectKind(self.ptr, kind)
    end

    function Stitcher:stitch(t)
        local argRules = {
            {"images", required = true}}
        local images = cv.argcheck(t, argRules)
        local  result = C.Stitcher_stitch(self.ptr, cv.wrap_tensors(images))
        return result.val, cv.unwrap_tensors(result.tensor)
    end

    --TODO add 2nd stitch

    function Stitcher:warper()
        local retval = torch.factory('cv.WarperCreator')()
        retval.ptr = ffi.gc(C.Stitcher_warper(self.ptr), C.WarperCreator_dtor)
        return retval
    end

    function Stitcher:waveCorrection()
        return C.Stitcher_waveCorrection(self.ptr)
    end

    function Stitcher:waveCorrectKind()
        return C.Stitcher_waveCorrectKind(self.ptr)
    end

    function Stitcher:workScale()
        return C.Stitcher_workScale(self.ptr)
    end
end


return cv