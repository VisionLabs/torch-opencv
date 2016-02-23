#include <stitching.hpp>

extern "C"
struct RectPlusBool detail_overlapRoi(
    struct PointWrapper tl1, struct PointWrapper tl2,
    struct SizeWrapper sz1, struct SizeWrapper sz2)
{
    struct RectPlusBool result;
    cv::Rect roi;
    result.val = detail::overlapRoi(tl1, tl2, sz1, sz2, roi);
    new(&result.rect) RectWrapper(roi);
    return result;
}

extern "C"
struct RectWrapper detail_resultRoi(
	struct PointArray corners,
	struct SizeArray sizes)
{
    return RectWrapper(detail::resultRoi(corners, sizes));
}	

extern "C"
struct RectWrapper detail_resultRoiIntersection(
	struct PointArray corners,
	struct SizeArray sizes)
{
    return RectWrapper(detail::resultRoiIntersection(corners, sizes));
}

extern "C"
struct PointWrapper detail_resultTl(
	struct PointArray corners)
{
    return PointWrapper(detail::resultTl(corners));
}

extern "C"
struct IntArray detail_selectRandomSubset(
	int count, int size)
{
    std::vector<int> subset;
    detail::selectRandomSubset(count, size, subset);
    return IntArray(subset);
}

extern "C"
int detail_stitchingLogLevel()
{
    return detail::stitchingLogLevel();
}	

/****************** Classes ******************/

//CameraParams

extern "C"
struct CameraParamsPtr CameraParams_ctor()
{
    return new cv::detail::CameraParams();
}

extern "C"
struct CameraParamsPtr CameraParams_ctor2(
	struct CameraParamsPtr other)
{
    cv::detail::CameraParams * instant = static_cast<cv::detail::CameraParams *>(other.ptr);
    return new cv::detail::CameraParams(*instant);
}

extern "C"
void CameraParams_dtor(
	struct CameraParamsPtr ptr)
{
    delete static_cast<cv::detail::CameraParams *>(ptr.ptr);
}

struct TensorWrapper CameraParams_K(
	struct CameraParamsPtr ptr)
{
    return TensorWrapper(ptr->K());
}

//TODO need to add const CameraParams& cv::detail::CameraParams::operator=(const CameraParams & other)	

//DisjointSets

extern "C"
struct DisjointSetsPtr DisjointSets_ctor(
	int elem_count)
{
    return new cv::detail::DisjointSets(elem_count);
}

extern "C"
void DisjointSets_dtor(
	struct DisjointSetsPtr ptr)
{
    delete static_cast<cv::detail::DisjointSets *>(ptr.ptr);
}

extern "C"
void DisjointSets_createOneElemSets(
	struct DisjointSetsPtr ptr, int elem_count)
{
    ptr->createOneElemSets(elem_count);
}

extern "C"
int DisjointSets_findSetByElem(
	struct DisjointSetsPtr ptr, int elem)
{
    return ptr->findSetByElem(elem);
}

extern "C"
int DisjointSets_mergeSets(
	struct DisjointSetsPtr ptr,
	int set1, int set2)
{
    return ptr->mergeSets(set1, set2);
}

//Graph

extern "C"
struct GraphPtr Graph_ctor(
	int num_vertices)
{
    return new cv::detail::Graph(num_vertices);
}

extern "C"
void Graph_dtor(
	struct GraphPtr ptr)
{
    delete static_cast<cv::detail::Graph *>(ptr.ptr);
}

extern "C"
void Graph_addEdge(
	struct GraphPtr ptr, int from, int to, float weight)
{
    ptr->addEdge(from, to, weight);
}

extern "C"
void Graph_create(
	struct GraphPtr ptr, int num_vertices)
{
    ptr->create(num_vertices);
}

extern "C"
int Graph_numVertices(
	struct GraphPtr ptr)
{
    return ptr->numVertices();
}

//GraphEdge

extern "C"
struct GraphEdgePtr GraphEdge_ctor(
	int from, int to, float weight)
{
    return new cv::detail::GraphEdge(from, to, weight);
}

extern "C"
void GraphEdge_dtor(
	struct GraphEdgePtr ptr)
{
    delete static_cast<cv::detail::GraphEdge *>(ptr.ptr);
}

//Timelapser

extern "C"
struct TimelapserPtr Timelapser_ctor(
	int type)
{
    return rescueObjectFromPtr(
			cv::detail::Timelapser::createDefault(type));
}

extern "C"
void Timelapser_dtor(
	struct TimelapserPtr ptr)
{
    ptr->~Timelapser();
}

extern "C"
void Timelapser_initialize(
	struct TimelapserPtr ptr, struct PointArray corners,
	struct SizeArray sizes)
{
    ptr->initialize(corners, sizes);
}

extern "C"
void Timelapser_process(
	struct TimelapserPtr ptr, struct TensorWrapper img,
	struct TensorWrapper mask, struct PointWrapper tl)
{
    ptr->process(img.toMat(), mask.toMat(), tl);
}

//TimelapserCrop

extern "C"
void TimelapserCrop_initialize(
	struct TimelapserCropPtr ptr, struct PointArray corners,
	struct SizeArray sizes)
{
    ptr->initialize(corners, sizes);
}

//Features Finding and Images Matching

//MatchesInfo

extern "C"
struct MatchesInfoPtr MatchesInfo_ctor()
{
    return new cv::detail::MatchesInfo();
}

extern "C"
struct MatchesInfoPtr MatchesInfo_ctor2(
        struct MatchesInfoPtr other)
{
    new cv::detail::MatchesInfo(*static_cast<cv::detail::MatchesInfo *>(other.ptr));
}

extern "C"
void MatchesInfo_dtor(
        struct MatchesInfoPtr ptr)
{
    std::cout<< "d_tor" << std::endl;
    delete static_cast<cv::detail::MatchesInfo *>(ptr.ptr);
}


//****************Features Finding and Images Matching************


//FeaturesFinder

extern "C"
void FeaturesFinder_dtor(
	struct FeaturesFinderPtr ptr)
{
    ptr->~FeaturesFinder();
    delete static_cast<cv::detail::FeaturesFinder *>(ptr.ptr);
}

extern "C"
void FeaturesFinder_collectGarbage(
	struct FeaturesFinderPtr ptr)
{
    ptr->collectGarbage();
}

extern "C"
struct ImageFeaturesPtr FeaturesFinder_call(
        struct FeaturesFinderPtr ptr, struct TensorWrapper image)
{
    detail::ImageFeatures *features = new cv::detail::ImageFeatures();
    ptr->operator()(image.toMat(), *features);
    return ImageFeaturesPtr(features);
}

extern "C"
struct ImageFeaturesPtr FeaturesFinder_call2(
        struct FeaturesFinderPtr ptr, struct TensorWrapper image,
        struct RectArray rois)
{
    detail::ImageFeatures *features = new cv::detail::ImageFeatures();
    ptr->operator()(image.toMat(), *features, rois);
    return ImageFeaturesPtr(features);
}

//OrbFeaturesFinder

extern "C"
struct OrbFeaturesFinderPtr OrbFeaturesFinder_ctor(
        struct SizeWrapper _grid_size, int nfeatures, float scaleFactor, int nlevels)
{
    new cv::detail::OrbFeaturesFinder(_grid_size, nfeatures, scaleFactor, nlevels);
}

//SurfFeaturesFinder

extern "C"
struct SurfFeaturesFinderPtr SurfFeaturesFinder_ctor(
        double hess_thresh, int num_octaves, int num_layers, int num_octaves_descr, int num_layers_descr)
{
    new cv::detail::SurfFeaturesFinder(hess_thresh, num_octaves, num_layers, num_octaves_descr, num_layers_descr);
}

//ImageFeatures

extern "C"
struct ImageFeaturesPtr ImageFeatures_ctor()
{
    return new cv::detail::ImageFeatures();
}

extern "C"
struct ImageFeaturesPtr ImageFeatures_dtor(
        struct ImageFeaturesPtr ptr)
{
    delete static_cast<cv::detail::ImageFeatures *>(ptr.ptr);
}

//FeaturesMatcher

//TODO need to do constructor protected

extern "C"
void FeaturesMatcher_dtor(
        struct FeaturesMatcherPtr ptr)
{
    ptr->~FeaturesMatcher();
    delete static_cast<cv::detail::FeaturesMatcher *>(ptr.ptr);
}

extern "C"
void FeaturesMatcher_collectGarbage(
        struct FeaturesMatcherPtr ptr)
{
    ptr->collectGarbage();
}

extern "C"
bool FeaturesMatcher_isThreadSafe(
        struct FeaturesMatcherPtr ptr)
{
    return ptr->isThreadSafe();
}

extern "C"
struct MatchesInfoPtr FeaturesMatcher_call(
        struct FeaturesMatcherPtr ptr, struct ImageFeaturesPtr features1,
        struct ImageFeaturesPtr features2)
{
    cv::detail::MatchesInfo *Mat_inf = new cv::detail::MatchesInfo();
    ptr->operator()(*static_cast<cv::detail::ImageFeatures *>(features1.ptr),
                    *static_cast<cv::detail::ImageFeatures *>(features2.ptr), *Mat_inf);
    return Mat_inf;
}

//BestOf2NearestMatcher

extern "C"
struct BestOf2NearestMatcherPtr BestOf2NearestMatcher_ctor(
        bool try_use_gpu, float match_conf,
        int num_matches_thresh1, int num_matches_thresh2)
{
    new cv::detail::BestOf2NearestMatcher(try_use_gpu, match_conf, num_matches_thresh1, num_matches_thresh2);
}

extern "C"
void BestOf2NearestMatcher_collectGarbage(
        struct BestOf2NearestMatcherPtr ptr)
{
   ptr->collectGarbage();
}

//BestOf2NearestRangeMatcher

extern "C"
struct BestOf2NearestRangeMatcherPtr BestOf2NearestRangeMatcher_ctor(
        int range_width, bool try_use_gpu, float match_conf,
        int num_matches_thresh1, int num_matches_thresh2)
{
    return new cv::detail::BestOf2NearestRangeMatcher(range_width, try_use_gpu, match_conf,
                                                      num_matches_thresh1, num_matches_thresh2);
}

//**********************Rotation Estimation********************************

extern "C"
struct GraphPtrPlusIntArray detail_findMaxSpanningTree(
        int num_images, struct ClassArray pairwise_matches)
{
    GraphPtrPlusIntArray result;
    cv::detail::Graph *span_tree = new cv::detail::Graph();
    std::vector<int> centers;
    cv::detail::findMaxSpanningTree(num_images, pairwise_matches, *span_tree, centers);
    result.graph = GraphPtr(span_tree);
    result.array = IntArray(centers);
    return result;
}

extern "C"
struct IntArray detail_leaveBiggestComponent(
        struct ClassArray features, struct ClassArray pairwise_matches, float conf_threshold)
{
    std::vector<cv::detail::ImageFeatures> features_vec = features;
    std::vector<cv::detail::MatchesInfo> pairwise_matches_vec = pairwise_matches;

    return IntArray(
               cv::detail::leaveBiggestComponent(
                       features_vec,
                       pairwise_matches_vec,
                       conf_threshold));
}

extern "C"
struct StringWrapper detail_matchesGraphAsString(
        struct StringArray pathes, struct ClassArray pairwise_matches, float conf_threshold)
{
    struct StringWrapper result;
    std::vector<cv::String> pathes_vec = pathes;
    std::vector<cv::detail::MatchesInfo> pairwise_matches_vec = pairwise_matches;
    cv::String retval = cv::detail::matchesGraphAsString(pathes_vec, pairwise_matches_vec, conf_threshold);
    result.str = retval.c_str();
    return result;
}

extern "C"
void detail_waveCorrect(
        struct TensorArray rmats, int kind)
{
    cv::detail::WaveCorrectKind enum_kind;
    if(kind == 0) enum_kind = cv::detail::WAVE_CORRECT_HORIZ;
    else enum_kind = cv::detail::WAVE_CORRECT_VERT;
    std::vector<cv::Mat> rmats_vec = rmats.toMatList();
    cv::detail::waveCorrect(rmats_vec, enum_kind);
}

//********************************************************
//*************************test***************************
//********************************************************

extern "C"
struct StringArray test(struct StringArray str){

    std::cout << "__C++__\n";

    std::vector<cv::String> vec = str;

    for(int i = 0; i < vec.size(); i++){
        std::cout << vec[i];
    }
    std::cout << std::endl;

    return StringArray(vec);
 }

//********************************************************
//*************************test***************************
//********************************************************
