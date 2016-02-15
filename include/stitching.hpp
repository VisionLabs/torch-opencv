#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/stitching.hpp>
#include "opencv2/stitching/detail/timelapsers.hpp"

namespace detail = cv::detail;

extern "C"
struct RectPlusBool detail_overlapRoi(
	struct PointWrapper tl1, struct PointWrapper tl2,
	struct SizeWrapper sz1, struct SizeWrapper sz2);

//TODO need to add 2-nd Rect cv::detail::resultRoi

extern "C"
struct RectWrapper detail_resultRoi(
	struct PointArray corners,
	struct SizeArray sizes);

extern "C"
struct RectWrapper detail_resultRoiIntersection(
	struct PointArray corners,
	struct SizeArray sizes);

extern "C"
struct PointWrapper detail_resultTl(
	struct PointArray corners);

extern "C"
struct IntArray detail_selectRandomSubset(
	int count, int size);

extern "C"
int detail_stitchingLogLevel();

/************Rotation Estimation*******/



/****************** Classes ******************/

//CameraParams 

extern "C"
struct CameraParamsPtr {
    void *ptr;

    inline detail::CameraParams * operator->() {
			return static_cast<detail::CameraParams *>(ptr); }
    inline CameraParamsPtr(detail::CameraParams *ptr) { this->ptr = ptr; }
};

extern "C"
struct CameraParamsPtr CameraParams_ctor();

extern "C"
struct CameraParamsPtr CameraParams_ctor2(
	struct CameraParamsPtr other);

extern "C"
void CameraParams_dtor(
	struct CameraParamsPtr ptr);

struct TensorWrapper CameraParams_K(
	struct CameraParamsPtr ptr);

//DisjointSets

extern "C"
struct DisjointSetsPtr {
    void *ptr;

    inline detail::DisjointSets * operator->() {
			return static_cast<detail::DisjointSets *>(ptr); }
    inline DisjointSetsPtr(detail::DisjointSets *ptr) { this->ptr = ptr; }
};

extern "C"
struct DisjointSetsPtr DisjointSets_ctor(
	int elem_count);

extern "C"
void DisjointSets_dtor(
	struct DisjointSetsPtr ptr);

extern "C"
void DisjointSets_createOneElemSets(
	struct DisjointSetsPtr ptr, int elem_count);

extern "C"
int DisjointSets_findSetByElem(
	struct DisjointSetsPtr ptr, int elem);

extern "C"
int DisjointSets_mergeSets(
	struct DisjointSetsPtr ptr,
	int set1, int set2);

//Graph

extern "C"
struct GraphPtr {
    void *ptr;

    inline detail::Graph * operator->() {
			return static_cast<detail::Graph *>(ptr); }
    inline GraphPtr(detail::Graph *ptr) { this->ptr = ptr; }
};

extern "C"
struct GraphPtr Graph_ctor(
	int num_vertices);

extern "C"
void Graph_dtor(
	struct GraphPtr ptr);

extern "C"
void Graph_addEdge(
	struct GraphPtr ptr, int from, int to, float weight);

extern "C"
void Graph_create(
	struct GraphPtr ptr, int num_vertices);

//TODO need to add template<typename B> B forEach(B body) const

extern "C"
int Graph_numVertices(
	struct GraphPtr ptr);

//TODO need to add template<typename B> B walkBreadthFirst (int from, B body) const

//GraphEdge

extern "C"
struct GraphEdgePtr {
    void *ptr;

    inline detail::GraphEdge * operator->() {
			return static_cast<detail::GraphEdge *>(ptr); }
    inline GraphEdgePtr(detail::GraphEdge *ptr) { this->ptr = ptr; }
};

extern "C"
struct GraphEdgePtr GraphEdge_ctor(
	int from, int to, float weight);

extern "C"
void GraphEdge_dtor(
	struct GraphEdgePtr ptr);

//Timelapser

extern "C"
struct TimelapserPtr {
    void *ptr;

    inline detail::Timelapser * operator->() {
			return static_cast<detail::Timelapser *>(ptr); }
    inline TimelapserPtr(detail::Timelapser * ptr) { this->ptr = ptr; }
};

extern "C"
struct TimelapserPtr Timelapser_ctor(
	int type);

extern "C"
void Timelapser_dtor(
	struct TimelapserPtr ptr);

//TODO need to add virtual const UMat & getDst()

extern "C"
void Timelapser_initialize(
	struct TimelapserPtr ptr, struct PointArray corners,
	struct SizeArray sizes);

extern "C"
void Timelapser_process(
	struct TimelapserPtr ptr, struct TensorWrapper img,
	struct TensorWrapper mask, struct PointWrapper tl);

//TimelapserCrop

extern "C"
struct TimelapserCropPtr {
    void *ptr;

    inline detail::TimelapserCrop * operator->() {
			return static_cast<detail::TimelapserCrop *>(ptr); }
    inline TimelapserCropPtr(detail::TimelapserCrop *ptr) { this->ptr = ptr; }
};

extern "C"
void TimelapserCrop_initialize(
	struct TimelapserCropPtr ptr, struct PointArray corners,
	struct SizeArray sizes); 

//Features Finding and Images Matching

//FeaturesFinder

extern "C"
struct FeaturesFinderPtr {
    void *ptr;

    inline detail::FeaturesFinder * operator->() {
			return static_cast<detail::FeaturesFinder *>(ptr); }
    inline FeaturesFinderPtr(detail::FeaturesFinder *ptr) { this->ptr = ptr; }
};

extern "C"
void FeaturesFinder_dtor(
	struct FeaturesFinderPtr ptr);

extern "C"
void FeaturesFinder_collectGarbage(
	struct FeaturesFinderPtr ptr);

extern "C"
struct ImageFeaturesPtr FeaturesFinder_call(
		struct FeaturesFinderPtr ptr, struct TensorWrapper image);

extern "C"
struct ImageFeaturesPtr FeaturesFinder_call2(
		struct FeaturesFinderPtr ptr, struct TensorWrapper image,
		struct RectArray);

//ImageFeatures

extern "C"
struct ImageFeaturesPtr {
	void *ptr;

	inline detail::ImageFeatures * operator->() {
		return static_cast<detail::ImageFeatures *>(ptr); }
	inline ImageFeaturesPtr(detail::ImageFeatures *ptr) { this->ptr = ptr; }
};

extern "C"
struct ImageFeaturesPtr ImageFeatures_ctor();

extern "C"
struct ImageFeaturesPtr ImageFeatures_dtor(
		struct ImageFeaturesPtr ptr);

//FeaturesMatcher

extern "C"
struct FeaturesMatcherPtr {
	void *ptr;

	inline detail::FeaturesMatcher * operator->() {
		return static_cast<detail::FeaturesMatcher *>(ptr); }
	inline FeaturesMatcherPtr(detail::FeaturesMatcher *ptr) { this->ptr = ptr; }
};

extern "C"
void FeaturesMatcher_dtor(
		struct FeaturesMatcherPtr ptr);

extern "C"
void FeaturesMatcher_FeaturesMatcher(
		struct FeaturesMatcherPtr ptr);


//BestOf2NearestMatcher

extern "C"
struct BestOf2NearestMatcherPtr {
	void *ptr;

	inline detail::BestOf2NearestMatcher * operator->() {
		return static_cast<detail::BestOf2NearestMatcher *>(ptr); }
	inline BestOf2NearestMatcherPtr(detail::BestOf2NearestMatcher *ptr) { this->ptr = ptr; }
};

extern "C"
struct BestOf2NearestMatcherPtr BestOf2NearestMatcher_ctor(
	bool try_use_gpu, float match_conf,
	int num_matches_thresh1, int num_matches_thresh2);

extern "C"
struct BestOf2NearestMatcherPtr BestOf2NearestMatcher_dtor(
	struct BestOf2NearestMatcherPtr ptr);

extern "C"
void BestOf2NearestMatcher_collectGarbage(
	struct BestOf2NearestMatcherPtr ptr);