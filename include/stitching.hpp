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
void detail_selectRandomSubset(
	int count, int size, struct IntArray subset);

extern "C"
int detail_stitchingLogLevel();

/************Rotation Estimation*******/



/****************** Classes ******************/

//CameraParams 

extern "C"
struct CameraParamsPtr {
    void *ptr;

    inline cv::detail::CameraParams * operator->() {
			return static_cast<cv::detail::CameraParams *>(ptr); }
    inline CameraParamsPtr(cv::detail::CameraParams *ptr) { this->ptr = ptr; }
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

    inline cv::detail::DisjointSets * operator->() {
			return static_cast<cv::detail::DisjointSets *>(ptr); }
    inline DisjointSetsPtr(cv::detail::DisjointSets *ptr) { this->ptr = ptr; }
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

    inline cv::detail::Graph * operator->() {
			return static_cast<cv::detail::Graph *>(ptr); }
    inline GraphPtr(cv::detail::Graph *ptr) { this->ptr = ptr; }
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

    inline cv::detail::GraphEdge * operator->() {
			return static_cast<cv::detail::GraphEdge *>(ptr); }
    inline GraphEdgePtr(cv::detail::GraphEdge *ptr) { this->ptr = ptr; }
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

    inline cv::detail::Timelapser * operator->() {
			return static_cast<cv::detail::Timelapser *>(ptr); }
    inline TimelapserPtr(cv::detail::Timelapser * ptr) { this->ptr = ptr; }
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

    inline cv::detail::TimelapserCrop * operator->() {
			return static_cast<cv::detail::TimelapserCrop *>(ptr); }
    inline TimelapserCropPtr(cv::detail::TimelapserCrop *ptr) { this->ptr = ptr; }
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

    inline cv::detail::FeaturesFinder * operator->() {
			return static_cast<cv::detail::FeaturesFinder *>(ptr); }
    inline FeaturesFinderPtr(cv::detail::FeaturesFinder *ptr) { this->ptr = ptr; }
};

void FeaturesFinder_dtor(
	struct FeaturesFinderPtr ptr);

void FeaturesFinder_collectGarbage(
	struct FeaturesFinderPtr ptr);


