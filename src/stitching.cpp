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
void detail_selectRandomSubset(
	int count, int size, struct IntArray subset)
{
    std::vector<int> subset_vec;
    detail::selectRandomSubset(count, size, subset.toIntList(subset_vec));
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

//FeaturesFinder

void FeaturesFinder_dtor(
	struct FeaturesFinderPtr ptr)
{
    ptr->~FeaturesFinder();
}

void FeaturesFinder_collectGarbage(
	struct FeaturesFinderPtr ptr)
{
    ptr->collectGarbage();
}






