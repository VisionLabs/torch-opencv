#include <stitching.hpp>

ClassArray::ClassArray(const std::vector<detail::MatchesInfo> & vec)
{
    MatchesInfoPtr *temp = static_cast<MatchesInfoPtr *>(malloc(vec.size() * sizeof(MatchesInfoPtr)));

    this->size = vec.size();

    MatchesInfoPtr class_wrapped;

    for (int i = 0; i < vec.size(); i++) {
        class_wrapped.ptr = new detail::MatchesInfo(vec[i]);;
        temp[i] = class_wrapped;
    }
    this->data = temp;
}

ClassArray::ClassArray(const std::vector<detail::ImageFeatures> & vec)
{
    ImageFeaturesPtr *temp = static_cast<ImageFeaturesPtr *>(malloc(vec.size() * sizeof(ImageFeaturesPtr)));

    this->size = vec.size();

    ImageFeaturesPtr class_wrapped;

    for (int i = 0; i < vec.size(); i++) {
        class_wrapped.ptr = new detail::ImageFeatures(vec[i]);;
        temp[i] = class_wrapped;
    }

    this->data = temp;
}

ClassArray::ClassArray(const std::vector<detail::CameraParams> & vec)
{
    CameraParamsPtr *temp = static_cast<CameraParamsPtr *>(malloc(vec.size() * sizeof(CameraParamsPtr)));

    this->size = vec.size();

    CameraParamsPtr class_wrapped;

    for (int i = 0; i < vec.size(); i++) {
        class_wrapped.ptr = new detail::CameraParams(vec[i]);;
        temp[i] = class_wrapped;
    }
    this->data = temp;
}

ClassArray::operator std::vector<detail::MatchesInfo>()
{
    MatchesInfoPtr *temp =
            static_cast<MatchesInfoPtr *>(this->data);

    std::vector<detail::MatchesInfo> retval(this->size);

    for(int i = 0; i < this->size; i++) {
        retval[i] = *static_cast<detail::MatchesInfo *>(temp[i].ptr);
        //memcpy(retval.data() + i, temp[i].ptr, sizeof(detail::MatchesInfo));
    }
    return retval;
}

ClassArray::operator std::vector<detail::ImageFeatures>()
{
    ImageFeaturesPtr *temp =
            static_cast<ImageFeaturesPtr *>(this->data);

    std::vector<detail::ImageFeatures> retval(this->size);

    for(int i = 0; i < this->size; i++) {
        retval[i] = *static_cast<detail::ImageFeatures *>(temp[i].ptr);
        //memcpy(retval.data() + i, temp[i].ptr, sizeof(detail::ImageFeatures));
    }
    return retval;
}

ClassArray::operator std::vector<detail::CameraParams>()
{
    CameraParamsPtr *temp =
            static_cast<CameraParamsPtr *>(this->data);

    std::vector<detail::CameraParams> retval(this->size);

    for(int i = 0; i < this->size; i++) {
        retval[i] = *static_cast<detail::CameraParams *>(temp[i].ptr);
        //memcpy(retval.data() + i, temp[i].ptr, sizeof(detail::CameraParams));
    }
    return retval;
}

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
struct RectWrapper detail_resultRoi2(
        struct PointArray corners,
        struct TensorArray images)
{
    std::vector<cv::UMat> uvec = get_vec_UMat(images.toMatList());
    return RectWrapper(detail::resultRoi(corners, uvec));
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
struct TensorWrapper detail_selectRandomSubset(int count, int size)
{
    std::vector<int> subset;
    detail::selectRandomSubset(count, size, subset);
    return TensorWrapper(cv::Mat(subset, true));
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
    return new detail::CameraParams();
}

extern "C"
struct CameraParamsPtr CameraParams_ctor2(
	struct CameraParamsPtr other)
{
    detail::CameraParams * instant = static_cast<detail::CameraParams *>(other.ptr);
    return new detail::CameraParams(*instant);
}

extern "C"
void CameraParams_dtor(
	struct CameraParamsPtr ptr)
{
    delete static_cast<detail::CameraParams *>(ptr.ptr);
}

struct TensorWrapper CameraParams_K(
	struct CameraParamsPtr ptr)
{
    return TensorWrapper(ptr->K());
}

//TODO need to add const CameraParams& detail::CameraParams::operator=(const CameraParams & other)

//DisjointSets

extern "C"
struct DisjointSetsPtr DisjointSets_ctor(
	int elem_count)
{
    return new detail::DisjointSets(elem_count);
}

extern "C"
void DisjointSets_dtor(
	struct DisjointSetsPtr ptr)
{
    delete static_cast<detail::DisjointSets *>(ptr.ptr);
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
    return new detail::Graph(num_vertices);
}

extern "C"
void Graph_dtor(
	struct GraphPtr ptr)
{
    delete static_cast<detail::Graph *>(ptr.ptr);
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
    return new detail::GraphEdge(from, to, weight);
}

extern "C"
void GraphEdge_dtor(
	struct GraphEdgePtr ptr)
{
    delete static_cast<detail::GraphEdge *>(ptr.ptr);
}

//Timelapser

extern "C"
struct TimelapserPtr Timelapser_ctor(
	int type)
{
    return rescueObjectFromPtr(
			detail::Timelapser::createDefault(type));
}

extern "C"
void Timelapser_dtor(
	struct TimelapserPtr ptr)
{
    delete static_cast<detail::Timelapser *>(ptr.ptr);
}

extern "C"
struct TensorWrapper Timelapser_getDst(
        struct TimelapserPtr ptr)
{
    cv::UMat umat = ptr->getDst();
    return TensorWrapper(MatT(umat.getMat(cv::ACCESS_RW)));
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
    return new detail::MatchesInfo();
}

extern "C"
struct MatchesInfoPtr MatchesInfo_ctor2(
        struct MatchesInfoPtr other)
{
    return new detail::MatchesInfo(*static_cast<detail::MatchesInfo *>(other.ptr));
}

extern "C"
void MatchesInfo_dtor(
        struct MatchesInfoPtr ptr)
{
    delete static_cast<detail::MatchesInfo *>(ptr.ptr);
}


//****************Features Finding and Images Matching************


//FeaturesFinder

extern "C"
void FeaturesFinder_dtor(
	struct FeaturesFinderPtr ptr)
{
    delete static_cast<detail::FeaturesFinder *>(ptr.ptr);
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
    detail::ImageFeatures *features = new detail::ImageFeatures();
    ptr->operator()(image.toMat(), *features);
    return ImageFeaturesPtr(features);
}

extern "C"
struct ImageFeaturesPtr FeaturesFinder_call2(
        struct FeaturesFinderPtr ptr, struct TensorWrapper image,
        struct RectArray rois)
{
    detail::ImageFeatures *features = new detail::ImageFeatures();
    ptr->operator()(image.toMat(), *features, rois);
    return ImageFeaturesPtr(features);
}

//OrbFeaturesFinder

extern "C"
struct OrbFeaturesFinderPtr OrbFeaturesFinder_ctor(
        struct SizeWrapper _grid_size, int nfeatures, float scaleFactor, int nlevels)
{
    return new detail::OrbFeaturesFinder(_grid_size, nfeatures, scaleFactor, nlevels);
}

//SurfFeaturesFinder

extern "C"
struct SurfFeaturesFinderPtr SurfFeaturesFinder_ctor(
        double hess_thresh, int num_octaves, int num_layers, int num_octaves_descr, int num_layers_descr)
{
    return new detail::SurfFeaturesFinder(
                                hess_thresh, num_octaves, num_layers,
                                num_octaves_descr, num_layers_descr);
}

//ImageFeatures

extern "C"
struct ImageFeaturesPtr ImageFeatures_ctor()
{
    return new detail::ImageFeatures();
}

extern "C"
void ImageFeatures_dtor(
        struct ImageFeaturesPtr ptr)
{
    delete static_cast<detail::ImageFeatures *>(ptr.ptr);
}

//FeaturesMatcher

extern "C"
void FeaturesMatcher_dtor(
        struct FeaturesMatcherPtr ptr)
{
    delete static_cast<detail::FeaturesMatcher *>(ptr.ptr);
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
    detail::MatchesInfo *Mat_inf = new detail::MatchesInfo();
    ptr->operator()(*static_cast<detail::ImageFeatures *>(features1.ptr),
                    *static_cast<detail::ImageFeatures *>(features2.ptr), *Mat_inf);
    return Mat_inf;
}

//BestOf2NearestMatcher

extern "C"
struct BestOf2NearestMatcherPtr BestOf2NearestMatcher_ctor(
        bool try_use_gpu, float match_conf,
        int num_matches_thresh1, int num_matches_thresh2)
{
    return new detail::BestOf2NearestMatcher(
                    try_use_gpu, match_conf, num_matches_thresh1, num_matches_thresh2);
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
    return new detail::BestOf2NearestRangeMatcher(range_width, try_use_gpu, match_conf,
                                                      num_matches_thresh1, num_matches_thresh2);
}

extern "C"
void BestOf2NearestRangeMatcher_call(
        struct BestOf2NearestRangeMatcherPtr ptr, struct ClassArray features,
        struct ClassArray pairwise_matches, struct TensorWrapper mask)
{
    std::vector<detail::ImageFeatures> features_vec = features;
    std::vector<detail::MatchesInfo> pairwise_matches_vec = pairwise_matches;

    if(mask.isNull()){
        ptr->operator()(features_vec, pairwise_matches_vec);
    }
    else {
        cv::UMat umat = mask.toMat().getUMat(cv::ACCESS_RW);
        ptr->operator()(features_vec, pairwise_matches_vec, umat);
    }
}

//**********************Rotation Estimation********************************

extern "C"
struct GraphPtrPlusTensor detail_findMaxSpanningTree(
        int num_images, struct ClassArray pairwise_matches)
{
    GraphPtrPlusTensor result;
    detail::Graph *span_tree = new detail::Graph();
    std::vector<int> centers;
    detail::findMaxSpanningTree(num_images, pairwise_matches, *span_tree, centers);
    result.graph.ptr = span_tree;
    new (&result.tensor) TensorWrapper(cv::Mat(centers, true));
    return result;
}

extern "C"
struct TensorWrapper detail_leaveBiggestComponent(
        struct ClassArray features, struct ClassArray pairwise_matches, float conf_threshold)
{
    std::vector<detail::ImageFeatures> features_vec = features;
    std::vector<detail::MatchesInfo> pairwise_matches_vec = pairwise_matches;

    return TensorWrapper(cv::Mat(
               detail::leaveBiggestComponent(
                       features_vec,
                       pairwise_matches_vec,
                       conf_threshold), true));
}

extern "C"
struct StringArray detail_matchesGraphAsString(
        struct StringArray pathes, struct ClassArray pairwise_matches, float conf_threshold)
{
    std::vector<cv::String> pathes_vec = pathes;
    std::vector<detail::MatchesInfo> pairwise_matches_vec = pairwise_matches;

    // TODO avoid copying (possibly by hacking cv::String)
    cv::String result = detail::matchesGraphAsString(
            pathes_vec, pairwise_matches_vec, conf_threshold);

    StringArray retval(1);
    retval.data[0] = static_cast<char *>(malloc(retval.size * sizeof(char)));
    strcpy(retval.data[0], result.c_str());
    return retval;
}

extern "C"
void detail_waveCorrect(
        struct TensorArray rmats, int kind)
{
    std::vector<cv::Mat> rmats_vec = rmats.toMatList();
    detail::waveCorrect(rmats_vec, static_cast<detail::WaveCorrectKind>(kind));
}

//Estimator

extern "C"
void Estimator_dtor(
        struct EstimatorPtr ptr)
{
    delete static_cast<detail::Estimator *>(ptr.ptr);
}

extern "C"
struct BoolPlusClassArray Estimator_call(
        struct EstimatorPtr ptr, struct ClassArray features, struct ClassArray 	pairwise_matches)
{
    struct BoolPlusClassArray result;
    std::vector<detail::CameraParams> cameras;
    std::vector<detail::ImageFeatures> features_vec = features;
    std::vector<detail::MatchesInfo> pairwise_matches_vec = pairwise_matches;

    result.val = ptr->operator()(features_vec, pairwise_matches_vec, cameras);
    new(&result.array) ClassArray(cameras);
    return result;
}

//HomographyBasedEstimator

extern "C"
struct HomographyBasedEstimatorPtr HomographyBasedEstimator_ctor(
        bool is_focals_estimated)
{
    return new detail::HomographyBasedEstimator(is_focals_estimated);
}

//BundleAdjusterBase

extern "C"
double BundleAdjusterBase_confThresh(
        struct BundleAdjusterBasePtr ptr)
{
    return ptr->confThresh();
}

extern "C"
struct TensorWrapper BundleAdjusterBase_refinementMask(
        struct BundleAdjusterBasePtr ptr)
{
    cv::Mat temp = ptr->refinementMask();
    return TensorWrapper(MatT(temp));
}

extern "C"
void BundleAdjusterBase_setConfThresh(
        struct BundleAdjusterBasePtr ptr, double conf_thresh)
{
    ptr->setConfThresh(conf_thresh);
}

extern "C"
void BundleAdjusterBase_setRefinementMask(
        struct BundleAdjusterBasePtr ptr, struct TensorWrapper mask)
{
    ptr->setRefinementMask(mask.toMat());
}

extern "C"
void BundleAdjusterBase_setTermCriteria(
        struct BundleAdjusterBasePtr ptr, struct TermCriteriaWrapper term_criteria)
{
    cv::TermCriteria term(term_criteria.type, term_criteria.maxCount, term_criteria.epsilon);
    ptr->setTermCriteria(term);
}

extern "C"
struct TermCriteriaWrapper BundleAdjusterBase_termCriteria(
        struct BundleAdjusterBasePtr ptr)
{
    struct TermCriteriaWrapper result;
    cv::TermCriteria term = ptr->termCriteria();
    result.type = term.type;
    result.epsilon = term.epsilon;
    result.maxCount = term.maxCount;
    return result;
}

//BundleAdjusterRay

extern "C"
struct BundleAdjusterRayPtr BundleAdjusterRay_ctor()
{
    return new detail::BundleAdjusterRay();
}

//BundleAdjusterReproj

extern "C"
struct BundleAdjusterReprojPtr BundleAdjusterReproj_ctor()
{
    return new detail::BundleAdjusterReproj();
}


//************************Autocalibration********************


extern "C"
struct TensorPlusBool detail_calibrateRotatingCamera(
        struct TensorArray Hs)
{
    struct TensorPlusBool result;
    cv::Mat K;
    result.val = detail::calibrateRotatingCamera(Hs.toMatList(), K);
    new(&result.tensor) TensorWrapper(MatT(K));
    return result;
}

extern "C"
struct TensorWrapper detail_estimateFocal(struct ClassArray features, struct ClassArray pairwise_matches)
{
    std::vector<double> focals;
    std::vector<detail::ImageFeatures> features_vec = features;
    std::vector<detail::MatchesInfo> pairwise_matches_vec = pairwise_matches;
    detail::estimateFocal(features_vec, pairwise_matches_vec, focals);
    return TensorWrapper(cv::Mat(focals, true));
}

extern "C"
struct focalsFromHomographyRetval detail_focalsFromHomography(
        struct TensorWrapper H)
{
    struct focalsFromHomographyRetval result;
    double f0, f1;
    bool f0_ok, f1_ok;

    detail::focalsFromHomography(H.toMat(), f0, f1, f0_ok, f1_ok);

    result.f0 = f0;
    result.f1 = f1;
    result.f0_ok = f0_ok;
    result.f1_ok = f1_ok;
    return result;
}


//***********************Images Warping***********************


//ProjectorBase

extern "C"
struct ProjectorBasePtr ProjectorBase_ctor()
{
    return new detail::ProjectorBase;
}

extern "C"
void ProjectorBase_dtor(
        struct ProjectorBasePtr ptr)
{
    delete static_cast<detail::ProjectorBase *>(ptr.ptr);
}

extern "C"
void ProjectorBase_setCameraParams(
        struct ProjectorBasePtr ptr, struct TensorWrapper K,
        struct TensorWrapper R, struct TensorWrapper T)
{
    ptr->setCameraParams(K.toMat(), R.toMat(), T.toMat());
}

////CompressedRectilinearPortraitProjector

extern "C"
struct CompressedRectilinearPortraitProjectorPtr CompressedRectilinearPortraitProjector_ctor()
{
    return new detail::CompressedRectilinearPortraitProjector;
}

extern "C"
void CompressedRectilinearPortraitProjector_dtor(
        struct CompressedRectilinearPortraitProjectorPtr ptr)
{
    delete static_cast<detail::CompressedRectilinearPortraitProjector *>(ptr.ptr);
}

extern "C"
struct FloatArray CompressedRectilinearPortraitProjector_mapBackward(
        struct CompressedRectilinearPortraitProjectorPtr ptr, float u, float v)
{
    std::vector<float> vec(2);
    ptr->mapBackward(u, v, vec[0], vec[0]);
    return FloatArray(vec);
}

extern "C"
struct FloatArray CompressedRectilinearPortraitProjector_mapForward(
        struct CompressedRectilinearPortraitProjectorPtr ptr, float x, float y)
{
    std::vector<float> vec(2);
    ptr->mapBackward(x, y, vec[0], vec[0]);
    return FloatArray(vec);
}

//CompressedRectilinearProjector

extern "C"
struct CompressedRectilinearProjectorPtr CompressedRectilinearProjector_ctor()
{
    return new detail::CompressedRectilinearProjector;
}

extern "C"
void CompressedRectilinearProjector_dtor(
        struct CompressedRectilinearProjectorPtr ptr)
{
    delete static_cast<detail::CompressedRectilinearProjector *>(ptr.ptr);
}

extern "C"
struct FloatArray CompressedRectilinearProjector_mapBackward(
        struct CompressedRectilinearProjectorPtr ptr, float u, float v)
{
    std::vector<float> vec(2);
    ptr->mapBackward(u, v, vec[0], vec[0]);
    return FloatArray(vec);
}

extern "C"
struct FloatArray CompressedRectilinearProjector_mapForward(
        struct CompressedRectilinearProjectorPtr ptr, float x, float y)
{
    std::vector<float> vec(2);
    ptr->mapBackward(x, y, vec[0], vec[0]);
    return FloatArray(vec);
}

//CylindricalPortraitProjector

extern "C"
struct CylindricalPortraitProjectorPtr CylindricalPortraitProjector_ctor()
{
    return new detail::CylindricalPortraitProjector;
}

extern "C"
void CylindricalPortraitProjector_dtor(
        struct CylindricalPortraitProjectorPtr ptr)
{
    delete static_cast<detail::CylindricalPortraitProjector *>(ptr.ptr);
}

extern "C"
struct FloatArray CylindricalPortraitProjector_mapBackward(
        struct CylindricalPortraitProjectorPtr ptr, float u, float v)
{
    std::vector<float> vec(2);
    ptr->mapBackward(u, v, vec[0], vec[0]);
    return FloatArray(vec);
}

extern "C"
struct FloatArray CylindricalPortraitProjector_mapForward(
        struct CylindricalPortraitProjectorPtr ptr, float x, float y)
{
    std::vector<float> vec(2);
    ptr->mapBackward(x, y, vec[0], vec[0]);
    return FloatArray(vec);
}

//CylindricalProjector

extern "C"
struct CylindricalProjectorPtr CylindricalProjector_ctor()
{
    return new detail::CylindricalProjector;
}

extern "C"
void CylindricalProjector_dtor(
        struct CylindricalProjectorPtr ptr)
{
    delete static_cast<detail::CylindricalProjector *>(ptr.ptr);
}

extern "C"
struct FloatArray CylindricalProjector_mapBackward(
        struct CylindricalProjectorPtr ptr, float u, float v)
{
    std::vector<float> vec(2);
    ptr->mapBackward(u, v, vec[0], vec[0]);
    return FloatArray(vec);
}

extern "C"
struct FloatArray CylindricalProjector_mapForward(
        struct CylindricalProjectorPtr ptr, float x, float y)
{
    std::vector<float> vec(2);
    ptr->mapBackward(x, y, vec[0], vec[0]);
    return FloatArray(vec);
}

//FisheyeProjector

extern "C"
void FisheyeProjector_dtor(
        struct FisheyeProjectorPtr ptr)
{
    delete static_cast<detail::FisheyeProjector *>(ptr.ptr);
}

extern "C"
struct FloatArray FisheyeProjector_mapBackward(
        struct FisheyeProjectorPtr ptr, float u, float v)
{
    std::vector<float> vec(2);
    ptr->mapBackward(u, v, vec[0], vec[0]);
    return FloatArray(vec);
}

extern "C"
struct FloatArray FisheyeProjector_mapForward(
        struct FisheyeProjectorPtr ptr, float x, float y)
{
    std::vector<float> vec(2);
    ptr->mapBackward(x, y, vec[0], vec[0]);
    return FloatArray(vec);
}

//MercatorProjector

extern "C"
void MercatorProjector_dtor(
        struct MercatorProjectorPtr ptr)
{
    delete static_cast<detail::MercatorProjector *>(ptr.ptr);
}

extern "C"
struct FloatArray MercatorProjector_mapBackward(
        struct MercatorProjectorPtr ptr, float u, float v)
{
    std::vector<float> vec(2);
    ptr->mapBackward(u, v, vec[0], vec[0]);
    return FloatArray(vec);
}

extern "C"
struct FloatArray MercatorProjector_mapForward(
        struct MercatorProjectorPtr ptr, float x, float y)
{
    std::vector<float> vec(2);
    ptr->mapBackward(x, y, vec[0], vec[0]);
    return FloatArray(vec);
}

//PaniniPortraitProjector

extern "C"
void PaniniPortraitProjector_dtor(
        struct PaniniPortraitProjectorPtr ptr)
{
    delete static_cast<detail::PaniniPortraitProjector *>(ptr.ptr);
}

extern "C"
struct FloatArray PaniniPortraitProjector_mapBackward(
        struct PaniniPortraitProjectorPtr ptr, float u, float v)
{
    std::vector<float> vec(2);
    ptr->mapBackward(u, v, vec[0], vec[0]);
    return FloatArray(vec);
}

extern "C"
struct FloatArray PaniniPortraitProjector_mapForward(
        struct PaniniPortraitProjectorPtr ptr, float x, float y)
{
    std::vector<float> vec(2);
    ptr->mapBackward(x, y, vec[0], vec[0]);
    return FloatArray(vec);
}

//PaniniProjector

extern "C"
void PaniniProjector_dtor(
        struct PaniniProjectorPtr ptr)
{
    delete static_cast<detail::PaniniProjector *>(ptr.ptr);
}

extern "C"
struct FloatArray PaniniProjector_mapBackward(
        struct PaniniProjectorPtr ptr, float u, float v)
{
    std::vector<float> vec(2);
    ptr->mapBackward(u, v, vec[0], vec[0]);
    return FloatArray(vec);
}

extern "C"
struct FloatArray PaniniProjector_mapForward(
        struct PaniniProjectorPtr ptr, float x, float y)
{
    std::vector<float> vec(2);
    ptr->mapBackward(x, y, vec[0], vec[0]);
    return FloatArray(vec);
}

//PlanePortraitProjector

extern "C"
void PlanePortraitProjector_dtor(
        struct PlanePortraitProjectorPtr ptr)
{
    delete static_cast<detail::PlanePortraitProjector *>(ptr.ptr);
}

extern "C"
struct FloatArray PlanePortraitProjector_mapBackward(
        struct PlanePortraitProjectorPtr ptr, float u, float v)
{
    std::vector<float> vec(2);
    ptr->mapBackward(u, v, vec[0], vec[0]);
    return FloatArray(vec);
}

extern "C"
struct FloatArray PlanePortraitProjector_mapForward(
        struct PlanePortraitProjectorPtr ptr, float x, float y)
{
    std::vector<float> vec(2);
    ptr->mapBackward(x, y, vec[0], vec[0]);
    return FloatArray(vec);
}

//PlaneProjector

extern "C"
void PlaneProjector_dtor(
        struct PlaneProjectorPtr ptr)
{
    delete static_cast<detail::PlaneProjector *>(ptr.ptr);
}

extern "C"
struct FloatArray PlaneProjector_mapBackward(
        struct PlaneProjectorPtr ptr, float u, float v)
{
    std::vector<float> vec(2);
    ptr->mapBackward(u, v, vec[0], vec[0]);
    return FloatArray(vec);
}

extern "C"
struct FloatArray PlaneProjector_mapForward(
        struct PlaneProjectorPtr ptr, float x, float y)
{
    std::vector<float> vec(2);
    ptr->mapBackward(x, y, vec[0], vec[0]);
    return FloatArray(vec);
}

//SphericalPortraitProjector

extern "C"
void SphericalPortraitProjector_dtor(
        struct SphericalPortraitProjectorPtr ptr)
{
    delete static_cast<detail::SphericalPortraitProjector *>(ptr.ptr);
}

extern "C"
struct FloatArray SphericalPortraitProjector_mapBackward(
        struct SphericalPortraitProjectorPtr ptr, float u, float v)
{
    std::vector<float> vec(2);
    ptr->mapBackward(u, v, vec[0], vec[0]);
    return FloatArray(vec);
}

extern "C"
struct FloatArray SphericalPortraitProjector_mapForward(
        struct SphericalPortraitProjectorPtr ptr, float x, float y)
{
    std::vector<float> vec(2);
    ptr->mapBackward(x, y, vec[0], vec[0]);
    return FloatArray(vec);
}

//SphericalProjector

extern "C"
void SphericalProjector_dtor(
        struct SphericalProjectorPtr ptr)
{
    delete static_cast<detail::SphericalProjector *>(ptr.ptr);
}

extern "C"
struct FloatArray SphericalProjector(
        struct SphericalProjectorPtr ptr, float u, float v)
{
    std::vector<float> vec(2);
    ptr->mapBackward(u, v, vec[0], vec[0]);
    return FloatArray(vec);
}

extern "C"
struct FloatArray SphericalProjector_mapForward(
        struct SphericalProjectorPtr ptr, float x, float y)
{
    std::vector<float> vec(2);
    ptr->mapBackward(x, y, vec[0], vec[0]);
    return FloatArray(vec);
}

//StereographicProjector

extern "C"
void StereographicProjector_dtor(
        struct StereographicProjectorPtr ptr)
{
    delete static_cast<detail::StereographicProjector *>(ptr.ptr);
}

extern "C"
struct FloatArray StereographicProjector_mapBackward(
        struct StereographicProjectorPtr ptr, float u, float v)
{
    std::vector<float> vec(2);
    ptr->mapBackward(u, v, vec[0], vec[0]);
    return FloatArray(vec);
}

extern "C"
struct FloatArray StereographicProjector_mapForward(
        struct StereographicProjectorPtr ptr, float x, float y)
{
    std::vector<float> vec(2);
    ptr->mapBackward(x, y, vec[0], vec[0]);
    return FloatArray(vec);
}

//TransverseMercatorProjector

extern "C"
void TransverseMercatorProjector_dtor(
        struct TransverseMercatorProjectorPtr ptr)
{
    delete static_cast<detail::TransverseMercatorProjector *>(ptr.ptr);
}

extern "C"
struct FloatArray TransverseMercatorProjector_mapBackward(
        struct TransverseMercatorProjectorPtr ptr, float u, float v)
{
    std::vector<float> vec(2);
    ptr->mapBackward(u, v, vec[0], vec[0]);
    return FloatArray(vec);
}

extern "C"
struct FloatArray TransverseMercatorProjector_mapForward(
        struct TransverseMercatorProjectorPtr ptr, float x, float y)
{
    std::vector<float> vec(2);
    ptr->mapBackward(x, y, vec[0], vec[0]);
    return FloatArray(vec);
}

//RotationWarper

extern "C"
void RotationWarper_dtor(
        struct RotationWarperPtr ptr)
{
    delete static_cast<detail::RotationWarper *>(ptr.ptr);
}

extern "C"
struct TensorArrayPlusRect RotationWarper_buildMaps(
        struct RotationWarperPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    struct TensorArrayPlusRect result;
    std::vector<MatT> map_mat(2);
    map_mat[0] = xmap.toMatT();
    map_mat[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMatT(), map_mat[0], map_mat[1]);
    new(&result.tensors) TensorArray(map_mat);

    return result;
}

extern "C"
float RotationWarper_getScale(
        struct RotationWarperPtr ptr)
{
    return ptr->getScale();
}

extern "C"
void RotationWarper_setScale(
        struct RotationWarperPtr ptr, float val)
{
    ptr->setScale(val);
}

extern "C"
struct TensorPlusPoint RotationWarper_warp(
        struct RotationWarperPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
        int border_mode, struct TensorWrapper dst)
{
    struct TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(),
                             interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

extern "C"
struct TensorWrapper RotationWarper_warpBackward(
        struct RotationWarperPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        int interp_mode, int border_mode, struct SizeWrapper dst_size,
        struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    ptr->warpBackward(src.toMat(), K.toMat(), R.toMat(), interp_mode,
                      border_mode, dst_size, dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct Point2fWrapper RotationWarper_warpPoint(
        struct RotationWarperPtr ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R)
{
    cv::Point2f pt2 = pt;
    ptr->warpPoint(pt2, K.toMat(), R.toMat());
    return Point2fWrapper(pt2);
}

extern "C"
struct RectWrapper RotationWarper_warpRoi(
        struct RotationWarperPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R)
{
    return ptr->warpRoi(src_size, K.toMat(), R.toMat());
}

//RotationWarperBase_CompressedRectilinearPortraitProjector

extern "C"
struct RotationWarperBase_CompressedRectilinearPortraitProjectorPtr
                        RotationWarperBase_CompressedRectilinearPortraitProjector_ctor()
{
    return new detail::RotationWarperBase<detail::CompressedRectilinearPortraitProjector>();
}

extern "C"
struct TensorArrayPlusRect RotationWarperBase_CompressedRectilinearPortraitProjector_buildMaps(
        struct RotationWarperBase_CompressedRectilinearPortraitProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    struct TensorArrayPlusRect result;
    std::vector<MatT> map_mat(2);
    map_mat[0] = xmap.toMatT();
    map_mat[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMatT(), map_mat[0], map_mat[1]);
    new(&result.tensors) TensorArray(map_mat);

    return result;
}

extern "C"
float RotationWarperBase_CompressedRectilinearPortraitProjector_getScale(
        struct RotationWarperBase_CompressedRectilinearPortraitProjectorPtr ptr)
{
    return ptr->getScale();
}

extern "C"
void RotationWarperBase_CompressedRectilinearPortraitProjector_setScale(
        struct RotationWarperBase_CompressedRectilinearPortraitProjectorPtr ptr, float val)
{
    ptr->setScale(val);
}

extern "C"
struct TensorPlusPoint RotationWarperBase_CompressedRectilinearPortraitProjector_warp(
        struct RotationWarperBase_CompressedRectilinearPortraitProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
        int border_mode, struct TensorWrapper dst)
{
    struct TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(),
                             interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

extern "C"
struct TensorWrapper RotationWarperBase_CompressedRectilinearPortraitProjector_warpBackward(
        struct RotationWarperBase_CompressedRectilinearPortraitProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        int interp_mode, int border_mode, struct SizeWrapper dst_size,
        struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    ptr->warpBackward(src.toMat(), K.toMat(), R.toMat(), interp_mode,
                      border_mode, dst_size, dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct Point2fWrapper RotationWarperBase_CompressedRectilinearPortraitProjector_warpPoint(
        struct RotationWarperBase_CompressedRectilinearPortraitProjectorPtr ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R)
{
    cv::Point2f pt2 = pt;
    ptr->warpPoint(pt2, K.toMat(), R.toMat());
    return Point2fWrapper(pt2);
}

extern "C"
struct RectWrapper RotationWarperBase_CompressedRectilinearPortraitProjector_warpRoi(
        struct RotationWarperBase_CompressedRectilinearPortraitProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R)
{
    return ptr->warpRoi(src_size, K.toMat(), R.toMat());
}

//RotationWarperBase_CompressedRectilinearProjector

extern "C"
struct RotationWarperBase_CompressedRectilinearProjectorPtr RotationWarperBase_CompressedRectilinearProjector_ctor()
{
    return new detail::RotationWarperBase<detail::CompressedRectilinearProjector>();
}

extern "C"
struct TensorArrayPlusRect RotationWarperBase_CompressedRectilinearProjector_buildMaps(
        struct RotationWarperBase_CompressedRectilinearProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    struct TensorArrayPlusRect result;
    std::vector<MatT> map_mat(2);
    map_mat[0] = xmap.toMatT();
    map_mat[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMatT(), map_mat[0], map_mat[1]);
    new(&result.tensors) TensorArray(map_mat);

    return result;
}

extern "C"
float RotationWarperBase_CompressedRectilinearProjector_getScale(
        struct RotationWarperBase_CompressedRectilinearProjectorPtr ptr)
{
    return ptr->getScale();
}

extern "C"
void RotationWarperBase_CompressedRectilinearProjector_setScale(
        struct RotationWarperBase_CompressedRectilinearProjectorPtr ptr, float val)
{
    ptr->setScale(val);
}

extern "C"
struct TensorPlusPoint RotationWarperBase_CompressedRectilinearProjector_warp(
        struct RotationWarperBase_CompressedRectilinearProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
        int border_mode, struct TensorWrapper dst)
{
    struct TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(),
                             interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

extern "C"
struct TensorWrapper RotationWarperBase_CompressedRectilinearProjector_warpBackward(
        struct RotationWarperBase_CompressedRectilinearProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        int interp_mode, int border_mode, struct SizeWrapper dst_size,
        struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    ptr->warpBackward(src.toMat(), K.toMat(), R.toMat(), interp_mode,
                      border_mode, dst_size, dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct Point2fWrapper RotationWarperBase_CompressedRectilinearProjector_warpPoint(
        struct RotationWarperBase_CompressedRectilinearProjectorPtr ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R)
{
    cv::Point2f pt2 = pt;
    ptr->warpPoint(pt2, K.toMat(), R.toMat());
    return Point2fWrapper(pt2);
}

extern "C"
struct RectWrapper RotationWarperBase_CompressedRectilinearProjector_warpRoi(
        struct RotationWarperBase_CompressedRectilinearProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R)
{
    return ptr->warpRoi(src_size, K.toMat(), R.toMat());
}


//RotationWarperBase_CylindricalPortraitProjector

extern "C"
struct RotationWarperBase_CylindricalPortraitProjectorPtr RotationWarperBase_CylindricalPortraitProjector_ctor()
{
    return new detail::RotationWarperBase<detail::CylindricalPortraitProjector>();
}

extern "C"
struct TensorArrayPlusRect RotationWarperBase_CylindricalPortraitProjector_buildMaps(
        struct RotationWarperBase_CylindricalPortraitProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    struct TensorArrayPlusRect result;
    std::vector<MatT> map_mat(2);
    map_mat[0] = xmap.toMatT();
    map_mat[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMatT(), map_mat[0], map_mat[1]);
    new(&result.tensors) TensorArray(map_mat);

    return result;
}

extern "C"
float RotationWarperBase_CylindricalPortraitProjector_getScale(
        struct RotationWarperBase_CylindricalPortraitProjectorPtr ptr)
{
    return ptr->getScale();
}

extern "C"
void RotationWarperBase_CylindricalPortraitProjector_setScale(
        struct RotationWarperBase_CylindricalPortraitProjectorPtr ptr, float val)
{
    ptr->setScale(val);
}

extern "C"
struct TensorPlusPoint RotationWarperBase_CylindricalPortraitProjector_warp(
        struct RotationWarperBase_CylindricalPortraitProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
        int border_mode, struct TensorWrapper dst)
{
    struct TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(),
                             interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

extern "C"
struct TensorWrapper RotationWarperBase_CylindricalPortraitProjector_warpBackward(
        struct RotationWarperBase_CylindricalPortraitProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        int interp_mode, int border_mode, struct SizeWrapper dst_size,
        struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    ptr->warpBackward(src.toMat(), K.toMat(), R.toMat(), interp_mode,
                      border_mode, dst_size, dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct Point2fWrapper RotationWarperBase_CylindricalPortraitProjector_warpPoint(
        struct RotationWarperBase_CylindricalPortraitProjectorPtr ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R)
{
    cv::Point2f pt2 = pt;
    ptr->warpPoint(pt2, K.toMat(), R.toMat());
    return Point2fWrapper(pt2);
}

extern "C"
struct RectWrapper RotationWarperBase_CylindricalPortraitProjector_warpRoi(
        struct RotationWarperBase_CylindricalPortraitProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R)
{
    return ptr->warpRoi(src_size, K.toMat(), R.toMat());
}

//RotationWarperBase_CylindricalProjector

extern "C"
struct RotationWarperBase_CylindricalProjectorPtr RotationWarperBase_CylindricalProjector_ctor()
{
    return new detail::RotationWarperBase<detail::CylindricalProjector>();
}

extern "C"
struct TensorArrayPlusRect RotationWarperBase_CylindricalProjector_buildMaps(
        struct RotationWarperBase_CylindricalProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    struct TensorArrayPlusRect result;
    std::vector<MatT> map_mat(2);
    map_mat[0] = xmap.toMatT();
    map_mat[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMatT(), map_mat[0], map_mat[1]);
    new(&result.tensors) TensorArray(map_mat);

    return result;
}

extern "C"
float RotationWarperBase_CylindricalProjector_getScale(
        struct RotationWarperBase_CylindricalProjectorPtr ptr)
{
    return ptr->getScale();
}

extern "C"
void RotationWarperBase_CylindricalProjector_setScale(
        struct RotationWarperBase_CylindricalProjectorPtr ptr, float val)
{
    ptr->setScale(val);
}

extern "C"
struct TensorPlusPoint RotationWarperBase_CylindricalProjector_warp(
        struct RotationWarperBase_CylindricalProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
        int border_mode, struct TensorWrapper dst)
{
    struct TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(),
                             interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

extern "C"
struct TensorWrapper RotationWarperBase_CylindricalProjector_warpBackward(
        struct RotationWarperBase_CylindricalProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        int interp_mode, int border_mode, struct SizeWrapper dst_size,
        struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    ptr->warpBackward(src.toMat(), K.toMat(), R.toMat(), interp_mode,
                      border_mode, dst_size, dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct Point2fWrapper RotationWarperBase_CylindricalProjector_warpPoint(
        struct RotationWarperBase_CylindricalProjectorPtr ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R)
{
    cv::Point2f pt2 = pt;
    ptr->warpPoint(pt2, K.toMat(), R.toMat());
    return Point2fWrapper(pt2);
}

extern "C"
struct RectWrapper RotationWarperBase_CylindricalProjector_warpRoi(
        struct RotationWarperBase_CylindricalProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R)
{
    return ptr->warpRoi(src_size, K.toMat(), R.toMat());
}

//RotationWarperBase_FisheyeProjector

extern "C"
struct RotationWarperBase_FisheyeProjectorPtr RotationWarperBase_FisheyeProjector_ctor()
{
    return new detail::RotationWarperBase<detail::FisheyeProjector>();
}

extern "C"
struct TensorArrayPlusRect RotationWarperBase_FisheyeProjector_buildMaps(
        struct RotationWarperBase_FisheyeProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    struct TensorArrayPlusRect result;
    std::vector<MatT> map_mat(2);
    map_mat[0] = xmap.toMatT();
    map_mat[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMatT(), map_mat[0], map_mat[1]);
    new(&result.tensors) TensorArray(map_mat);

    return result;
}

extern "C"
float RotationWarperBase_FisheyeProjector_getScale(
        struct RotationWarperBase_FisheyeProjectorPtr ptr)
{
    return ptr->getScale();
}

extern "C"
void RotationWarperBase_FisheyeProjector_setScale(
        struct RotationWarperBase_FisheyeProjectorPtr ptr, float val)
{
    ptr->setScale(val);
}

extern "C"
struct TensorPlusPoint RotationWarperBase_FisheyeProjector_warp(
        struct RotationWarperBase_FisheyeProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
        int border_mode, struct TensorWrapper dst)
{
    struct TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(),
                             interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

extern "C"
struct TensorWrapper RotationWarperBase_FisheyeProjector_warpBackward(
        struct RotationWarperBase_FisheyeProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        int interp_mode, int border_mode, struct SizeWrapper dst_size,
        struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    ptr->warpBackward(src.toMat(), K.toMat(), R.toMat(), interp_mode,
                      border_mode, dst_size, dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct Point2fWrapper RotationWarperBase_FisheyeProjector_warpPoint(
        struct RotationWarperBase_FisheyeProjectorPtr ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R)
{
    cv::Point2f pt2 = pt;
    ptr->warpPoint(pt2, K.toMat(), R.toMat());
    return Point2fWrapper(pt2);
}

extern "C"
struct RectWrapper RotationWarperBase_FisheyeProjector_warpRoi(
        struct RotationWarperBase_FisheyeProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R)
{
    return ptr->warpRoi(src_size, K.toMat(), R.toMat());
}

//RotationWarperBase_MercatorProjector

extern "C"
struct RotationWarperBase_MercatorProjectorPtr RotationWarperBase_MercatorProjector_ctor()
{
    return new detail::RotationWarperBase<detail::MercatorProjector>();
}

extern "C"
struct TensorArrayPlusRect RotationWarperBase_MercatorProjector_buildMaps(
        struct RotationWarperBase_MercatorProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    struct TensorArrayPlusRect result;
    std::vector<MatT> map_mat(2);
    map_mat[0] = xmap.toMatT();
    map_mat[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMatT(), map_mat[0], map_mat[1]);
    new(&result.tensors) TensorArray(map_mat);

    return result;
}

extern "C"
float RotationWarperBase_MercatorProjector_getScale(
        struct RotationWarperBase_MercatorProjectorPtr ptr)
{
    return ptr->getScale();
}

extern "C"
void RotationWarperBase_MercatorProjector_setScale(
        struct RotationWarperBase_MercatorProjectorPtr ptr, float val)
{
    ptr->setScale(val);
}

extern "C"
struct TensorPlusPoint RotationWarperBase_MercatorProjector_warp(
        struct RotationWarperBase_MercatorProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
        int border_mode, struct TensorWrapper dst)
{
    struct TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(),
                             interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

extern "C"
struct TensorWrapper RotationWarperBase_MercatorProjector_warpBackward(
        struct RotationWarperBase_MercatorProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        int interp_mode, int border_mode, struct SizeWrapper dst_size,
        struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    ptr->warpBackward(src.toMat(), K.toMat(), R.toMat(), interp_mode,
                      border_mode, dst_size, dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct Point2fWrapper RotationWarperBase_MercatorProjector_warpPoint(
        struct RotationWarperBase_MercatorProjectorPtr ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R)
{
    cv::Point2f pt2 = pt;
    ptr->warpPoint(pt2, K.toMat(), R.toMat());
    return Point2fWrapper(pt2);
}

extern "C"
struct RectWrapper RotationWarperBase_MercatorProjector_warpRoi(
        struct RotationWarperBase_MercatorProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R)
{
    return ptr->warpRoi(src_size, K.toMat(), R.toMat());
}

//RotationWarperBase_PaniniPortraitProjector

extern "C"
struct RotationWarperBase_PaniniPortraitProjectorPtr RotationWarperBase_PaniniPortraitProjector_ctor()
{
    return new detail::RotationWarperBase<detail::PaniniPortraitProjector>();
}

extern "C"
struct TensorArrayPlusRect RotationWarperBase_PaniniPortraitProjector_buildMaps(
        struct RotationWarperBase_PaniniPortraitProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    struct TensorArrayPlusRect result;
    std::vector<MatT> map_mat(2);
    map_mat[0] = xmap.toMatT();
    map_mat[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMatT(), map_mat[0], map_mat[1]);
    new(&result.tensors) TensorArray(map_mat);

    return result;
}

extern "C"
float RotationWarperBase_PaniniPortraitProjector_getScale(
        struct RotationWarperBase_PaniniPortraitProjectorPtr ptr)
{
    return ptr->getScale();
}

extern "C"
void RotationWarperBase_PaniniPortraitProjector_setScale(
        struct RotationWarperBase_PaniniPortraitProjectorPtr ptr, float val)
{
    ptr->setScale(val);
}

extern "C"
struct TensorPlusPoint RotationWarperBase_PaniniPortraitProjector_warp(
        struct RotationWarperBase_PaniniPortraitProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
        int border_mode, struct TensorWrapper dst)
{
    struct TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(),
                             interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

extern "C"
struct TensorWrapper RotationWarperBase_PaniniPortraitProjector_warpBackward(
        struct RotationWarperBase_PaniniPortraitProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        int interp_mode, int border_mode, struct SizeWrapper dst_size,
        struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    ptr->warpBackward(src.toMat(), K.toMat(), R.toMat(), interp_mode,
                      border_mode, dst_size, dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct Point2fWrapper RotationWarperBase_PaniniPortraitProjector_warpPoint(
        struct RotationWarperBase_PaniniPortraitProjectorPtr ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R)
{
    cv::Point2f pt2 = pt;
    ptr->warpPoint(pt2, K.toMat(), R.toMat());
    return Point2fWrapper(pt2);
}

extern "C"
struct RectWrapper RotationWarperBase_PaniniPortraitProjector_warpRoi(
        struct RotationWarperBase_PaniniPortraitProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R)
{
    return ptr->warpRoi(src_size, K.toMat(), R.toMat());
}

//RotationWarperBase_PaniniProjector

extern "C"
struct RotationWarperBase_PaniniProjectorPtr RotationWarperBase_PaniniProjector_ctor()
{
    return new detail::RotationWarperBase<detail::PaniniProjector>();
}

extern "C"
struct TensorArrayPlusRect RotationWarperBase_PaniniProjector_buildMaps(
        struct RotationWarperBase_PaniniProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    struct TensorArrayPlusRect result;
    std::vector<MatT> map_mat(2);
    map_mat[0] = xmap.toMatT();
    map_mat[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMatT(), map_mat[0], map_mat[1]);
    new(&result.tensors) TensorArray(map_mat);

    return result;
}

extern "C"
float RotationWarperBase_PaniniProjector_getScale(
        struct RotationWarperBase_PaniniProjectorPtr ptr)
{
    return ptr->getScale();
}

extern "C"
void RotationWarperBase_PaniniProjector_setScale(
        struct RotationWarperBase_PaniniProjectorPtr ptr, float val)
{
    ptr->setScale(val);
}

extern "C"
struct TensorPlusPoint RotationWarperBase_PaniniProjector_warp(
        struct RotationWarperBase_PaniniProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
        int border_mode, struct TensorWrapper dst)
{
    struct TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(),
                             interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

extern "C"
struct TensorWrapper RotationWarperBase_PaniniProjector_warpBackward(
        struct RotationWarperBase_PaniniProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        int interp_mode, int border_mode, struct SizeWrapper dst_size,
        struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    ptr->warpBackward(src.toMat(), K.toMat(), R.toMat(), interp_mode,
                      border_mode, dst_size, dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct Point2fWrapper RotationWarperBase_PaniniProjector_warpPoint(
        struct RotationWarperBase_PaniniProjectorPtr ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R)
{
    cv::Point2f pt2 = pt;
    ptr->warpPoint(pt2, K.toMat(), R.toMat());
    return Point2fWrapper(pt2);
}

extern "C"
struct RectWrapper RotationWarperBase_PaniniProjector_warpRoi(
        struct RotationWarperBase_PaniniProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R)
{
    return ptr->warpRoi(src_size, K.toMat(), R.toMat());
}

//RotationWarperBase_PlanePortraitProjector

extern "C"
struct RotationWarperBase_PlanePortraitProjectorPtr RotationWarperBase_PlanePortraitProjector_ctor()
{
    return new detail::RotationWarperBase<detail::PlanePortraitProjector>();
}

extern "C"
struct TensorArrayPlusRect RotationWarperBase_PlanePortraitProjector_buildMaps(
        struct RotationWarperBase_PlanePortraitProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    struct TensorArrayPlusRect result;
    std::vector<MatT> map_mat(2);
    map_mat[0] = xmap.toMatT();
    map_mat[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMatT(), map_mat[0], map_mat[1]);
    new(&result.tensors) TensorArray(map_mat);

    return result;
}

extern "C"
float RotationWarperBase_PlanePortraitProjector_getScale(
        struct RotationWarperBase_PlanePortraitProjectorPtr ptr)
{
    return ptr->getScale();
}

extern "C"
void RotationWarperBase_PlanePortraitProjector_setScale(
        struct RotationWarperBase_PlanePortraitProjectorPtr ptr, float val)
{
    ptr->setScale(val);
}

extern "C"
struct TensorPlusPoint RotationWarperBase_PlanePortraitProjector_warp(
        struct RotationWarperBase_PlanePortraitProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
        int border_mode, struct TensorWrapper dst)
{
    struct TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(),
                             interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

extern "C"
struct TensorWrapper RotationWarperBase_PlanePortraitProjector_warpBackward(
        struct RotationWarperBase_PlanePortraitProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        int interp_mode, int border_mode, struct SizeWrapper dst_size,
        struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    ptr->warpBackward(src.toMat(), K.toMat(), R.toMat(), interp_mode,
                      border_mode, dst_size, dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct Point2fWrapper RotationWarperBase_PlanePortraitProjector_warpPoint(
        struct RotationWarperBase_PlanePortraitProjectorPtr ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R)
{
    cv::Point2f pt2 = pt;
    ptr->warpPoint(pt2, K.toMat(), R.toMat());
    return Point2fWrapper(pt2);
}

extern "C"
struct RectWrapper RotationWarperBase_PlanePortraitProjector_warpRoi(
        struct RotationWarperBase_PlanePortraitProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R)
{
    return ptr->warpRoi(src_size, K.toMat(), R.toMat());
}

//RotationWarperBase_PlaneProjector

extern "C"
struct RotationWarperBase_PlaneProjectorPtr RotationWarperBase_PlaneProjector_ctor()
{
    return new detail::RotationWarperBase<detail::PlaneProjector>();
}

extern "C"
struct TensorArrayPlusRect RotationWarperBase_PlaneProjector_buildMaps(
        struct RotationWarperBase_PlaneProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    struct TensorArrayPlusRect result;
    std::vector<MatT> map_mat(2);
    map_mat[0] = xmap.toMatT();
    map_mat[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMatT(), map_mat[0], map_mat[1]);
    new(&result.tensors) TensorArray(map_mat);

    return result;
}

extern "C"
float RotationWarperBase_PlaneProjector_getScale(
        struct RotationWarperBase_PlaneProjectorPtr ptr)
{
    return ptr->getScale();
}

extern "C"
void RotationWarperBase_PlaneProjector_setScale(
        struct RotationWarperBase_PlaneProjectorPtr ptr, float val)
{
    ptr->setScale(val);
}

extern "C"
struct TensorPlusPoint RotationWarperBase_PlaneProjector_warp(
        struct RotationWarperBase_PlaneProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
        int border_mode, struct TensorWrapper dst)
{
    struct TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(),
                             interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

extern "C"
struct TensorWrapper RotationWarperBase_PlaneProjector_warpBackward(
        struct RotationWarperBase_PlaneProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        int interp_mode, int border_mode, struct SizeWrapper dst_size,
        struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    ptr->warpBackward(src.toMat(), K.toMat(), R.toMat(), interp_mode,
                      border_mode, dst_size, dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct Point2fWrapper RotationWarperBase_PlaneProjector_warpPoint(
        struct RotationWarperBase_PlaneProjectorPtr ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R)
{
    cv::Point2f pt2 = pt;
    ptr->warpPoint(pt2, K.toMat(), R.toMat());
    return Point2fWrapper(pt2);
}

extern "C"
struct RectWrapper RotationWarperBase_PlaneProjector_warpRoi(
        struct RotationWarperBase_PlaneProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R)
{
    return ptr->warpRoi(src_size, K.toMat(), R.toMat());
}

//RotationWarperBase_SphericalPortraitProjector

extern "C"
struct RotationWarperBase_SphericalPortraitProjectorPtr RotationWarperBase_SphericalPortraitProjector_ctor()
{
    return new detail::RotationWarperBase<detail::SphericalPortraitProjector>();
}

extern "C"
struct TensorArrayPlusRect RotationWarperBase_SphericalPortraitProjector_buildMaps(
        struct RotationWarperBase_SphericalPortraitProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    struct TensorArrayPlusRect result;
    std::vector<MatT> map_mat(2);
    map_mat[0] = xmap.toMatT();
    map_mat[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMatT(), map_mat[0], map_mat[1]);
    new(&result.tensors) TensorArray(map_mat);

    return result;
}

extern "C"
float RotationWarperBase_SphericalPortraitProjector_getScale(
        struct RotationWarperBase_SphericalPortraitProjectorPtr ptr)
{
    return ptr->getScale();
}

extern "C"
void RotationWarperBase_SphericalPortraitProjector_setScale(
        struct RotationWarperBase_SphericalPortraitProjectorPtr ptr, float val)
{
    ptr->setScale(val);
}

extern "C"
struct TensorPlusPoint RotationWarperBase_SphericalPortraitProjector_warp(
        struct RotationWarperBase_SphericalPortraitProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
        int border_mode, struct TensorWrapper dst)
{
    struct TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(),
                             interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

extern "C"
struct TensorWrapper RotationWarperBase_SphericalPortraitProjector_warpBackward(
        struct RotationWarperBase_SphericalPortraitProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        int interp_mode, int border_mode, struct SizeWrapper dst_size,
        struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    ptr->warpBackward(src.toMat(), K.toMat(), R.toMat(), interp_mode,
                      border_mode, dst_size, dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct Point2fWrapper RotationWarperBase_SphericalPortraitProjector_warpPoint(
        struct RotationWarperBase_SphericalPortraitProjectorPtr ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R)
{
    cv::Point2f pt2 = pt;
    ptr->warpPoint(pt2, K.toMat(), R.toMat());
    return Point2fWrapper(pt2);
}

extern "C"
struct RectWrapper RotationWarperBase_SphericalPortraitProjector_warpRoi(
        struct RotationWarperBase_SphericalPortraitProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R)
{
    return ptr->warpRoi(src_size, K.toMat(), R.toMat());
}

//RotationWarperBase_SphericalProjector

extern "C"
struct RotationWarperBase_SphericalProjectorPtr RotationWarperBase_SphericalProjector_ctor()
{
    return new detail::RotationWarperBase<detail::SphericalProjector>();
}

extern "C"
struct TensorArrayPlusRect RotationWarperBase_SphericalProjector_buildMaps(
        struct RotationWarperBase_SphericalProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    struct TensorArrayPlusRect result;
    std::vector<MatT> map_mat(2);
    map_mat[0] = xmap.toMatT();
    map_mat[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMatT(), map_mat[0], map_mat[1]);
    new(&result.tensors) TensorArray(map_mat);

    return result;
}

extern "C"
float RotationWarperBase_SphericalProjector_getScale(
        struct RotationWarperBase_SphericalProjectorPtr ptr)
{
    return ptr->getScale();
}

extern "C"
void RotationWarperBase_SphericalProjector_setScale(
        struct RotationWarperBase_SphericalProjectorPtr ptr, float val)
{
    ptr->setScale(val);
}

extern "C"
struct TensorPlusPoint RotationWarperBase_SphericalProjector_warp(
        struct RotationWarperBase_SphericalProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
        int border_mode, struct TensorWrapper dst)
{
    struct TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(),
                             interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

extern "C"
struct TensorWrapper RotationWarperBase_SphericalProjector_warpBackward(
        struct RotationWarperBase_SphericalProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        int interp_mode, int border_mode, struct SizeWrapper dst_size,
        struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    ptr->warpBackward(src.toMat(), K.toMat(), R.toMat(), interp_mode,
                      border_mode, dst_size, dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct Point2fWrapper RotationWarperBase_SphericalProjector_warpPoint(
        struct RotationWarperBase_SphericalProjectorPtr ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R)
{
    cv::Point2f pt2 = pt;
    ptr->warpPoint(pt2, K.toMat(), R.toMat());
    return Point2fWrapper(pt2);
}

extern "C"
struct RectWrapper RotationWarperBase_SphericalProjector_warpRoi(
        struct RotationWarperBase_SphericalProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R)
{
    return ptr->warpRoi(src_size, K.toMat(), R.toMat());
}

//RotationWarperBase_StereographicProjector

extern "C"
struct RotationWarperBase_StereographicProjectorPtr RotationWarperBase_StereographicProjector_ctor()
{
    return new detail::RotationWarperBase<detail::StereographicProjector>();
}

extern "C"
struct TensorArrayPlusRect RotationWarperBase_StereographicProjector_buildMaps(
        struct RotationWarperBase_StereographicProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    struct TensorArrayPlusRect result;
    std::vector<MatT> map_mat(2);
    map_mat[0] = xmap.toMatT();
    map_mat[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMatT(), map_mat[0], map_mat[1]);
    new(&result.tensors) TensorArray(map_mat);

    return result;
}

extern "C"
float RotationWarperBase_StereographicProjector_getScale(
        struct RotationWarperBase_StereographicProjectorPtr ptr)
{
    return ptr->getScale();
}

extern "C"
void RotationWarperBase_StereographicProjector_setScale(
        struct RotationWarperBase_StereographicProjectorPtr ptr, float val)
{
    ptr->setScale(val);
}

extern "C"
struct TensorPlusPoint RotationWarperBase_StereographicProjector_warp(
        struct RotationWarperBase_StereographicProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
        int border_mode, struct TensorWrapper dst)
{
    struct TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(),
                             interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

extern "C"
struct TensorWrapper RotationWarperBase_StereographicProjector_warpBackward(
        struct RotationWarperBase_StereographicProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        int interp_mode, int border_mode, struct SizeWrapper dst_size,
        struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    ptr->warpBackward(src.toMat(), K.toMat(), R.toMat(), interp_mode,
                      border_mode, dst_size, dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct Point2fWrapper RotationWarperBase_StereographicProjector_warpPoint(
        struct RotationWarperBase_StereographicProjectorPtr ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R)
{
    cv::Point2f pt2 = pt;
    ptr->warpPoint(pt2, K.toMat(), R.toMat());
    return Point2fWrapper(pt2);
}

extern "C"
struct RectWrapper RotationWarperBase_StereographicProjector_warpRoi(
        struct RotationWarperBase_StereographicProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R)
{
    return ptr->warpRoi(src_size, K.toMat(), R.toMat());
}

//RotationWarperBase_TransverseMercatorProjector

extern "C"
struct RotationWarperBase_TransverseMercatorProjectorPtr RotationWarperBase_TransverseMercatorProjector_ctor()
{
    return new detail::RotationWarperBase<detail::TransverseMercatorProjector>();
}

extern "C"
struct TensorArrayPlusRect RotationWarperBase_TransverseMercatorProjector_buildMaps(
        struct RotationWarperBase_TransverseMercatorProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    struct TensorArrayPlusRect result;
    std::vector<MatT> map_mat(2);
    map_mat[0] = xmap.toMatT();
    map_mat[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMatT(), map_mat[0], map_mat[1]);
    new(&result.tensors) TensorArray(map_mat);

    return result;
}

extern "C"
float RotationWarperBase_TransverseMercatorProjector_getScale(
        struct RotationWarperBase_TransverseMercatorProjectorPtr ptr)
{
    return ptr->getScale();
}

extern "C"
void RotationWarperBase_TransverseMercatorProjector_setScale(
        struct RotationWarperBase_TransverseMercatorProjectorPtr ptr, float val)
{
    ptr->setScale(val);
}

extern "C"
struct TensorPlusPoint RotationWarperBase_TransverseMercatorProjector_warp(
        struct RotationWarperBase_TransverseMercatorProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
        int border_mode, struct TensorWrapper dst)
{
    struct TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(),
                             interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

extern "C"
struct TensorWrapper RotationWarperBase_TransverseMercatorProjector_warpBackward(
        struct RotationWarperBase_TransverseMercatorProjectorPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        int interp_mode, int border_mode, struct SizeWrapper dst_size,
        struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    ptr->warpBackward(src.toMat(), K.toMat(), R.toMat(), interp_mode,
                      border_mode, dst_size, dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct Point2fWrapper RotationWarperBase_TransverseMercatorProjector_warpPoint(
        struct RotationWarperBase_TransverseMercatorProjectorPtr ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R)
{
    cv::Point2f pt2 = pt;
    ptr->warpPoint(pt2, K.toMat(), R.toMat());
    return Point2fWrapper(pt2);
}

extern "C"
struct RectWrapper RotationWarperBase_TransverseMercatorProjector_warpRoi(
        struct RotationWarperBase_TransverseMercatorProjectorPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R)
{
    return ptr->warpRoi(src_size, K.toMat(), R.toMat());
}

//WarperCreator

extern "C"
void WarperCreator_dtor(
        struct WarperCreatorPtr ptr)
{
    delete static_cast<cv::WarperCreator *>(ptr.ptr);
}

extern "C"
struct RotationWarperPtr WarperCreator_create(
        struct WarperCreatorPtr ptr, float scale)
{
    return rescueObjectFromPtr(ptr->create(scale));
}

//CompressedRectilinearPortraitWarper

extern "C"
struct CompressedRectilinearPortraitWarperPtr CompressedRectilinearPortraitWarper_ctor(
        float A, float B)
{
    return new cv::CompressedRectilinearPortraitWarper(A, B);
}

extern "C"
struct RotationWarperPtr CompressedRectilinearPortraitWarper_create(
        struct CompressedRectilinearPortraitWarperPtr ptr, float scale)
{
    return rescueObjectFromPtr(ptr->create(scale));
}

//CompressedRectilinearWarper

extern "C"
struct CompressedRectilinearWarperPtr CompressedRectilinearWarper_ctor(
        float A, float B)
{
    return new cv::CompressedRectilinearWarper(A, B);
}

extern "C"
struct RotationWarperPtr CompressedRectilinearWarper_create(
        struct CompressedRectilinearWarperPtr ptr, float scale)
{
    return rescueObjectFromPtr(ptr->create(scale));
}

//CylindricalWarper

extern "C"
struct CylindricalWarperPtr CylindricalWarper_ctor()
{
    return new cv::CylindricalWarper();
}

extern "C"
struct RotationWarperPtr CylindricalWarper_create(
        struct CylindricalWarperPtr ptr, float scale)
{
    return rescueObjectFromPtr(ptr->create(scale));
}

//FisheyeWarper

extern "C"
struct FisheyeWarperPtr FisheyeWarper_ctor()
{
    return new cv::FisheyeWarper();
}

extern "C"
struct RotationWarperPtr FisheyeWarper_create(
        struct FisheyeWarperPtr ptr, float scale)
{
    return rescueObjectFromPtr(ptr->create(scale));
}

//MercatorWarper

extern "C"
struct MercatorWarperPtr MercatorWarper_ctor()
{
    return new cv::MercatorWarper();
}

extern "C"
struct RotationWarperPtr MercatorWarper_create(
        struct MercatorWarperPtr ptr, float scale)
{
    return rescueObjectFromPtr(ptr->create(scale));
}

//PaniniPortraitWarper

extern "C"
struct PaniniPortraitWarperPtr PaniniPortraitWarper_ctor(
        float A, float B)
{
    return new cv::PaniniPortraitWarper(A, B);
}

extern "C"
struct RotationWarperPtr PaniniPortraitWarper_create(
        struct PaniniPortraitWarperPtr ptr, float scale)
{
    return rescueObjectFromPtr(ptr->create(scale));
}

//PaniniWarper

extern "C"
struct PaniniWarperPtr PaniniWarper_ctor(
        float A, float B)
{
    return new cv::PaniniWarper(A, B);
}

extern "C"
struct RotationWarperPtr PaniniWarper_create(
        struct PaniniWarperPtr ptr, float scale)
{
    return rescueObjectFromPtr(ptr->create(scale));
}

//PlaneWarper

extern "C"
struct PlaneWarperPtr PlaneWarper_ctor()
{
    return new cv::PlaneWarper();
}

extern "C"
struct RotationWarperPtr PlaneWarper_create(
        struct PlaneWarperPtr ptr, float scale)
{
    return rescueObjectFromPtr(ptr->create(scale));
}

//SphericalWarper

extern "C"
struct SphericalWarperPtr SphericalWarper_ctor()
{
    return new cv::SphericalWarper();
}

extern "C"
struct RotationWarperPtr SphericalWarper_create(
        struct SphericalWarperPtr ptr, float scale)
{
    return rescueObjectFromPtr(ptr->create(scale));
}

//StereographicWarper

extern "C"
struct StereographicWarperPtr StereographicWarper_ctor()
{
    return new cv::StereographicWarper();
}

extern "C"
struct RotationWarperPtr StereographicWarper_create(
        struct StereographicWarperPtr ptr, float scale)
{
    return rescueObjectFromPtr(ptr->create(scale));
}

//TransverseMercatorWarper

extern "C"
struct TransverseMercatorWarperPtr TransverseMercatorWarper_ctor()
{
    return new cv::TransverseMercatorWarper();
}

extern "C"
struct RotationWarperPtr TransverseMercatorWarper_create(
        struct TransverseMercatorWarperPtr ptr, float scale)
{
    return rescueObjectFromPtr(ptr->create(scale));
}

//detail_CompressedRectilinearPortraitWarper

extern "C"
struct detail_CompressedRectilinearPortraitWarperPtr detail_CompressedRectilinearPortraitWarper_ctor(
        float scale, float A, float B)
{
    return new detail::CompressedRectilinearPortraitWarper(scale, A, B);
}

//detail_CompressedRectilinearWarper

extern "C"
struct detail_CompressedRectilinearWarperPtr detail_CompressedRectilinearWarper_ctor(
        float scale, float A, float B)
{
    return new detail::CompressedRectilinearWarper(scale, A, B);
}

//detail_CylindricalPortraitWarper

extern "C"
struct detail_CylindricalPortraitWarperPtr detail_CylindricalPortraitWarper_ctor(
        float scale)
{
    return new detail::CylindricalPortraitWarper(scale);
}

//detail_CylindricalWarper

extern "C"
struct detail_CylindricalWarperPtr detail_CylindricalWarper_ctor(
        float scale)
{
    return new detail::CylindricalWarper(scale);
}

extern "C"
struct TensorArrayPlusRect detail_CylindricalWarper_buildMaps(
        struct detail_CylindricalWarperPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    TensorArrayPlusRect result;
    std::vector<MatT> vec(2);
    vec[0] = xmap.toMatT();
    vec[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMat(), vec[0], vec[1]);
    new(&result.tensors) TensorArray(vec);
    return result;
}

extern "C"
struct TensorPlusPoint detail_CylindricalWarper_warp(
        struct detail_CylindricalWarperPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        int interp_mode, int border_mode, struct TensorWrapper dst)
{
    TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(), interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

//detail_CylindricalWarperGpu

extern "C"
struct detail_CylindricalWarperGpuPtr detail_CylindricalWarperGpu_ctor(
        float scale)
{
    return new detail::CylindricalWarperGpu(scale);
}

extern "C"
struct TensorArrayPlusRect detail_CylindricalWarperGpu_buildMaps(
        struct detail_CylindricalWarperGpuPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    TensorArrayPlusRect result;
    std::vector<MatT> vec(2);
    vec[0] = xmap.toMatT();
    vec[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMat(), vec[0], vec[1]);
    new(&result.tensors) TensorArray(vec);
    return result;
}

extern "C"
struct TensorPlusPoint detail_CylindricalWarperGpu_warp(
        struct detail_CylindricalWarperGpuPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        int interp_mode, int border_mode, struct TensorWrapper dst)
{
    TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(
            src.toMat(), K.toMat(), R.toMat(),
            interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

//detail_FisheyeWarper

extern "C"
struct detail_FisheyeWarperPtr detail_FisheyeWarper_ctor(
        float scale)
{
    return new detail::FisheyeWarper(scale);
}

//detail_MercatorWarper

extern "C"
struct detail_MercatorWarperPtr detail_MercatorWarper_ctor(
        float scale)
{
    return new detail::MercatorWarper(scale);
}

//detail_PaniniPortraitWarper

extern "C"
struct detail_PaniniPortraitWarperPtr detail_PaniniPortraitWarper_ctor(
        float scale, float A, float B)
{
    return new detail::PaniniPortraitWarper(scale, A, B);
}

//detail_PaniniWarper

extern "C"
struct detail_PaniniWarperPtr detail_PaniniWarper_ctor(
        float scale, float A, float B)
{
    return new detail::PaniniWarper(scale, A, B);
}

//detail_PlanePortraitWarper

extern "C"
struct detail_PlanePortraitWarperPtr detail_PlanePortraitWarper_ctor(
        float scale)
{
    return new detail::PlanePortraitWarper(scale);
}

//detail_PlaneWarper

extern "C"
struct detail_PlaneWarperPtr detail_PlaneWarper_ctor(
        float scale)
{
    return new detail::PlaneWarper(scale);
}

extern "C"
struct TensorArrayPlusRect detail_PlaneWarper_buildMaps2(
        struct detail_PlaneWarperPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    TensorArrayPlusRect result;
    std::vector<MatT> vec(2);
    vec[0] = xmap.toMatT();
    vec[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMat(), vec[0], vec[1]);
    new(&result.tensors) TensorArray(vec);
    return result;
}

extern "C"
struct TensorArrayPlusRect detail_PlaneWarper_buildMaps(
        struct detail_PlaneWarperPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper T, struct TensorWrapper xmap,
        struct TensorWrapper ymap)
{
    struct TensorArrayPlusRect result;
    std::vector<MatT> map_mat(2);
    map_mat[0] = xmap.toMatT();
    map_mat[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(
            src_size, K.toMat(),R.toMatT(),
            T.toMat(), map_mat[0], map_mat[1]);
    new(&result.tensors) TensorArray(map_mat);

    return result;
}

extern "C"
struct TensorPlusPoint detail_PlaneWarper_warp(
        struct detail_PlaneWarperPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper T, int interp_mode,
        int border_mode, struct TensorWrapper dst)
{
    struct TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(), T.toMat(),
                             interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

extern "C"
struct TensorPlusPoint detail_PlaneWarper_warp2(
        struct detail_PlaneWarperPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,
        int interp_mode, int border_mode, struct TensorWrapper dst)
{
    TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(), interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

extern "C"
struct Point2fWrapper detail_PlaneWarper_warpPoint(
        struct detail_PlaneWarperPtr ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R, struct TensorWrapper T)
{
    cv::Point2f pt2 = pt;
    ptr->warpPoint(pt2, K.toMat(), R.toMat(), T.toMat());
    return Point2fWrapper(pt2);
}

extern "C"
struct Point2fWrapper detail_PlaneWarper_warpPoint2(
        struct detail_PlaneWarperPtr ptr, struct Point2fWrapper pt,
        struct TensorWrapper K, struct TensorWrapper R)
{
    cv::Point2f pt2 = pt;
    ptr->warpPoint(pt2, K.toMat(), R.toMat());
    return Point2fWrapper(pt2);
}

extern "C"
struct RectWrapper detail_PlaneWarper_warpRoi(
        struct detail_PlaneWarperPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R, struct TensorWrapper T)
{
    return ptr->warpRoi(src_size, K.toMat(), R.toMat(), T.toMat());
}

extern "C"
struct RectWrapper detail_PlaneWarper_warpRoi2(
        struct detail_PlaneWarperPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R)
{
    return ptr->warpRoi(src_size, K.toMat(), R.toMat());
}

//detail_SphericalPortraitWarper

extern "C"
struct detail_SphericalPortraitWarperPtr detail_SphericalPortraitWarper_ctor(
        float scale)
{
    return new detail::SphericalPortraitWarper(scale);
}

//detail_SphericalWarper

extern "C"
struct detail_SphericalWarperPtr detail_SphericalWarper_ctor(
        float scale)
{
    return new detail::SphericalWarper(scale);
}

extern "C"
struct TensorArrayPlusRect detail_SphericalWarper_buildMaps(
        struct detail_SphericalWarperPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    TensorArrayPlusRect result;
    std::vector<MatT> vec(2);
    vec[0] = xmap.toMatT();
    vec[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMat(), vec[0], vec[1]);
    new(&result.tensors) TensorArray(vec);
    return result;
}

extern "C"
struct TensorPlusPoint detail_SphericalWarper_warp(
        struct detail_SphericalWarperPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
        int border_mode, struct TensorWrapper dst)
{
    struct TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(),
                             interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

//detail_SphericalWarperGpu

extern "C"
struct detail_SphericalWarperGpuPtr detail_SphericalWarperGpu_ctor(
        float scale)
{
    return new detail::SphericalWarperGpu(scale);
}

extern "C"
struct TensorArrayPlusRect detail_SphericalWarperGpu_buildMaps(
        struct detail_SphericalWarperGpuPtr ptr, struct SizeWrapper src_size,
        struct TensorWrapper K, struct TensorWrapper R,
        struct TensorWrapper xmap, struct TensorWrapper ymap)
{
    TensorArrayPlusRect result;
    std::vector<MatT> vec(2);
    vec[0] = xmap.toMatT();
    vec[1] = ymap.toMatT();

    result.rect = ptr->buildMaps(src_size, K.toMat(), R.toMat(), vec[0], vec[1]);
    new(&result.tensors) TensorArray(vec);
    return result;
}

extern "C"
struct TensorPlusPoint detail_SphericalWarperGpu_warp(
        struct detail_SphericalWarperGpuPtr ptr, struct TensorWrapper src,
        struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
        int border_mode, struct TensorWrapper dst)
{
    struct TensorPlusPoint result;
    MatT dst_mat = dst.toMatT();
    result.point = ptr->warp(src.toMat(), K.toMat(), R.toMat(),
                             interp_mode, border_mode, dst_mat);
    new(&result.tensor) TensorWrapper(dst_mat);
    return result;
}

//detail_StereographicWarper

extern "C"
struct detail_StereographicWarperPtr detail_StereographicWarper_ctor(
        float scale)
{
    return new detail::StereographicWarper(scale);
}

//detail_TransverseMercatorWarper

extern "C"
struct detail_TransverseMercatorWarperPtr detail_TransverseMercatorWarper_ctor(
        float scale)
{
    return new detail::TransverseMercatorWarper(scale);
}


//************************Seam Estimation******************************


//SeamFinder

extern "C"
void SeamFinder_dtor(
        struct SeamFinderPtr ptr)
{
    delete static_cast<detail::SeamFinder *>(ptr.ptr);
}

extern "C"
void SeamFinder_find(
        struct SeamFinderPtr ptr, struct TensorArray src,
        struct PointArray corners, struct TensorArray masks)
{
    std::vector<cv::UMat> src_uvec = get_vec_UMat(src.toMatList());
    std::vector<cv::UMat> masks_uvec = get_vec_UMat(masks.toMatList());
    ptr->find(src_uvec, corners, masks_uvec);
}

//DpSeamFinder

extern "C"
struct DpSeamFinderPtr DpSeamFinder_ctor(int costFunc)
{
    enum  	detail::DpSeamFinder::CostFunction costFunc_enum;
    if(costFunc == 0) costFunc_enum = detail::DpSeamFinder::COLOR;
    else costFunc_enum = detail::DpSeamFinder::COLOR_GRAD;
    return new detail::DpSeamFinder(costFunc_enum);
}

extern "C"
int DpSeamFinder_costFunction(
        struct DpSeamFinderPtr ptr)
{
    detail::DpSeamFinder::CostFunction costFunc = ptr->costFunction();
    if(costFunc == detail::DpSeamFinder::COLOR) return 0;
    else return 1;
}

extern "C"
void DpSeamFinder_find(
        struct DpSeamFinderPtr ptr, struct TensorArray src, struct PointArray corners, struct TensorArray masks)
{
    std::vector<cv::UMat> src_uvec = get_vec_UMat(src.toMatList());
    std::vector<cv::UMat> masks_uvec = get_vec_UMat(masks.toMatList());
    ptr->find(src_uvec, corners, masks_uvec);
}

extern "C"
void DpSeamFinder_setCostFunction(
        struct DpSeamFinderPtr ptr, int val)
{
    enum detail::DpSeamFinder::CostFunction costFunc;
    if(val == 0) costFunc = detail::DpSeamFinder::COLOR;
    else costFunc = detail::DpSeamFinder::COLOR_GRAD;
    ptr->setCostFunction(costFunc);
}

//GraphCutSeamFinder

extern "C"
struct GraphCutSeamFinderPtr GraphCutSeamFinder_ctor(
        int cost_type, float terminal_cost, float bad_region_penalty)
{
    return new detail::GraphCutSeamFinder(cost_type, terminal_cost, bad_region_penalty);
}

extern "C"
void GraphCutSeamFinder_dtor(
        struct GraphCutSeamFinderPtr ptr)
{
    delete static_cast<detail::GraphCutSeamFinder *>(ptr.ptr);
}

extern "C"
void GraphCutSeamFinder_find(
        struct GraphCutSeamFinderPtr ptr, struct TensorArray src,
        struct PointArray corners, struct TensorArray masks)
{
    std::vector<cv::UMat> src_uvec = get_vec_UMat(src.toMatList());
    std::vector<cv::UMat> masks_uvec = get_vec_UMat(masks.toMatList());
    ptr->find(src_uvec, corners, masks_uvec);
}

//NoSeamFinder

extern "C"
struct NoSeamFinderPtr NoSeamFinder_ctor()
{
    return new detail::NoSeamFinder();
}

extern "C"
void NoSeamFinder_find(
        struct NoSeamFinderPtr ptr, struct TensorArray src,
        struct PointArray corners, struct TensorArray masks)
{
    std::vector<cv::UMat> src_uvec = get_vec_UMat(src.toMatList());
    std::vector<cv::UMat> masks_uvec = get_vec_UMat(masks.toMatList());
    ptr->find(src_uvec, corners, masks_uvec);
}

//PairwiseSeamFinder

extern "C"
void PairwiseSeamFinder_find(
        struct PairwiseSeamFinderPtr ptr, struct TensorArray src,
        struct PointArray corners, struct TensorArray masks)
{
    std::vector<cv::UMat> src_uvec = get_vec_UMat(src.toMatList());
    std::vector<cv::UMat> masks_uvec = get_vec_UMat(masks.toMatList());
    ptr->find(src_uvec, corners, masks_uvec);
}

//VoronoiSeamFinder

extern "C"
struct VoronoiSeamFinderPtr VoronoiSeamFinder_ctor()
{
    return new detail::VoronoiSeamFinder();
}

extern "C"
void VoronoiSeamFinder_find(
        struct VoronoiSeamFinderPtr ptr, struct TensorArray src,
        struct PointArray corners, struct TensorArray masks)
{
    std::vector<cv::UMat> src_uvec = get_vec_UMat(src.toMatList());
    std::vector<cv::UMat> masks_uvec = get_vec_UMat(masks.toMatList());
    ptr->find(src_uvec, corners, masks_uvec);
}

extern "C"
void VoronoiSeamFinder_find2(
        struct VoronoiSeamFinderPtr ptr, struct SizeArray size,
        struct PointArray corners, struct TensorArray masks)
{
    std::vector<cv::UMat> masks_uvec = get_vec_UMat(masks.toMatList());
    ptr->find(size, corners, masks_uvec);
}

//************************ExposureCompensator*******************************


//ExposureCompensator

extern "C"
struct ExposureCompensatorPtr ExposureCompensator_ctor(
        int type)
{
    return rescueObjectFromPtr(detail::ExposureCompensator::createDefault(type));
}

extern "C"
void ExposureCompensator_dtor(
        struct ExposureCompensatorPtr ptr)
{
    delete static_cast<detail::ExposureCompensator *>(ptr.ptr);
}

extern "C"
void  ExposureCompensator_apply(
        struct ExposureCompensatorPtr ptr, int index, struct PointWrapper corner,
        struct TensorWrapper image, struct TensorWrapper mask)
{
    ptr->apply(index, corner, image.toMat(), mask.toMat());
}

extern "C"
void ExposureCompensator_feed(
        struct ExposureCompensatorPtr ptr, struct PointArray corners,
        struct TensorArray images, struct TensorArray masks)
{
    std::vector<cv::UMat> images_uvec = get_vec_UMat(images.toMatList());
    std::vector<cv::UMat> masks_uvec = get_vec_UMat(masks.toMatList());
    ptr->feed(corners, images_uvec, masks_uvec);
}

//BlocksGainCompensator

extern "C"
struct BlocksGainCompensatorPtr BlocksGainCompensator_ctor(
        int bl_width, int bl_height)
{
    return new detail::BlocksGainCompensator(bl_width, bl_height);
}

extern "C"
void  BlocksGainCompensator_apply(
        struct BlocksGainCompensatorPtr ptr, int index, struct PointWrapper corner,
        struct TensorWrapper image, struct TensorWrapper mask)
{
    ptr->apply(index, corner, image.toMat(), mask.toMat());
}

extern "C"
void BlocksGainCompensator_feed(
        struct BlocksGainCompensatorPtr ptr, struct PointArray corners,
        struct TensorArray images, struct TensorArray mat, struct UCharArray chr)
{
    std::vector<cv::UMat> images_uvec = get_vec_UMat(images.toMatList());
    std::vector<cv::UMat> masks_uvec = get_vec_UMat(mat.toMatList());
    std::vector<std::pair<cv::UMat, uchar>> masks_vec(mat.size);
    for(int i = 0; i < mat.size; i++){
        masks_vec[i] = std::pair<cv::UMat, uchar>(masks_uvec[i], chr.data[i]);
    }
    ptr->feed(corners, images_uvec, masks_vec);
}

//GainCompensator

extern "C"
struct GainCompensatorPtr GainCompensator_ctor()
{
    return new detail::GainCompensator();
}

extern "C"
void  GainCompensator_apply(
        struct GainCompensatorPtr ptr, int index, struct PointWrapper corner,
        struct TensorWrapper image, struct TensorWrapper mask)
{
    ptr->apply(index, corner, image.toMat(), mask.toMat());
}

extern "C"
void GainCompensator_feed(
        struct GainCompensatorPtr ptr, struct PointArray corners,
        struct TensorArray images, struct TensorArray mat, struct UCharArray chr)
{
    std::vector<cv::UMat> images_uvec = get_vec_UMat(images.toMatList());
    std::vector<cv::UMat> masks_uvec = get_vec_UMat(mat.toMatList());
    std::vector<std::pair<cv::UMat, uchar>> masks_vec(mat.size);
    for(int i = 0; i < mat.size; i++){
        masks_vec[i] = std::pair<cv::UMat, uchar>(masks_uvec[i], chr.data[i]);
    }
    ptr->feed(corners, images_uvec, masks_vec);
}

extern "C"
struct TensorWrapper GainCompensator_gains(
        struct GainCompensatorPtr ptr)
{
    return TensorWrapper(cv::Mat(ptr->gains(), true));
}

//NoExposureCompensator

extern "C"
struct NoExposureCompensatorPtr NoExposureCompensator_ctor()
{
    return new detail::NoExposureCompensator();
}

extern "C"
void  NoExposureCompensator_apply(
        struct NoExposureCompensatorPtr ptr, int index, struct PointWrapper corner,
        struct TensorWrapper image, struct TensorWrapper mask)
{
    ptr->apply(index, corner, image.toMat(), mask.toMat());
}

extern "C"
void NoExposureCompensator_feed(
        struct NoExposureCompensatorPtr ptr, struct PointArray corners,
        struct TensorArray images, struct TensorArray mat, struct UCharArray chr)
{
    std::vector<cv::UMat> images_uvec = get_vec_UMat(images.toMatList());
    std::vector<cv::UMat> masks_uvec = get_vec_UMat(mat.toMatList());
    std::vector<std::pair<cv::UMat, uchar>> masks_vec(mat.size);
    for(int i = 0; i < mat.size; i++){
        masks_vec[i] = std::pair<cv::UMat, uchar>(masks_uvec[i], chr.data[i]);
    }
    ptr->feed(corners, images_uvec, masks_vec);
}


//*******************************Image Blenders**********************


extern "C"
struct TensorArray detail_createLaplacePyr(
        struct TensorWrapper img, int num_levels)
{
    std::vector<cv::UMat> pyr_uvec;
    detail::createLaplacePyr(img.toMat(), num_levels, pyr_uvec);
    std::vector<cv::Mat> pyr_vec = get_vec_Mat(pyr_uvec);
    return TensorArray(pyr_vec);
}

extern "C"
struct TensorArray detail_createLaplacePyrGpu(
        struct TensorWrapper img, int num_levels)
{
    std::vector<cv::UMat> pyr_uvec;
    detail::createLaplacePyr(img.toMat(), num_levels, pyr_uvec);
    std::vector<cv::Mat> pyr_vec = get_vec_Mat(pyr_uvec);
    return TensorArray(pyr_vec);
}

extern "C"
void detail_createWeightMap(
        struct TensorWrapper mask, float sharpness,
        struct TensorWrapper weight)
{
    detail::createWeightMap(mask.toMat(), sharpness, weight.toMat());
}

extern "C"
void detail_normalizeUsingWeightMap(
        struct TensorWrapper weight, struct TensorWrapper src)
{
    detail::normalizeUsingWeightMap(weight.toMat(), src.toMat());
}

extern "C"
void detail_restoreImageFromLaplacePyr(
        struct TensorArray pyr) {
    std::vector<cv::UMat> pyr_uvec = get_vec_UMat(pyr.toMatList());
    detail::restoreImageFromLaplacePyr(pyr_uvec);
}

extern "C"
void detail_restoreImageFromLaplacePyrGpu(
        struct TensorArray pyr) {
    std::vector<cv::UMat> pyr_uvec = get_vec_UMat(pyr.toMatList());
    detail::restoreImageFromLaplacePyrGpu(pyr_uvec);
}

//Blender

extern "C"
struct BlenderPtr Blender_ctor(
        int type, bool try_gpu)
{
    return rescueObjectFromPtr(detail::Blender::createDefault(type, try_gpu));
}

extern "C"
void Blender_dtor(
        struct BlenderPtr ptr)
{
    delete static_cast<detail::Blender *>(ptr.ptr);
}

extern "C"
void Blender_blend(
        struct BlenderPtr ptr, struct TensorWrapper dst,
        struct TensorWrapper dst_mask)
{
    ptr->blend(dst.toMat(), dst_mask.toMat());
}

extern "C"
void Blender_feed(
        struct BlenderPtr ptr, struct TensorWrapper img,
        struct TensorWrapper mask, struct PointWrapper tl)
{
    ptr->feed(img.toMat(), mask.toMat(), tl);
}

extern "C"
void Blender_prepare(
        struct BlenderPtr ptr, struct RectWrapper dst_roi)
{
    ptr->prepare(dst_roi);
}

extern "C"
void Blender_prepare2(
        struct BlenderPtr ptr, struct PointArray corners,
        struct SizeArray sizes)
{
    ptr->prepare(corners, sizes);
}

//FeatherBlender

extern "C"
struct FeatherBlenderPtr FeatherBlender_ctor(
        float sharpness)
{
    return new detail::FeatherBlender(sharpness);
}

extern "C"
void FeatherBlender_blend(
        struct FeatherBlenderPtr ptr, struct TensorWrapper dst,
        struct TensorWrapper dst_mask)
{
    ptr->blend(dst.toMat(), dst_mask.toMat());
}

extern "C"
struct RectWrapper FeatherBlender_createWeightMaps(
        struct FeatherBlenderPtr ptr, struct TensorArray masks,
        struct PointArray corners, struct TensorArray weight_maps)
{
    std::vector<cv::UMat> masks_uvec = get_vec_UMat(masks.toMatList());
    std::vector<cv::UMat> weight_maps_uvec = get_vec_UMat(weight_maps.toMatList());
    return RectWrapper(ptr->createWeightMaps(masks_uvec, corners, weight_maps_uvec));
}

extern "C"
void FeatherBlender_feed(
        struct FeatherBlenderPtr ptr, struct TensorWrapper img,
        struct TensorWrapper mask, struct PointWrapper tl)
{
    ptr->feed(img.toMat(), mask.toMat(), tl);
}

extern "C"
void FeatherBlender_prepare(
        struct FeatherBlenderPtr ptr, struct RectWrapper dst_roi)
{
    ptr->prepare(dst_roi);
}

extern "C"
void FeatherBlender_setSharpness(
        struct FeatherBlenderPtr ptr, float val)
{
    ptr->setSharpness(val);
}

extern "C"
float FeatherBlender_sharpness(
        struct FeatherBlenderPtr ptr)
{
    return ptr->sharpness();
}

//MultiBandBlender

extern "C"
struct MultiBandBlenderPtr MultiBandBlender_ctor(
        int try_gpu, int num_bands, int weight_type)
{
    return new detail::MultiBandBlender(try_gpu, num_bands, weight_type);
}

extern "C"
void MultiBandBlender_blend(
        struct MultiBandBlenderPtr ptr, struct TensorWrapper dst,
        struct TensorWrapper dst_mask)
{
    ptr->blend(dst.toMat(), dst_mask.toMat());
}

extern "C"
void MultiBandBlender_feed(
        struct MultiBandBlenderPtr ptr, struct TensorWrapper img,
        struct TensorWrapper mask, struct PointWrapper tl)
{
    ptr->feed(img.toMat(), mask.toMat(), tl);
}

extern "C"
int MultiBandBlender_numBands(
        struct MultiBandBlenderPtr ptr)
{
    return ptr->numBands();
}

extern "C"
void MultiBandBlender_prepare(
        struct MultiBandBlenderPtr ptr, struct RectWrapper dst_roi)
{
    ptr->prepare(dst_roi);
}

extern "C"
void MultiBandBlender_setNumBands(
        struct MultiBandBlenderPtr ptr, int val)
{
    ptr->setNumBands(val);
}

//Stitcher

extern "C"
struct StitcherPtr Stitcher_ctor(bool try_use_gpu)
{
    cv::Stitcher *ptr = new cv::Stitcher();
    *ptr = cv::Stitcher::createDefault(try_use_gpu);
    return ptr;
}

extern "C"
void Stitcher_dtor(
        struct StitcherPtr ptr)
{
    delete static_cast<cv::Stitcher *>(ptr.ptr);
}

extern "C"
struct BlenderPtr Stitcher_blender(
        struct StitcherPtr ptr)
{
    return rescueObjectFromPtr(ptr->blender());
}

extern "C"
struct BundleAdjusterBasePtr Stitcher_bundleAdjuster(
        struct StitcherPtr ptr)
{
    return rescueObjectFromPtr(ptr->bundleAdjuster());
}

extern "C"
struct ClassArray Stitcher_cameras(
        struct StitcherPtr ptr)
{
    return ClassArray(ptr->cameras());
}

extern "C"
struct TensorWrapper Stitcher_component(
        struct StitcherPtr ptr)
{
    return TensorWrapper(cv::Mat(ptr->component(), true));
}

extern "C"
struct TensorPlusInt Stitcher_composePanorama(
        struct StitcherPtr ptr)
{
    TensorPlusInt result;
    MatT pano;
    cv::Stitcher::Status status;
    status = ptr->composePanorama(pano);
    result.val = status;
    new(&result.tensor) TensorWrapper(pano);
    return result;
}

extern "C"
struct TensorPlusInt Stitcher_composePanorama2(
        struct StitcherPtr ptr, struct TensorArray images)
{
    TensorPlusInt result;
    MatT pano;
    cv::Stitcher::Status status;
    status = ptr->composePanorama(images.toMatList(), pano);
    result.val = status;
    new(&result.tensor) TensorWrapper(pano);
    return result;
}

extern "C"
double Stitcher_compositingResol(
        struct StitcherPtr ptr)
{
    return ptr->compositingResol();
}

extern "C"
int Stitcher_estimateTransform(
        struct StitcherPtr ptr, struct TensorArray images)
{
    return ptr->estimateTransform(images.toMatList());
}

extern "C"
struct ExposureCompensatorPtr Stitcher_exposureCompensator(
        struct StitcherPtr ptr)
{
    return rescueObjectFromPtr(ptr->exposureCompensator());
}

extern "C"
struct FeaturesFinderPtr Stitcher_featuresFinder(
        struct StitcherPtr ptr)
{
    return rescueObjectFromPtr(ptr->featuresFinder());
}

extern "C"
struct FeaturesMatcherPtr Stitcher_featuresMatcher(
        struct StitcherPtr ptr)
{
    return rescueObjectFromPtr(ptr->featuresMatcher());
}

extern "C"
struct TensorWrapper Stitcher_matchingMask(
        struct StitcherPtr ptr)
{
    return TensorWrapper(MatT(ptr->matchingMask().getMat(cv::ACCESS_RW)));
}

extern "C"
double Stitcher_panoConfidenceThresh(
        struct StitcherPtr ptr)
{
    return ptr->panoConfidenceThresh();
}

extern "C"
double Stitcher_registrationResol(
        struct StitcherPtr ptr)
{
    return ptr->registrationResol();
}

extern "C"
double Stitcher_seamEstimationResol(
        struct StitcherPtr ptr)
{
    return ptr->seamEstimationResol();
}

extern "C"
struct SeamFinderPtr Stitcher_seamFinder(
        struct StitcherPtr ptr)
{
    return rescueObjectFromPtr(ptr->seamFinder());
}

extern "C"
void Stitcher_setBlender(
        struct StitcherPtr ptr, struct BlenderPtr b)
{
    cv::Ptr<detail::Blender> p(static_cast<detail::Blender *>(b.ptr));
    rescueObjectFromPtr(p);
    ptr->setBlender(p);
}

extern "C"
void Stitcher_setBundleAdjuster(
        struct StitcherPtr ptr, struct BundleAdjusterBasePtr bundle_adjuster)
{
    cv::Ptr<detail::BundleAdjusterBase> p(static_cast<detail::BundleAdjusterBase *>(bundle_adjuster.ptr));
    rescueObjectFromPtr(p);
    ptr->setBundleAdjuster(p);
}

extern "C"
void Stitcher_setCompositingResol(
        struct StitcherPtr ptr, double resol_mpx)
{
    ptr->setCompositingResol(resol_mpx);
}

extern "C"
void Stitcher_setExposureCompensator(
        struct StitcherPtr ptr, struct ExposureCompensatorPtr exposure_comp)
{
    cv::Ptr<detail::ExposureCompensator>
            p(static_cast<detail::ExposureCompensator *>(exposure_comp.ptr));
    rescueObjectFromPtr(p);
    ptr->setExposureCompensator(p);
}

extern "C"
void Stitcher_setFeaturesFinder(
        struct StitcherPtr ptr, struct FeaturesFinderPtr features_finder)
{
    cv::Ptr<detail::FeaturesFinder>
            p(static_cast<detail::FeaturesFinder *>(features_finder.ptr));
    rescueObjectFromPtr(p);
    ptr->setFeaturesFinder(p);
}

extern "C"
void Stitcher_setFeaturesMatcher(
        struct StitcherPtr ptr, FeaturesMatcherPtr features_matcher)
{
    cv::Ptr<detail::FeaturesMatcher>
            p(static_cast<detail::FeaturesMatcher *>(features_matcher.ptr));
    rescueObjectFromPtr(p);
    ptr->setFeaturesMatcher(p);
}

extern "C"
void Stitcher_setMatchingMask(
        struct StitcherPtr ptr, struct TensorWrapper mask)
{
    cv::UMat umat =  mask.toMat().getUMat(cv::ACCESS_RW);
    ptr->setMatchingMask(umat);
}

extern "C"
void Stitcher_setPanoConfidenceThresh(
        struct StitcherPtr ptr, double conf_thresh)
{
    ptr->setPanoConfidenceThresh(conf_thresh);
}

extern "C"
void Stitcher_setRegistrationResol(
        struct StitcherPtr ptr, double resol_mpx)
{
    ptr->setRegistrationResol(resol_mpx);
}

extern "C"
void Stitcher_setSeamEstimationResol(
        struct StitcherPtr ptr, double resol_mpx)
{
    ptr->setSeamEstimationResol(resol_mpx);
}

extern "C"
void Stitcher_setSeamFinder(
        struct StitcherPtr ptr, struct SeamFinderPtr seam_finder)
{
    cv::Ptr<detail::SeamFinder>
            p(static_cast<detail::SeamFinder *>(seam_finder.ptr));
    rescueObjectFromPtr(p);
    ptr->setSeamFinder(p);
}

extern "C"
void Stitcher_setWarper(
        struct StitcherPtr ptr, struct WarperCreatorPtr creator)
{
    cv::Ptr<cv::WarperCreator> p(static_cast<cv::WarperCreator *>(creator.ptr));
    rescueObjectFromPtr(p);
    ptr->setWarper(p);
}

extern "C"
void Stitcher_setWaveCorrection(
        struct StitcherPtr ptr, bool flag)
{
    ptr->setWaveCorrection(flag);
}

extern "C"
void Stitcher_setWaveCorrectKind(
        struct StitcherPtr ptr, int kind)
{
    detail::WaveCorrectKind wave;
    if(kind == 0) wave = detail::WAVE_CORRECT_HORIZ;
    else wave = detail::WAVE_CORRECT_VERT;
    ptr->setWaveCorrectKind(wave);
}

extern "C"
struct TensorPlusInt Stitcher_stitch(
        struct StitcherPtr ptr, struct TensorArray images)
{
    TensorPlusInt result;
    MatT pano_mat;
    result.val = ptr->stitch(images.toMatList(), pano_mat);
    new(&result.tensor) TensorWrapper(pano_mat);
    return result;
}

extern "C"
struct WarperCreatorPtr Stitcher_warper(
        struct StitcherPtr ptr)
{
    return rescueObjectFromPtr(ptr->warper());
}

extern "C"
bool Stitcher_waveCorrection(
        struct StitcherPtr ptr)
{
    return ptr->waveCorrection();
}

extern "C"
int Stitcher_waveCorrectKind(
        struct StitcherPtr ptr)
{
    return ptr->waveCorrectKind();
}

extern "C"
double Stitcher_workScale(
        struct StitcherPtr ptr)
{
    return ptr->workScale();
}
