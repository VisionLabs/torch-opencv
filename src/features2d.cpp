#include <features2d.hpp>

// KeyPointsFilter

extern "C"
struct KeyPointsFilterPtr KeyPointsFilter_ctor()
{
    return new cv::KeyPointsFilter();
}

extern "C"
void KeyPointsFilter_dtor(struct KeyPointsFilterPtr ptr)
{
    delete static_cast<cv::KeyPointsFilter *>(ptr.ptr);
}

extern "C"
struct KeyPointArray KeyPointsFilter_runByImageBorder(struct KeyPointArray keypoints,
                                                      struct SizeWrapper imageSize, int borderSize)
{
    std::vector<cv::KeyPoint> keypointsVector(keypoints);
    cv::KeyPointsFilter::runByImageBorder(keypointsVector, imageSize, borderSize);
    return KeyPointArray(keypointsVector);
}

extern "C"
struct KeyPointArray KeyPointsFilter_runByKeypointSize(struct KeyPointArray keypoints,
                                                       float minSize, float maxSize)
{
    std::vector<cv::KeyPoint> keypointsVector(keypoints);
    cv::KeyPointsFilter::runByKeypointSize(keypointsVector, minSize, maxSize);
    return KeyPointArray(keypointsVector);
}

extern "C"
struct KeyPointArray KeyPointsFilter_runByPixelsMask(struct KeyPointArray keypoints,
                                                     struct TensorWrapper mask)
{
    std::vector<cv::KeyPoint> keypointsVector(keypoints);
    cv::KeyPointsFilter::runByPixelsMask(keypointsVector, mask.toMat());
    return KeyPointArray(keypointsVector);
}

extern "C"
struct KeyPointArray KeyPointsFilter_removeDuplicated(struct KeyPointArray keypoints)
{
    std::vector<cv::KeyPoint> keypointsVector(keypoints);
    cv::KeyPointsFilter::removeDuplicated(keypointsVector);
    return KeyPointArray(keypointsVector);
}

extern "C"
struct KeyPointArray KeyPointsFilter_retainBest(struct KeyPointArray keypoints, int npoints)
{
    std::vector<cv::KeyPoint> keypointsVector(keypoints);
    cv::KeyPointsFilter::retainBest(keypointsVector, npoints);
    return KeyPointArray(keypointsVector);
}

// Feature2D

extern "C"
struct KeyPointArray Feature2D_detect(
        struct Feature2DPtr ptr, struct TensorWrapper image, struct TensorWrapper mask)
{
    std::vector<cv::KeyPoint> keypointsVector;
    ptr->detect(image.toMat(), keypointsVector, TO_MAT_OR_NOARRAY(mask));
    return KeyPointArray(keypointsVector);
}

extern "C"
struct TensorPlusKeyPointArray Feature2D_compute(
	struct Feature2DPtr ptr, struct TensorWrapper image,
	struct KeyPointArray keypoints, struct TensorWrapper descriptors)
{
    TensorPlusKeyPointArray retval;

    std::vector<cv::KeyPoint> keypointsVector(keypoints);

    MatT descriptors_mat;
    if(descriptors.isNull()) descriptors_mat = descriptors.toMatT();
    ptr->compute(image.toMat(), keypointsVector, descriptors_mat);

    new (&retval.tensor) TensorWrapper(descriptors_mat);
    new (&retval.keypoints) KeyPointArray(keypointsVector);
    return retval;
}

extern "C"
struct TensorPlusKeyPointArray Feature2D_detectAndCompute(
        struct Feature2DPtr ptr, struct TensorWrapper image, struct TensorWrapper mask,
        struct TensorWrapper descriptors, bool useProvidedKeypoints)
{
    struct TensorPlusKeyPointArray retval;
    std::vector<cv::KeyPoint> keypointsVector;

    MatT descriptors_mat;
    if(descriptors.isNull()) descriptors_mat = descriptors.toMatT();
    ptr->detectAndCompute(
                image.toMat(), mask.toMat(), keypointsVector,
                descriptors_mat, useProvidedKeypoints);

    new (&retval.tensor) TensorWrapper(descriptors_mat);
    new (&retval.keypoints) KeyPointArray(keypointsVector);
    return retval;
}

extern "C"
int Feature2D_descriptorSize(struct Feature2DPtr ptr)
{
    return ptr->descriptorSize();
}

extern "C"
int Feature2D_descriptorType(struct Feature2DPtr ptr)
{
    return ptr->descriptorType();
}

extern "C"
int Feature2D_defaultNorm(struct Feature2DPtr ptr)
{
    return ptr->defaultNorm();
}

extern "C"
bool Feature2D_empty(struct Feature2DPtr ptr)
{
    return ptr->empty();
}

// BRISK

extern "C"
struct BRISKPtr BRISK_ctor(int thresh, int octaves, float patternScale)
{
    return rescueObjectFromPtr(cv::BRISK::create(thresh, octaves, patternScale));
}

extern "C"
struct BRISKPtr BRISK_ctor2(struct TensorWrapper radiusList, struct TensorWrapper numberList,
                            float dMax, float dMin, struct TensorWrapper indexChange)
{
    std::vector<int> indexVec;
    if (!indexChange.isNull())
        indexVec = indexChange.toMat();

    return rescueObjectFromPtr(cv::BRISK::create(
            radiusList.toMat(), numberList.toMat(), dMax, dMin, indexVec));
}

// ORB

extern "C"
struct ORBPtr ORB_ctor(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel,
                       int WTA_K, int scoreType, int patchSize, int fastThreshold)
{
    return rescueObjectFromPtr(cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel,
                                               WTA_K, scoreType, patchSize, fastThreshold));
}

extern "C"
void ORB_setMaxFeatures(struct ORBPtr ptr, int maxFeatures)
{
    ptr->setMaxFeatures(maxFeatures);
}

extern "C"
int ORB_getMaxFeatures(struct ORBPtr ptr)
{
    return ptr->getMaxFeatures();
}

extern "C"
void ORB_setScaleFactor(struct ORBPtr ptr, int scaleFactor)
{
    ptr->setScaleFactor(scaleFactor);
}

extern "C"
double ORB_getScaleFactor(struct ORBPtr ptr)
{
    return ptr->getScaleFactor();
}

extern "C"
void ORB_setNLevels(struct ORBPtr ptr, int nlevels)
{
    ptr->setNLevels(nlevels);
}

extern "C"
int ORB_getNLevels(struct ORBPtr ptr)
{
    return ptr->getNLevels();
}

extern "C"
void ORB_setEdgeThreshold(struct ORBPtr ptr, int edgeThreshold)
{
    ptr->setEdgeThreshold(edgeThreshold);
}

extern "C"
int ORB_getEdgeThreshold(struct ORBPtr ptr)
{
    return ptr->getEdgeThreshold();
}

extern "C"
void ORB_setFirstLevel(struct ORBPtr ptr, int firstLevel)
{
    ptr->setFirstLevel(firstLevel);
}

extern "C"
int ORB_getFirstLevel(struct ORBPtr ptr)
{
    return ptr->getFirstLevel();
}

extern "C"
void ORB_setWTA_K(struct ORBPtr ptr, int wta_k)
{
    ptr->setWTA_K(wta_k);
}

extern "C"
int ORB_getWTA_K(struct ORBPtr ptr)
{
    return ptr->getWTA_K();
}

extern "C"
void ORB_setScoreType(struct ORBPtr ptr, int scoreType)
{
    ptr->setScoreType(scoreType);
}

extern "C"
int ORB_getScoreType(struct ORBPtr ptr)
{
    return ptr->getScoreType();
}

extern "C"
void ORB_setPatchSize(struct ORBPtr ptr, int patchSize)
{
    ptr->setPatchSize(patchSize);
}

extern "C"
int ORB_getPatchSize(struct ORBPtr ptr)
{
    return ptr->getPatchSize();
}

extern "C"
void ORB_setFastThreshold(struct ORBPtr ptr, int fastThreshold)
{
    ptr->setFastThreshold(fastThreshold);
}

extern "C"
int ORB_getFastThreshold(struct ORBPtr ptr)
{
    return ptr->getFastThreshold();
}

// MSER

extern "C"
struct MSERPtr MSER_ctor(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity,
                         int _max_evolution, double _area_threshold, double _min_margin, int _edge_blur_size)
{
    return rescueObjectFromPtr(cv::MSER::create(_delta, _min_area, _max_area, _max_variation, _min_diversity, _max_evolution,
                                                _area_threshold, _min_margin, _edge_blur_size));
}

extern "C"
struct TensorArray MSER_detectRegions(struct MSERPtr ptr,
        struct TensorWrapper image, struct TensorWrapper bboxes)
{
    std::vector<std::vector<cv::Point>> result;
    cv::Mat bboxesMat = bboxes;
    std::vector<cv::Rect> bboxesVec(bboxesMat.rows);
    for (int row = 0; row < bboxesMat.rows; ++row) {
        int *rowPtr = reinterpret_cast<int*>(bboxesMat.ptr(row));
        bboxesVec[row].x = *rowPtr++;
        bboxesVec[row].y = *rowPtr++;
        bboxesVec[row].width = *rowPtr++;
        bboxesVec[row].height = *rowPtr++;
    }

    ptr->detectRegions(image.toMat(), result, bboxesVec);

    bboxesMat.create(bboxesVec.size(), 4, CV_32S);
    for (int row = 0; row < bboxesMat.rows; ++row) {
        int *rowPtr = reinterpret_cast<int*>(bboxesMat.ptr(row));
        *rowPtr++ = bboxesVec[row].x;
        *rowPtr++ = bboxesVec[row].y;
        *rowPtr++ = bboxesVec[row].width;
        *rowPtr++ = bboxesVec[row].height;
    }

    TensorArray retval;
    retval.size = result.size() + 1;
    retval.tensors = static_cast<TensorWrapper *>(
            malloc(retval.size * sizeof(TensorWrapper)));
    for (int i = 0; i < result.size(); ++i) {
        new (retval.tensors + i) TensorWrapper(MatT(cv::Mat(result[i])));
    }
    new (retval.tensors + retval.size - 1) TensorWrapper(bboxesMat);

    return retval;
}

extern "C"
void MSER_setDelta(struct MSERPtr ptr, int delta)
{
    ptr->setDelta(delta);
}

extern "C"
int MSER_getDelta(struct MSERPtr ptr)
{
    return ptr->getDelta();
}

extern "C"
void MSER_setMinArea(struct MSERPtr ptr, int minArea)
{
    ptr->setMinArea(minArea);
}

extern "C"
int MSER_getMinArea(struct MSERPtr ptr)
{
    return ptr->getMinArea();
}

extern "C"
void MSER_setMaxArea(struct MSERPtr ptr, int MaxArea)
{
    ptr->setMaxArea(MaxArea);
}

extern "C"
int MSER_getMaxArea(struct MSERPtr ptr)
{
    return ptr->getMaxArea();
}

extern "C"
void MSER_setPass2Only(struct MSERPtr ptr, bool Pass2Only)
{
    ptr->setPass2Only(Pass2Only);
}

extern "C"
bool MSER_getPass2Only(struct MSERPtr ptr)
{
    return ptr->getPass2Only();
}

extern "C"
struct KeyPointArray FAST(
        struct TensorWrapper image, int threshold, bool nonmaxSuppression)
{
    std::vector<cv::KeyPoint> retval;
    cv::FAST(image.toMat(), retval, threshold, nonmaxSuppression);
    return KeyPointArray(retval);
}

extern "C"
struct KeyPointArray FAST_type(
        struct TensorWrapper image, int threshold, bool nonmaxSuppression, int type)
{
    std::vector<cv::KeyPoint> retval;
    cv::FAST(image.toMat(), retval, threshold, nonmaxSuppression, type);
    return KeyPointArray(retval);
}

extern "C"
struct KeyPointArray AGAST(struct TensorWrapper image, int threshold, bool nonmaxSuppression)
{
    std::vector<cv::KeyPoint> retval;
    cv::AGAST(image.toMat(), retval, threshold, nonmaxSuppression);
    return KeyPointArray(retval);
}

extern "C"
struct KeyPointArray AGAST_type(
	struct TensorWrapper image, int threshold, bool nonmaxSuppression, int type)
{
    std::vector<cv::KeyPoint> retval;
    cv::AGAST(image.toMat(), retval, threshold, nonmaxSuppression, type);
    return KeyPointArray(retval);
}

// FastFeatureDetector

extern "C"
struct FastFeatureDetectorPtr FastFeatureDetector_ctor(
        int threshold, bool nonmaxSuppression, int type)
{
    return rescueObjectFromPtr(
            cv::FastFeatureDetector::create(threshold, nonmaxSuppression, type));
}

extern "C"
void FastFeatureDetector_setThreshold(struct FastFeatureDetectorPtr ptr, int val)
{
    ptr->setThreshold(val);
}

extern "C"
int FastFeatureDetector_getThreshold(struct FastFeatureDetectorPtr ptr)
{
    return ptr->getThreshold();
}

extern "C"
void FastFeatureDetector_setNonmaxSuppression(struct FastFeatureDetectorPtr ptr, bool val)
{
    ptr->setNonmaxSuppression(val);
}

extern "C"
bool FastFeatureDetector_getNonmaxSuppression(struct FastFeatureDetectorPtr ptr)
{
    return ptr->getNonmaxSuppression();
}

extern "C"
void FastFeatureDetector_setType(struct FastFeatureDetectorPtr ptr, int val)
{
    ptr->setType(val);
}

extern "C"
int FastFeatureDetector_getType(struct FastFeatureDetectorPtr ptr)
{
    return ptr->getType();
}

// AgastFeatureDetector

extern "C"
struct AgastFeatureDetectorPtr AgastFeatureDetector_ctor(
        int threshold, bool nonmaxSuppression, int type)
{
    return rescueObjectFromPtr(
            cv::AgastFeatureDetector::create(threshold, nonmaxSuppression, type));
}

extern "C"
void AgastFeatureDetector_setThreshold(struct AgastFeatureDetectorPtr ptr, int val)
{
    ptr->setThreshold(val);
}

extern "C"
int AgastFeatureDetector_getThreshold(struct AgastFeatureDetectorPtr ptr)
{
    return ptr->getThreshold();
}

extern "C"
void AgastFeatureDetector_setNonmaxSuppression(struct AgastFeatureDetectorPtr ptr, bool val)
{
    ptr->setNonmaxSuppression(val);
}

extern "C"
bool AgastFeatureDetector_getNonmaxSuppression(struct AgastFeatureDetectorPtr ptr)
{
    return ptr->getNonmaxSuppression();
}

extern "C"
void AgastFeatureDetector_setType(struct AgastFeatureDetectorPtr ptr, int val)
{
    ptr->setType(val);
}

extern "C"
int AgastFeatureDetector_getType(struct AgastFeatureDetectorPtr ptr)
{
    return ptr->getType();
}

// GFTTDetector

extern "C"
struct GFTTDetectorPtr GFTTDetector_ctor(
        int maxCorners, double qualityLevel, double minDistance,
        int blockSize, bool useHarrisDetector, double k)
{
    return rescueObjectFromPtr(cv::GFTTDetector::create(
            maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, k));
}

extern "C"
void GFTTDetector_setMaxFeatures(struct GFTTDetectorPtr ptr, int val)
{
    ptr->setMaxFeatures(val);
}

extern "C"
int GFTTDetector_getMaxFeatures(struct GFTTDetectorPtr ptr)
{
    return ptr->getMaxFeatures();
}

extern "C"
void GFTTDetector_setQualityLevel(struct GFTTDetectorPtr ptr, double val)
{
    ptr->setQualityLevel(val);
}

extern "C"
double GFTTDetector_getQualityLevel(struct GFTTDetectorPtr ptr)
{
    return ptr->getQualityLevel();
}

extern "C"
void GFTTDetector_setMinDistance(struct GFTTDetectorPtr ptr, double val)
{
    ptr->setMinDistance(val);
}

extern "C"
double GFTTDetector_getMinDistance(struct GFTTDetectorPtr ptr)
{
    return ptr->getMinDistance();
}

extern "C"
void GFTTDetector_setBlockSize(struct GFTTDetectorPtr ptr, int val)
{
    ptr->setBlockSize(val);
}

extern "C"
int GFTTDetector_getBlockSize(struct GFTTDetectorPtr ptr)
{
    return ptr->getBlockSize();
}

extern "C"
void GFTTDetector_setHarrisDetector(struct GFTTDetectorPtr ptr, bool val)
{
    ptr->setHarrisDetector(val);
}

extern "C"
bool GFTTDetector_getHarrisDetector(struct GFTTDetectorPtr ptr)
{
    return ptr->getHarrisDetector();
}

extern "C"
void GFTTDetector_setK(struct GFTTDetectorPtr ptr, double val)
{
    ptr->setK(val);
}

extern "C"
double GFTTDetector_getK(struct GFTTDetectorPtr ptr)
{
    return ptr->getK();
}

// SimpleBlobDetector

extern "C"
cv::SimpleBlobDetector::Params SimpleBlobDetector_Params_default()
{
    return cv::SimpleBlobDetector::Params();
}

extern "C"
struct SimpleBlobDetectorPtr SimpleBlobDetector_ctor(cv::SimpleBlobDetector::Params params)
{
    return rescueObjectFromPtr(cv::SimpleBlobDetector::create(params));
}

// KAZE

extern "C"
struct KAZEPtr KAZE_ctor(
        bool extended, bool upright, float threshold,
        int nOctaves, int nOctaveLayers, int diffusivity)
{
    return rescueObjectFromPtr(cv::KAZE::create(
            extended, upright, threshold, nOctaves, nOctaveLayers, diffusivity));
}

extern "C"
void KAZE_setExtended(struct KAZEPtr ptr, bool val)
{
    ptr->setExtended(val);
}

extern "C"
bool KAZE_getExtended(struct KAZEPtr ptr)
{
    return ptr->getExtended();
}

extern "C"
void KAZE_setUpright(struct KAZEPtr ptr, bool val)
{
    ptr->setUpright(val);
}

extern "C"
bool KAZE_getUpright(struct KAZEPtr ptr)
{
    return ptr->getUpright();
}

extern "C"
void KAZE_setThreshold(struct KAZEPtr ptr, double val)
{
    ptr->setThreshold(val);
}

extern "C"
double KAZE_getThreshold(struct KAZEPtr ptr)
{
    return ptr->getThreshold();
}

extern "C"
void KAZE_setNOctaves(struct KAZEPtr ptr, int val)
{
    ptr->setNOctaves(val);
}

extern "C"
int KAZE_getNOctaves(struct KAZEPtr ptr)
{
    return ptr->getNOctaves();
}

extern "C"
void KAZE_setNOctaveLayers(struct KAZEPtr ptr, int val)
{
    ptr->setNOctaveLayers(val);
}

extern "C"
int KAZE_getNOctaveLayers(struct KAZEPtr ptr)
{
    return ptr->getNOctaveLayers();
}

extern "C"
void KAZE_setDiffusivity(struct KAZEPtr ptr, int val)
{
    ptr->setDiffusivity(val);
}

extern "C"
int KAZE_getDiffusivity(struct KAZEPtr ptr)
{
    return ptr->getDiffusivity();
}

// AKAZE

extern "C"
struct AKAZEPtr AKAZE_ctor(
        int descriptor_type, int descriptor_size, int descriptor_channels,
        float threshold, int nOctaves, int nOctaveLayers, int diffusivity)
{
    return rescueObjectFromPtr(cv::AKAZE::create(
            descriptor_type, descriptor_size, descriptor_channels, threshold,
            nOctaves, nOctaveLayers, diffusivity));
}

extern "C"
void AKAZE_setDescriptorType(struct AKAZEPtr ptr, int val)
{
    ptr->setDescriptorType(val);
}

extern "C"
int AKAZE_getDescriptorType(struct AKAZEPtr ptr)
{
    return ptr->getDescriptorType();
}

extern "C"
void AKAZE_setDescriptorSize(struct AKAZEPtr ptr, int val)
{
    ptr->setDescriptorSize(val);
}

extern "C"
int AKAZE_getDescriptorSize(struct AKAZEPtr ptr)
{
    return ptr->getDescriptorSize();
}

extern "C"
void AKAZE_setDescriptorChannels(struct AKAZEPtr ptr, int val)
{
    ptr->setDescriptorChannels(val);
}

extern "C"
int AKAZE_getDescriptorChannels(struct AKAZEPtr ptr)
{
    return ptr->getDescriptorChannels();
}

extern "C"
void AKAZE_setThreshold(struct AKAZEPtr ptr, double val)
{
    ptr->setThreshold(val);
}

extern "C"
double AKAZE_getThreshold(struct AKAZEPtr ptr)
{
    return ptr->getThreshold();
}

extern "C"
void AKAZE_setNOctaves(struct AKAZEPtr ptr, int val)
{
    ptr->setNOctaves(val);
}

extern "C"
int AKAZE_getNOctaves(struct AKAZEPtr ptr)
{
    return ptr->getNOctaves();
}

extern "C"
void AKAZE_setNOctaveLayers(struct AKAZEPtr ptr, int val)
{
    ptr->setNOctaveLayers(val);
}

extern "C"
int AKAZE_getNOctaveLayers(struct AKAZEPtr ptr)
{
    return ptr->getNOctaveLayers();
}

extern "C"
void AKAZE_setDiffusivity(struct AKAZEPtr ptr, int val)
{
    ptr->setDiffusivity(val);
}

extern "C"
int AKAZE_getDiffusivity(struct AKAZEPtr ptr)
{
    return ptr->getDiffusivity();
}

// DescriptorMatcher

extern "C"
struct DescriptorMatcherPtr DescriptorMatcher_ctor(const char *descriptorMatcherType)
{
    return rescueObjectFromPtr(cv::DescriptorMatcher::create(descriptorMatcherType));
}

extern "C"
void DescriptorMatcher_add(struct DescriptorMatcherPtr ptr, struct TensorArray descriptors)
{
    ptr->add(descriptors.toMatList());
}

extern "C"
struct TensorArray DescriptorMatcher_getTrainDescriptors(struct DescriptorMatcherPtr ptr)
{
    std::vector<cv::Mat> vec = ptr->getTrainDescriptors();
    std::vector<MatT> retval(vec.size());
    for(int i = 0; i < vec.size(); i++) retval[i] = MatT(vec[i]);
    return TensorArray(retval);
}

extern "C"
void DescriptorMatcher_clear(struct DescriptorMatcherPtr ptr)
{
    ptr->clear();
}

extern "C"
bool DescriptorMatcher_empty(struct DescriptorMatcherPtr ptr)
{
    return ptr->empty();
}

extern "C"
bool DescriptorMatcher_isMaskSupported(struct DescriptorMatcherPtr ptr)
{
    return ptr->isMaskSupported();
}

extern "C"
void DescriptorMatcher_train(struct DescriptorMatcherPtr ptr)
{
    ptr->train();
}

extern "C"
struct DMatchArray DescriptorMatcher_match(struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper mask)
{
    std::vector<cv::DMatch> retval;
    ptr->match(
            queryDescriptors.toMat(), retval, TO_MAT_OR_NOARRAY(mask));
    return retval;
}

extern "C"
struct DMatchArray DescriptorMatcher_match_trainDescriptors(struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper mask)
{
    std::vector<cv::DMatch> retval;
    ptr->match(
            queryDescriptors.toMat(), trainDescriptors.toMat(), retval, TO_MAT_OR_NOARRAY(mask));
    return retval;
}

extern "C"
struct DMatchArrayOfArrays DescriptorMatcher_knnMatch(struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, int k,
        struct TensorWrapper mask, bool compactResult)
{
    std::vector<std::vector<cv::DMatch>> retval;
    ptr->knnMatch(
            queryDescriptors.toMat(), retval, k,
            TO_MAT_OR_NOARRAY(mask), compactResult);
    return DMatchArrayOfArrays(retval);
}

extern "C"
struct DMatchArrayOfArrays DescriptorMatcher_knnMatch_trainDescriptors(
        struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        int k, struct TensorWrapper mask, bool compactResult) {
    std::vector<std::vector<cv::DMatch>> retval;
    ptr->knnMatch(
            queryDescriptors.toMat(), trainDescriptors.toMat(),
            retval, k, TO_MAT_OR_NOARRAY(mask), compactResult);
    return DMatchArrayOfArrays(retval);
}

// BFMatcher

extern "C"
struct BFMatcherPtr BFMatcher_ctor(int normType, bool crossCheck)
{
    return new cv::BFMatcher(normType, crossCheck);
}

// FlannBasedMatcher

extern "C"
struct FlannBasedMatcherPtr FlannBasedMatcher_ctor(
        struct IndexParamsPtr indexParams, struct SearchParamsPtr searchParams)
{
    cv::Ptr<flann::IndexParams> indexParamsPtr(
            static_cast<flann::IndexParams *>(indexParams.ptr));
    cv::Ptr<flann::SearchParams> searchParamsPtr(
            static_cast<flann::SearchParams *>(searchParams.ptr));

    rescueObjectFromPtr(indexParamsPtr);
    rescueObjectFromPtr(searchParamsPtr);

    return new cv::FlannBasedMatcher(indexParamsPtr, searchParamsPtr);
}

extern "C"
struct TensorWrapper drawKeypoints(
        struct TensorWrapper image, struct KeyPointArray keypoints, struct TensorWrapper outImage,
        struct ScalarWrapper color, int flags)
{
    MatT outImage_mat;
    if(!outImage.isNull()) outImage_mat = outImage.toMatT();
    cv::drawKeypoints(image.toMat(), keypoints, outImage_mat, color, flags);
    return TensorWrapper(outImage_mat);
}

extern "C"
struct TensorWrapper drawMatches(
        struct TensorWrapper img1, struct KeyPointArray keypoints1,
        struct TensorWrapper img2, struct KeyPointArray keypoints2,
        struct DMatchArray matches1to2, struct TensorWrapper outImg,
        struct ScalarWrapper matchColor, struct ScalarWrapper singlePointColor,
        struct TensorWrapper matchesMask, int flags)
{
    // TODO: enhance this. Built-in Mat->std::vector conversion fails
    cv::Mat matchesMaskMat = matchesMask;
    std::vector<char> matchesMaskVec(matchesMaskMat.rows * matchesMaskMat.cols);
    if (!matchesMaskMat.empty()) {
        std::copy(
                matchesMaskMat.begin<char>(),
                matchesMaskMat.end<char>(),
                matchesMaskVec.begin());
    }

    MatT outImg_mat;
    if(!outImg.isNull()) outImg_mat = outImg.toMatT();
    cv::drawMatches(
            img1.toMat(), keypoints1, img2.toMat(), keypoints2, matches1to2, outImg_mat,
            matchColor, singlePointColor, matchesMaskVec, flags);
    return TensorWrapper(outImg_mat);
}

extern "C"
struct TensorWrapper drawMatchesKnn(
        struct TensorWrapper img1, struct KeyPointArray keypoints1,
        struct TensorWrapper img2, struct KeyPointArray keypoints2,
        struct DMatchArrayOfArrays matches1to2, struct TensorWrapper outImg,
        struct ScalarWrapper matchColor, struct ScalarWrapper singlePointColor,
        struct TensorArray matchesMask, int flags)
{
    std::vector<std::vector<char>> matchesMaskVec(matchesMask.size);
    for (int i = 0; i < matchesMask.size; ++i) {
        cv::Mat matchesMaskMat = matchesMask.tensors[i];
        matchesMaskVec[i].resize(matchesMaskMat.rows * matchesMaskMat.cols);

        if (!matchesMaskMat.empty()) {
            std::copy(
                    matchesMaskMat.begin<char>(),
                    matchesMaskMat.end<char>(),
                    matchesMaskVec[i].begin());
        }
    }

    MatT outImg_mat;
    if(!outImg.isNull()) outImg_mat = outImg.toMatT();
    cv::drawMatches(
                img1.toMat(), keypoints1, img2.toMat(), keypoints2, matches1to2, outImg_mat,
                matchColor, singlePointColor, matchesMaskVec, flags);
    return TensorWrapper(outImg_mat);
}

extern "C"
struct evaluateFeatureDetectorRetval evaluateFeatureDetector(
        struct TensorWrapper img1, struct TensorWrapper img2, struct TensorWrapper H1to2,
        struct Feature2DPtr fdetector)
{
    cv::Ptr<cv::FeatureDetector> fdetectorPtr(static_cast<cv::Feature2D *>(fdetector.ptr));
    rescueObjectFromPtr(fdetectorPtr);

    evaluateFeatureDetectorRetval retval;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::evaluateFeatureDetector(
            img1.toMat(), img2.toMat(), H1to2.toMat(), &keypoints1, &keypoints2,
            retval.repeatability, retval.correspCount, fdetectorPtr);

    new (&retval.keypoints1) KeyPointArray(keypoints1);
    new (&retval.keypoints2) KeyPointArray(keypoints2);

    return retval;
}

extern "C"
struct TensorWrapper computeRecallPrecisionCurve(
        struct DMatchArrayOfArrays matches1to2, struct TensorArray correctMatches1to2Mask)
{
    std::vector<std::vector<unsigned char>> correctMatches1to2MaskVec(correctMatches1to2Mask.size);
    for (int i = 0; i < correctMatches1to2Mask.size; ++i) {
        cv::Mat tempMat = correctMatches1to2Mask.tensors[i];
        correctMatches1to2MaskVec[i].resize(tempMat.rows * tempMat.cols);

        if (!tempMat.empty()) {
            std::copy(
                    tempMat.begin<char>(),
                    tempMat.end<char>(),
                    correctMatches1to2MaskVec[i].begin());
        }
    }

    std::vector<cv::Point2f> result;
    cv::computeRecallPrecisionCurve(matches1to2, correctMatches1to2MaskVec, result);
    return TensorWrapper(MatT(cv::Mat(result)));
}

extern "C"
float getRecall(struct TensorWrapper recallPrecisionCurve, float l_precision)
{
    std::vector<cv::Point2f> recallPrecisionCurveVec = recallPrecisionCurve.toMat();
    return cv::getRecall(recallPrecisionCurveVec, l_precision);
}

extern "C"
int getNearestPoint(struct TensorWrapper recallPrecisionCurve, float l_precision)
{
    std::vector<cv::Point2f> recallPrecisionCurveVec = recallPrecisionCurve.toMat();
    return cv::getNearestPoint(recallPrecisionCurveVec, l_precision);
}

// BOWTrainer

extern "C"
void BOWTrainer_dtor(struct BOWTrainerPtr ptr)
{
    delete static_cast<cv::BOWTrainer *>(ptr.ptr);
}

extern "C"
void BOWTrainer_add(struct BOWTrainerPtr ptr, struct TensorWrapper descriptors)
{
    ptr->add(descriptors.toMat());
}

extern "C"
struct TensorArray BOWTrainer_getDescriptors(struct BOWTrainerPtr ptr)
{
    std::vector<cv::Mat> vec = ptr->getDescriptors();
    std::vector<MatT> retval = get_vec_MatT(vec);
    return TensorArray(retval);
}

extern "C"
int BOWTrainer_descriptorsCount(struct BOWTrainerPtr ptr)
{
    return ptr->descriptorsCount();
}

extern "C"
void BOWTrainer_clear(struct BOWTrainerPtr ptr)
{
    ptr->clear();
}

extern "C"
struct TensorWrapper BOWTrainer_cluster(struct BOWTrainerPtr ptr)
{
    return TensorWrapper(MatT(ptr->cluster()));
}

extern "C"
struct TensorWrapper BOWTrainer_cluster_descriptors(struct BOWTrainerPtr ptr, struct TensorWrapper descriptors)
{
    return TensorWrapper(MatT(ptr->cluster(descriptors.toMat())));
}

// BOWKMeansTrainer

extern "C"
struct BOWKMeansTrainerPtr BOWKMeansTrainer_ctor(
        int clusterCount, struct TermCriteriaWrapper termcrit,
        int attempts, int flags)
{
    return new cv::BOWKMeansTrainer(
            clusterCount, termcrit.orDefault(cv::TermCriteria()), attempts, flags);
}

// BOWImgDescriptorExtractor

extern "C"
struct BOWImgDescriptorExtractorPtr BOWImgDescriptorExtractor_ctor(
        struct Feature2DPtr dextractor, struct DescriptorMatcherPtr dmatcher)
{
    cv::Ptr<cv::Feature2D> dextractorPtr(static_cast<cv::Feature2D *>(dextractor.ptr));
    cv::Ptr<cv::DescriptorMatcher> dmatcherPtr(static_cast<cv::DescriptorMatcher *>(dmatcher.ptr));

    return new cv::BOWImgDescriptorExtractor(
            rescueObjectFromPtr(dextractorPtr),
            rescueObjectFromPtr(dmatcherPtr));
}

extern "C"
void BOWImgDescriptorExtractor_dtor(struct BOWImgDescriptorExtractorPtr ptr)
{
    delete static_cast<cv::BOWImgDescriptorExtractor *>(ptr.ptr);
}

extern "C"
void BOWImgDescriptorExtractor_setVocabulary(
        struct BOWImgDescriptorExtractorPtr ptr, struct TensorWrapper vocabulary)
{
    ptr->setVocabulary(vocabulary.toMat());
}

extern "C"
struct TensorWrapper BOWImgDescriptorExtractor_getVocabulary(struct BOWImgDescriptorExtractorPtr ptr)
{
    return TensorWrapper(MatT(ptr->getVocabulary().clone()));
}

extern "C"
struct TensorWrapper BOWImgDescriptorExtractor_compute(
        struct BOWImgDescriptorExtractorPtr ptr, struct TensorWrapper image,
        struct KeyPointArray keypoints, struct TensorWrapper imgDescriptor)
{
    std::vector<cv::KeyPoint> keypointsVec = keypoints;

    cv::Mat imgDescriptor_mat;
    if(!imgDescriptor.isNull()) imgDescriptor_mat = imgDescriptor.toMat();
    ptr->compute2(image.toMat(), keypointsVec, imgDescriptor_mat);
    return TensorWrapper(MatT(imgDescriptor_mat));
}

extern "C"
int BOWImgDescriptorExtractor_descriptorSize(struct BOWImgDescriptorExtractorPtr ptr)
{
    return ptr->descriptorSize();
}

extern "C"
int BOWImgDescriptorExtractor_descriptorType(struct BOWImgDescriptorExtractorPtr ptr)
{
    return ptr->descriptorType();
}
