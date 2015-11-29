#include <features2d.hpp>

KeyPointWrapper::KeyPointWrapper(const cv::KeyPoint & other) {
    this->pt = other.pt;
    this->size = other.size;
    this->angle = other.angle;
    this->response = other.response;
    this->octave = other.octave;
    this->class_id = other.class_id;
}

KeyPointArray::KeyPointArray(const std::vector<cv::KeyPoint> & v)
{
    this->size = v.size();
    this->data = static_cast<KeyPointWrapper *>(
            malloc(sizeof(KeyPointWrapper) * this->size));
    for (int i = 0; i < this->size; ++i) {
        this->data[i] = v[i];
    }
}

KeyPointArray::operator std::vector<cv::KeyPoint>()
{
    std::vector<cv::KeyPoint> retval(this->size);
    for (int i = 0; i < this->size; ++i) {
        retval[i] = this->data[i];
    }
    return retval;
}

KeyPointMat::KeyPointMat(const std::vector<std::vector<cv::KeyPoint> > & v)
{
    // TODO: this function
    this->size1 = v.size();
    this->size2 = v[0].size();
}

KeyPointMat::operator std::vector<std::vector<cv::KeyPoint> >()
{
    // TODO: this function
    std::vector<std::vector<cv::KeyPoint> > retval;
    return retval;
}

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
struct TensorPlusKeyPointArray Feature2D_compute(struct Feature2DPtr ptr, struct TensorWrapper image,
                                                 struct KeyPointArray keypoints, struct TensorWrapper descriptors)
{
    TensorPlusKeyPointArray retval;

    std::vector<cv::KeyPoint> keypointsVector(keypoints);
    if (descriptors.isNull()) {
        cv::Mat result;
        ptr->compute(image.toMat(), keypointsVector, result);
        new (&retval.tensor) TensorWrapper(result);
    } else {
        ptr->compute(image.toMat(), keypointsVector, descriptors.toMat());
        retval.tensor = descriptors;
    }
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
    if (descriptors.isNull()) {
        cv::Mat result;
        ptr->detectAndCompute(
                image.toMat(), mask.toMat(), keypointsVector,
                result, useProvidedKeypoints);
        new (&retval.tensor) TensorWrapper(result);
    } else {
        ptr->detectAndCompute(
                image.toMat(), mask.toMat(), keypointsVector,
                descriptors.toMat(), useProvidedKeypoints);
        retval.tensor = descriptors;
    }
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
        new (retval.tensors + i) TensorWrapper(cv::Mat(result[i]));
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
struct KeyPointArray AGAST_type(struct TensorWrapper image, int threshold, bool nonmaxSuppression, int type)
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
    std::vector<cv::Mat> result = ptr->getDescriptors();
    return TensorArray(result);
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
    return TensorWrapper(ptr->cluster());
}

extern "C"
struct TensorWrapper BOWTrainer_cluster_descriptors(struct BOWTrainerPtr ptr, struct TensorWrapper descriptors)
{
    return TensorWrapper(ptr->cluster(descriptors.toMat()));
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
struct TensorWrapper getVocabulary(struct BOWImgDescriptorExtractorPtr ptr)
{
    return TensorWrapper(ptr->getVocabulary().clone());
}

extern "C"
struct TensorWrapper compute(
        struct BOWImgDescriptorExtractorPtr ptr, struct TensorWrapper image,
        struct KeyPointArray keypoints, struct TensorWrapper imgDescriptor)
{
    std::vector<cv::KeyPoint> keypointsVec = keypoints;

    if (imgDescriptor.isNull()) {
        cv::Mat retval;
        ptr->compute2(image.toMat(), keypointsVec, retval);
        return TensorWrapper(retval);
    } else {
        cv::Mat imgDescriptorMat = imgDescriptor;
        ptr->compute2(image.toMat(), keypointsVec, imgDescriptorMat);
        return imgDescriptor;
    }
}

extern "C"
int descriptorSize(struct BOWImgDescriptorExtractorPtr ptr)
{
    return ptr->descriptorSize();
}

extern "C"
int descriptorType(struct BOWImgDescriptorExtractorPtr ptr)
{
    return ptr->descriptorType();
}
