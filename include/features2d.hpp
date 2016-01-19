#include <Common.hpp>
#include <Classes.hpp>
#include <flann.hpp>
#include <opencv2/features2d.hpp>

struct evaluateFeatureDetectorRetval {
    struct KeyPointArray keypoints1, keypoints2;
    float repeatability;
    int correspCount;
};

// KeyPointsFilter
struct KeyPointsFilterPtr {
    void *ptr;

    inline cv::KeyPointsFilter * operator->() { return static_cast<cv::KeyPointsFilter *>(ptr); }
    inline KeyPointsFilterPtr(cv::KeyPointsFilter *ptr) { this->ptr = ptr; }
    inline cv::KeyPointsFilter & operator*() { return *static_cast<cv::KeyPointsFilter *>(this->ptr); }
};

extern "C"
struct KeyPointsFilterPtr KeyPointsFilter_ctor();

extern "C"
void KeyPointsFilter_dtor(struct KeyPointsFilterPtr ptr);

extern "C"
struct KeyPointArray KeyPointsFilter_runByImageBorder(struct KeyPointArray keypoints,
                                                      struct SizeWrapper imageSize, int borderSize);

extern "C"
struct KeyPointArray KeyPointsFilter_runByKeypointSize(struct KeyPointArray keypoints,
                                                       float minSize, float maxSize);

extern "C"
struct KeyPointArray KeyPointsFilter_runByPixelsMask(struct KeyPointArray keypoints,
                                                     struct TensorWrapper mask);

extern "C"
struct KeyPointArray KeyPointsFilter_removeDuplicated(struct KeyPointArray keypoints);

extern "C"
struct KeyPointArray KeyPointsFilter_retainBest(struct KeyPointArray keypoints, int npoints);

// Feature2D

struct Feature2DPtr {
    void *ptr;

    inline cv::Feature2D * operator->() { return static_cast<cv::Feature2D *>(ptr); }
    inline Feature2DPtr(cv::Feature2D *ptr) { this->ptr = ptr; }
    inline cv::Feature2D & operator*() { return *static_cast<cv::Feature2D *>(this->ptr); }
};

extern "C"
struct KeyPointArray Feature2D_detect(
        struct Feature2DPtr ptr, struct TensorWrapper image, struct TensorWrapper mask);

extern "C"
struct TensorPlusKeyPointArray Feature2D_compute(struct Feature2DPtr ptr, struct TensorWrapper image,
                        struct KeyPointArray keypoints, struct TensorWrapper descriptors);

extern "C"
struct TensorPlusKeyPointArray Feature2D_detectAndCompute(struct Feature2DPtr ptr, struct TensorWrapper image,
                        struct TensorWrapper mask,
                        struct TensorWrapper descriptors, bool useProvidedKeypoints);

extern "C"
int Feature2D_descriptorSize(struct Feature2DPtr ptr);

extern "C"
int Feature2D_descriptorType(struct Feature2DPtr ptr);

extern "C"
int Feature2D_defaultNorm(struct Feature2DPtr ptr);

extern "C"
bool Feature2D_empty(struct Feature2DPtr ptr);

// BRISK

struct BRISKPtr {
    void *ptr;
    inline cv::BRISK * operator->() { return static_cast<cv::BRISK *>(ptr); }
    inline BRISKPtr(cv::BRISK *ptr) { this->ptr = ptr; }
    inline cv::BRISK & operator*() { return *static_cast<cv::BRISK *>(this->ptr); }
};

extern "C"
struct BRISKPtr BRISK_ctor(int thresh, int octaves, float patternScale);

extern "C"
struct BRISKPtr BRISK_ctor2(struct TensorWrapper radiusList, struct TensorWrapper numberList,
                            float dMax, float dMin, struct TensorWrapper indexChange);

// ORB

struct ORBPtr {
    void *ptr;
    inline cv::ORB * operator->() { return static_cast<cv::ORB *>(ptr); }
    inline ORBPtr(cv::ORB *ptr) { this->ptr = ptr; }
    inline cv::ORB & operator*() { return *static_cast<cv::ORB *>(this->ptr); }
};

extern "C"
struct ORBPtr ORB_ctor(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel,
                       int WTA_K, int scoreType, int patchSize, int fastThreshold);

extern "C"
void ORB_setMaxFeatures(struct ORBPtr ptr, int maxFeatures);

extern "C"
int ORB_getMaxFeatures(struct ORBPtr ptr);

extern "C"
void ORB_setScaleFactor(struct ORBPtr ptr, int scaleFactor);

extern "C"
double ORB_getScaleFactor(struct ORBPtr ptr);

extern "C"
void ORB_setNLevels(struct ORBPtr ptr, int nlevels);

extern "C"
int ORB_getNLevels(struct ORBPtr ptr);

extern "C"
void ORB_setEdgeThreshold(struct ORBPtr ptr, int edgeThreshold);

extern "C"
int ORB_getEdgeThreshold(struct ORBPtr ptr);

extern "C"
void ORB_setFirstLevel(struct ORBPtr ptr, int firstLevel);

extern "C"
int ORB_getFirstLevel(struct ORBPtr ptr);

extern "C"
void ORB_setWTA_K(struct ORBPtr ptr, int wta_k);

extern "C"
int ORB_getWTA_K(struct ORBPtr ptr);

extern "C"
void ORB_setScoreType(struct ORBPtr ptr, int scoreType);

extern "C"
int ORB_getScoreType(struct ORBPtr ptr);

extern "C"
void ORB_setPatchSize(struct ORBPtr ptr, int patchSize);

extern "C"
int ORB_getPatchSize(struct ORBPtr ptr);

extern "C"
void ORB_setFastThreshold(struct ORBPtr ptr, int fastThreshold);

extern "C"
int ORB_getFastThreshold(struct ORBPtr ptr);

// MSER

struct MSERPtr {
    void *ptr;
    inline cv::MSER * operator->() { return static_cast<cv::MSER *>(ptr); }
    inline MSERPtr(cv::MSER *ptr) { this->ptr = ptr; }
    inline cv::MSER & operator*() { return *static_cast<cv::MSER *>(this->ptr); }
};

extern "C"
struct MSERPtr MSER_ctor(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity,
                         int _max_evolution, double _area_threshold, double _min_margin, int _edge_blur_size);

extern "C"
struct TensorArray MSER_detectRegions(struct MSERPtr ptr,
                                      struct TensorWrapper image, struct TensorWrapper bboxes);

extern "C"
void MSER_setDelta(struct MSERPtr ptr, int delta);

extern "C"
int MSER_getDelta(struct MSERPtr ptr);

extern "C"
void MSER_setMinArea(struct MSERPtr ptr, int minArea);

extern "C"
int MSER_getMinArea(struct MSERPtr ptr);

extern "C"
void MSER_setMaxArea(struct MSERPtr ptr, int MaxArea);

extern "C"
int MSER_getMaxArea(struct MSERPtr ptr);

extern "C"
void MSER_setPass2Only(struct MSERPtr ptr, bool Pass2Only);

extern "C"
bool MSER_getPass2Only(struct MSERPtr ptr);

// functions

extern "C"
struct KeyPointArray FAST(
        struct TensorWrapper image, int threshold, bool nonmaxSuppression);

extern "C"
struct KeyPointArray FAST_type(
        struct TensorWrapper image, int threshold, bool nonmaxSuppression, int type);

extern "C"
struct KeyPointArray AGAST(struct TensorWrapper image, int threshold, bool nonmaxSuppression);

extern "C"
struct KeyPointArray AGAST_type(struct TensorWrapper image, int threshold, bool nonmaxSuppression, int type);

// FastFeatureDetector

struct FastFeatureDetectorPtr {
    void *ptr;
    inline cv::FastFeatureDetector * operator->() { return static_cast<cv::FastFeatureDetector *>(ptr); }
    inline FastFeatureDetectorPtr(cv::FastFeatureDetector *ptr) { this->ptr = ptr; }
    inline cv::FastFeatureDetector & operator*() { return *static_cast<cv::FastFeatureDetector *>(this->ptr); }
};

extern "C"
struct FastFeatureDetectorPtr FastFeatureDetector_ctor(
        int threshold, bool nonmaxSuppression, int type);

extern "C"
void FastFeatureDetector_setThreshold(struct FastFeatureDetectorPtr ptr, int val);

extern "C"
int FastFeatureDetector_getThreshold(struct FastFeatureDetectorPtr ptr);

extern "C"
void FastFeatureDetector_setNonmaxSuppression(struct FastFeatureDetectorPtr ptr, bool val);

extern "C"
bool FastFeatureDetector_getNonmaxSuppression(struct FastFeatureDetectorPtr ptr);

extern "C"
void FastFeatureDetector_setType(struct FastFeatureDetectorPtr ptr, int val);

extern "C"
int FastFeatureDetector_getType(struct FastFeatureDetectorPtr ptr);

// AgastFeatureDetector

struct AgastFeatureDetectorPtr {
    void *ptr;
    inline cv::AgastFeatureDetector * operator->() { return static_cast<cv::AgastFeatureDetector *>(ptr); }
    inline AgastFeatureDetectorPtr(cv::AgastFeatureDetector *ptr) { this->ptr = ptr; }
    inline cv::AgastFeatureDetector & operator*() { return *static_cast<cv::AgastFeatureDetector *>(this->ptr); }
};

extern "C"
struct AgastFeatureDetectorPtr AgastFeatureDetector_ctor(
        int threshold, bool nonmaxSuppression, int type);

extern "C"
void AgastFeatureDetector_setThreshold(struct AgastFeatureDetectorPtr ptr, int val);

extern "C"
int AgastFeatureDetector_getThreshold(struct AgastFeatureDetectorPtr ptr);

extern "C"
void AgastFeatureDetector_setNonmaxSuppression(struct AgastFeatureDetectorPtr ptr, bool val);

extern "C"
bool AgastFeatureDetector_getNonmaxSuppression(struct AgastFeatureDetectorPtr ptr);

extern "C"
void AgastFeatureDetector_setType(struct AgastFeatureDetectorPtr ptr, int val);

extern "C"
int AgastFeatureDetector_getType(struct AgastFeatureDetectorPtr ptr);

// GFTTDetector

struct GFTTDetectorPtr {
    void *ptr;
    inline cv::GFTTDetector * operator->() { return static_cast<cv::GFTTDetector *>(ptr); }
    inline GFTTDetectorPtr(cv::GFTTDetector *ptr) { this->ptr = ptr; }
    inline cv::GFTTDetector & operator*() { return *static_cast<cv::GFTTDetector *>(this->ptr); }
};

extern "C"
struct GFTTDetectorPtr GFTTDetector_ctor(
        int maxCorners, double qualityLevel, double minDistance,
        int blockSize, bool useHarrisDetector, double k);

extern "C"
void GFTTDetector_setMaxFeatures(struct GFTTDetectorPtr ptr, int val);

extern "C"
int GFTTDetector_getMaxFeatures(struct GFTTDetectorPtr ptr);

extern "C"
void GFTTDetector_setQualityLevel(struct GFTTDetectorPtr ptr, double val);

extern "C"
double GFTTDetector_getQualityLevel(struct GFTTDetectorPtr ptr);

extern "C"
void GFTTDetector_setMinDistance(struct GFTTDetectorPtr ptr, double val);

extern "C"
double GFTTDetector_getMinDistance(struct GFTTDetectorPtr ptr);

extern "C"
void GFTTDetector_setBlockSize(struct GFTTDetectorPtr ptr, int val);

extern "C"
int GFTTDetector_getBlockSize(struct GFTTDetectorPtr ptr);

extern "C"
void GFTTDetector_setHarrisDetector(struct GFTTDetectorPtr ptr, bool val);

extern "C"
bool GFTTDetector_getHarrisDetector(struct GFTTDetectorPtr ptr);

extern "C"
void GFTTDetector_setK(struct GFTTDetectorPtr ptr, double val);

extern "C"
double GFTTDetector_getK(struct GFTTDetectorPtr ptr);

// SimpleBlobDetector

struct SimpleBlobDetectorPtr {
    void *ptr;
    inline cv::SimpleBlobDetector * operator->() { return static_cast<cv::SimpleBlobDetector *>(ptr); }
    inline SimpleBlobDetectorPtr(cv::SimpleBlobDetector *ptr) { this->ptr = ptr; }
    inline cv::SimpleBlobDetector & operator*() { return *static_cast<cv::SimpleBlobDetector *>(this->ptr); }
};

extern "C"
cv::SimpleBlobDetector::Params SimpleBlobDetector_Params_default();

extern "C"
struct SimpleBlobDetectorPtr SimpleBlobDetector_ctor(cv::SimpleBlobDetector::Params params);

extern "C"
cv::SimpleBlobDetector::Params SimpleBlobDetector_Params_default();

extern "C"
struct SimpleBlobDetectorPtr SimpleBlobDetector_ctor(cv::SimpleBlobDetector::Params params);

// KAZE

struct KAZEPtr {
    void *ptr;
    inline cv::KAZE * operator->() { return static_cast<cv::KAZE *>(ptr); }
    inline KAZEPtr(cv::KAZE *ptr) { this->ptr = ptr; }
    inline cv::KAZE & operator*() { return *static_cast<cv::KAZE *>(this->ptr); }
};

extern "C"
struct KAZEPtr KAZE_ctor(
        bool extended, bool upright, float threshold,
        int nOctaves, int nOctaveLayers, int diffusivity);

extern "C"
void KAZE_setExtended(struct KAZEPtr ptr, bool val);

extern "C"
bool KAZE_getExtended(struct KAZEPtr ptr);

extern "C"
void KAZE_setUpright(struct KAZEPtr ptr, bool val);

extern "C"
bool KAZE_getUpright(struct KAZEPtr ptr);

extern "C"
void KAZE_setThreshold(struct KAZEPtr ptr, double val);

extern "C"
double KAZE_getThreshold(struct KAZEPtr ptr);

extern "C"
void KAZE_setNOctaves(struct KAZEPtr ptr, int val);

extern "C"
int KAZE_getNOctaves(struct KAZEPtr ptr);

extern "C"
void KAZE_setNOctaveLayers(struct KAZEPtr ptr, int val);

extern "C"
int KAZE_getNOctaveLayers(struct KAZEPtr ptr);

extern "C"
void KAZE_setDiffusivity(struct KAZEPtr ptr, int val);

extern "C"
int KAZE_getDiffusivity(struct KAZEPtr ptr);

// AKAZE

struct AKAZEPtr {
    void *ptr;
    inline cv::AKAZE * operator->() { return static_cast<cv::AKAZE *>(ptr); }
    inline AKAZEPtr(cv::AKAZE *ptr) { this->ptr = ptr; }
    inline cv::AKAZE & operator*() { return *static_cast<cv::AKAZE *>(this->ptr); }
};

extern "C"
struct AKAZEPtr AKAZE_ctor(
        int descriptor_type, int descriptor_size, int descriptor_channels,
        float threshold, int nOctaves, int nOctaveLayers, int diffusivity);

extern "C"
void AKAZE_setDescriptorType(struct AKAZEPtr ptr, int val);

extern "C"
int AKAZE_getDescriptorType(struct AKAZEPtr ptr);

extern "C"
void AKAZE_setDescriptorSize(struct AKAZEPtr ptr, int val);

extern "C"
int AKAZE_getDescriptorSize(struct AKAZEPtr ptr);

extern "C"
void AKAZE_setDescriptorChannels(struct AKAZEPtr ptr, int val);

extern "C"
int AKAZE_getDescriptorChannels(struct AKAZEPtr ptr);

extern "C"
void AKAZE_setThreshold(struct AKAZEPtr ptr, double val);

extern "C"
double AKAZE_getThreshold(struct AKAZEPtr ptr);

extern "C"
void AKAZE_setNOctaves(struct AKAZEPtr ptr, int val);

extern "C"
int AKAZE_getNOctaves(struct AKAZEPtr ptr);

extern "C"
void AKAZE_setNOctaveLayers(struct AKAZEPtr ptr, int val);

extern "C"
int AKAZE_getNOctaveLayers(struct AKAZEPtr ptr);

extern "C"
void AKAZE_setDiffusivity(struct AKAZEPtr ptr, int val);

extern "C"
int AKAZE_getDiffusivity(struct AKAZEPtr ptr);

// DescriptorMatcher

struct DescriptorMatcherPtr {
    void *ptr;
    inline cv::DescriptorMatcher * operator->() { return static_cast<cv::DescriptorMatcher *>(ptr); }
    inline DescriptorMatcherPtr(cv::DescriptorMatcher *ptr) { this->ptr = ptr; }
    inline cv::DescriptorMatcher & operator*() { return *static_cast<cv::DescriptorMatcher *>(this->ptr); }
};

extern "C"
struct DescriptorMatcherPtr DescriptorMatcher_ctor(const char *descriptorMatcherType);

extern "C"
void DescriptorMatcher_add(struct DescriptorMatcherPtr ptr, struct TensorArray descriptors);

extern "C"
struct TensorArray DescriptorMatcher_getTrainDescriptors(struct DescriptorMatcherPtr ptr);

extern "C"
void DescriptorMatcher_clear(struct DescriptorMatcherPtr ptr);

extern "C"
bool DescriptorMatcher_empty(struct DescriptorMatcherPtr ptr);

extern "C"
bool DescriptorMatcher_isMaskSupported(struct DescriptorMatcherPtr ptr);

extern "C"
void DescriptorMatcher_train(struct DescriptorMatcherPtr ptr);

extern "C"
struct DMatchArray DescriptorMatcher_match(struct DescriptorMatcherPtr ptr,
                                           struct TensorWrapper queryDescriptors, struct TensorWrapper mask);

extern "C"
struct DMatchArray DescriptorMatcher_match_trainDescriptors(struct DescriptorMatcherPtr ptr,
                                                            struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
                                                            struct TensorWrapper mask);

extern "C"
struct DMatchArrayOfArrays DescriptorMatcher_knnMatch(struct DescriptorMatcherPtr ptr,
                                                      struct TensorWrapper queryDescriptors, int k,
                                                      struct TensorWrapper mask, bool compactResult);

extern "C"
struct DMatchArrayOfArrays DescriptorMatcher_knnMatch_trainDescriptors(struct DescriptorMatcherPtr ptr,
                                                                       struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
                                                                       int k, struct TensorWrapper mask, bool compactResult);

// BFMatcher

struct BFMatcherPtr {
    void *ptr;
    inline cv::BFMatcher * operator->() { return static_cast<cv::BFMatcher *>(ptr); }
    inline BFMatcherPtr(cv::BFMatcher *ptr) { this->ptr = ptr; }
    inline cv::BFMatcher & operator*() { return *static_cast<cv::BFMatcher *>(this->ptr); }
};

extern "C"
struct BFMatcherPtr BFMatcher_ctor(int normType, bool crossCheck);

// FlannBasedMatcher

struct FlannBasedMatcherPtr {
    void *ptr;
    inline cv::FlannBasedMatcher * operator->() { return static_cast<cv::FlannBasedMatcher *>(ptr); }
    inline FlannBasedMatcherPtr(cv::FlannBasedMatcher *ptr) { this->ptr = ptr; }
    inline cv::FlannBasedMatcher & operator*() { return *static_cast<cv::FlannBasedMatcher *>(this->ptr); }
};

extern "C"
struct FlannBasedMatcherPtr FlannBasedMatcher_ctor(
        struct IndexParamsPtr indexParams, struct SearchParamsPtr searchParams);

// functions

extern "C"
struct TensorWrapper drawKeypoints(
        struct TensorWrapper image, struct KeyPointArray keypoints, struct TensorWrapper outImage,
        struct ScalarWrapper color, int flags);

extern "C"
struct TensorWrapper drawMatches(
        struct TensorWrapper img1, struct KeyPointArray keypoints1,
        struct TensorWrapper img2, struct KeyPointArray keypoints2,
        struct DMatchArray matches1to2, struct TensorWrapper outImg,
        struct ScalarWrapper matchColor, struct ScalarWrapper singlePointColor,
        struct TensorWrapper matchesMask, int flags);

extern "C"
struct TensorWrapper drawMatchesKnn(
        struct TensorWrapper img1, struct KeyPointArray keypoints1,
        struct TensorWrapper img2, struct KeyPointArray keypoints2,
        struct DMatchArrayOfArrays matches1to2, struct TensorWrapper outImg,
        struct ScalarWrapper matchColor, struct ScalarWrapper singlePointColor,
        struct TensorArray matchesMask, int flags);

extern "C"
struct evaluateFeatureDetectorRetval evaluateFeatureDetector(
        struct TensorWrapper img1, struct TensorWrapper img2, struct TensorWrapper H1to2,
        struct Feature2DPtr fdetector);

extern "C"
struct TensorWrapper computeRecallPrecisionCurve(
        struct DMatchArrayOfArrays matches1to2, struct TensorArray correctMatches1to2Mask);

extern "C"
float getRecall(struct TensorWrapper recallPrecisionCurve, float l_precision);

extern "C"
int getNearestPoint(struct TensorWrapper recallPrecisionCurve, float l_precision);

// BOWTrainer

struct BOWTrainerPtr {
    void *ptr;
    inline cv::BOWTrainer * operator->() { return static_cast<cv::BOWTrainer *>(ptr); }
    inline BOWTrainerPtr(cv::BOWTrainer *ptr) { this->ptr = ptr; }
    inline cv::BOWTrainer & operator*() { return *static_cast<cv::BOWTrainer *>(this->ptr); }
};

extern "C"
void BOWTrainer_dtor(struct BOWTrainerPtr ptr);

extern "C"
void BOWTrainer_add(struct BOWTrainerPtr ptr, struct TensorWrapper descriptors);

extern "C"
struct TensorArray BOWTrainer_getDescriptors(struct BOWTrainerPtr ptr);

extern "C"
int BOWTrainer_descriptorsCount(struct BOWTrainerPtr ptr);

extern "C"
void BOWTrainer_clear(struct BOWTrainerPtr ptr);

extern "C"
struct TensorWrapper BOWTrainer_cluster(struct BOWTrainerPtr ptr);

extern "C"
struct TensorWrapper BOWTrainer_cluster_descriptors(struct BOWTrainerPtr ptr, struct TensorWrapper descriptors);

// BOWKMeansTrainer

struct BOWKMeansTrainerPtr {
    void *ptr;
    inline cv::BOWKMeansTrainer * operator->() { return static_cast<cv::BOWKMeansTrainer *>(ptr); }
    inline BOWKMeansTrainerPtr(cv::BOWKMeansTrainer *ptr) { this->ptr = ptr; }
    inline cv::BOWKMeansTrainer & operator*() { return *static_cast<cv::BOWKMeansTrainer *>(this->ptr); }
};

extern "C"
struct BOWKMeansTrainerPtr BOWKMeansTrainer_ctor(
        int clusterCount, struct TermCriteriaWrapper termcrit,
        int attempts, int flags);

// BOWImgDescriptorExtractor

struct BOWImgDescriptorExtractorPtr {
    void *ptr;
    inline cv::BOWImgDescriptorExtractor * operator->() { return static_cast<cv::BOWImgDescriptorExtractor *>(ptr); }
    inline BOWImgDescriptorExtractorPtr(cv::BOWImgDescriptorExtractor *ptr) { this->ptr = ptr; }
    inline cv::BOWImgDescriptorExtractor & operator*() { return *static_cast<cv::BOWImgDescriptorExtractor *>(this->ptr); }
};

extern "C"
struct BOWImgDescriptorExtractorPtr BOWImgDescriptorExtractor_ctor(
        struct Feature2DPtr dextractor, struct DescriptorMatcherPtr dmatcher);

extern "C"
void BOWImgDescriptorExtractor_dtor(struct BOWImgDescriptorExtractorPtr ptr);

extern "C"
void BOWImgDescriptorExtractor_setVocabulary(
        struct BOWImgDescriptorExtractorPtr ptr, struct TensorWrapper vocabulary);

extern "C"
struct TensorWrapper BOWImgDescriptorExtractor_getVocabulary(struct BOWImgDescriptorExtractorPtr ptr);

extern "C"
struct TensorWrapper BOWImgDescriptorExtractor_compute(
        struct BOWImgDescriptorExtractorPtr ptr, struct TensorWrapper image,
        struct KeyPointArray keypoints, struct TensorWrapper imgDescriptor);

extern "C"
int BOWImgDescriptorExtractor_descriptorSize(struct BOWImgDescriptorExtractorPtr ptr);

extern "C"
int BOWImgDescriptorExtractor_descriptorType(struct BOWImgDescriptorExtractorPtr ptr);
