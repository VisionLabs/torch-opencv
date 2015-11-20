#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/features2d.hpp>

struct KeyPointWrapper {
    struct Point2fWrapper pt;
    float size, angle, response;
    int octave, class_id;
    
    KeyPointWrapper(const cv::KeyPoint & other);
    inline operator cv::KeyPoint() { return cv::KeyPoint(pt, size, angle, response, octave, class_id); }
};

struct KeyPointArray {
    struct KeyPointWrapper *data;
    int size;

    KeyPointArray(const std::vector<cv::KeyPoint> & v);
    operator std::vector<cv::KeyPoint>();
};

struct KeyPointMat {
    struct KeyPointWrapper **data;
    int size1, size2;

    KeyPointMat(const std::vector<std::vector<cv::KeyPoint> > & v);
    operator std::vector<std::vector<cv::KeyPoint> >();
};

// KeyPointsFilter

struct KeyPointsFilterPtr {
    void *ptr;

    inline cv::KeyPointsFilter * operator->() { return static_cast<cv::KeyPointsFilter *>(ptr); }
    inline KeyPointsFilterPtr(cv::KeyPointsFilter *ptr) { this->ptr = ptr; }
    inline cv::KeyPointsFilter & operator*() { return *static_cast<cv::KeyPointsFilter *>(this->ptr); }
};

extern "C" struct KeyPointsFilterPtr KeyPointsFilter_ctor();

extern "C" void KeyPointsFilter_dtor(struct KeyPointsFilterPtr ptr);

extern "C" struct KeyPointArray KeyPointsFilter_runByImageBorder(struct KeyPointArray keypoints,
                        struct SizeWrapper imageSize, int borderSize);

extern "C" struct KeyPointArray KeyPointsFilter_runByKeypointSize(struct KeyPointArray keypoints,
                        float minSize, float maxSize);

extern "C" struct KeyPointArray KeyPointsFilter_runByPixelsMask(struct KeyPointArray keypoints,
                        struct TensorWrapper mask);

extern "C" struct KeyPointArray KeyPointsFilter_removeDuplicated(struct KeyPointArray keypoints);

extern "C" struct KeyPointArray KeyPointsFilter_retainBest(struct KeyPointArray keypoints, int npoints);

// Feature2D

struct Feature2DPtr {
    void *ptr;

    inline cv::Feature2D * operator->() { return static_cast<cv::Feature2D *>(ptr); }
    inline Feature2DPtr(cv::Feature2D *ptr) { this->ptr = ptr; }
    inline cv::Feature2D & operator*() { return *static_cast<cv::Feature2D *>(this->ptr); }
};

extern "C" struct Feature2DPtr Feature2D_ctor();

extern "C" struct KeyPointArray Feature2D_detect(struct Feature2DPtr ptr, struct TensorWrapper image,
                        struct KeyPointArray keypoints, struct TensorWrapper mask);

extern "C" struct KeyPointMat Feature2D_detect2(struct Feature2DPtr ptr, struct TensorArray images,
                        struct KeyPointMat keypoints, struct TensorArray masks);

extern "C" struct KeyPointArray Feature2D_compute(struct Feature2DPtr ptr, struct TensorWrapper image,
                        struct KeyPointArray keypoints, struct TensorWrapper descriptors);

extern "C" struct KeyPointMat Feature2D_compute2(struct Feature2DPtr ptr, struct TensorArray images,
                        struct KeyPointMat keypoints, struct TensorArray descriptors);

extern "C" struct KeyPointArray Feature2D_detectAndCompute(struct Feature2DPtr ptr, struct TensorWrapper image,
                        struct TensorWrapper mask, struct KeyPointArray keypoints,
                        struct TensorWrapper descriptors, bool useProvidedKeypoints);

extern "C" int Feature2D_descriptorSize(struct Feature2DPtr ptr);

extern "C" int Feature2D_descriptorType(struct Feature2DPtr ptr);

extern "C" int Feature2D_defaultNorm(struct Feature2DPtr ptr);

extern "C" bool Feature2D_empty(struct Feature2DPtr ptr);

// BRISK

struct BRISKPtr {
    void *ptr;
    inline cv::BRISK * operator->() { return static_cast<cv::BRISK *>(ptr); }
    inline BRISKPtr(cv::BRISK *ptr) { this->ptr = ptr; }
    inline cv::BRISK & operator*() { return *static_cast<cv::BRISK *>(this->ptr); }
};

extern "C" struct BRISKPtr BRISK_ctor(int thresh, int octaves, float patternScale);

extern "C" struct BRISKPtr BRISK_ctor2(struct TensorWrapper radiusList, struct TensorWrapper numberList,
                        float dMax, float dMin, struct TensorWrapper indexChange);

// ORB

struct ORBPtr {
    void *ptr;
    inline cv::ORB * operator->() { return static_cast<cv::ORB *>(ptr); }
    inline ORBPtr(cv::ORB *ptr) { this->ptr = ptr; }
    inline cv::ORB & operator*() { return *static_cast<cv::ORB *>(this->ptr); }
};

extern "C" struct ORBPtr ORB_ctor(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel,
                        int WTA_K, int scoreType, int patchSize, int fastThreshold);

extern "C" void ORB_setMaxFeatures(struct ORBPtr ptr, int maxFeatures);

extern "C" int ORB_getMaxFeatures(struct ORBPtr ptr);

extern "C" void ORB_setScaleFactor(struct ORBPtr ptr, int scaleFactor);

extern "C" int ORB_getScaleFactor(struct ORBPtr ptr);

extern "C" void ORB_setNLevels(struct ORBPtr ptr, int nlevels);

extern "C" int ORB_getNLevels(struct ORBPtr ptr);

extern "C" void ORB_setEdgeThreshold(struct ORBPtr ptr, int edgeThreshold);

extern "C" int ORB_getEdgeThreshold(struct ORBPtr ptr);

extern "C" void ORB_setFirstLevel(struct ORBPtr ptr, int firstLevel);

extern "C" int ORB_getFirstLevel(struct ORBPtr ptr);

extern "C" void ORB_setWTA_K(struct ORBPtr ptr, int wta_k);

extern "C" int ORB_getWTA_K(struct ORBPtr ptr);

extern "C" void ORB_setScoreType(struct ORBPtr ptr, int scoreType);

extern "C" int ORB_getScoreType(struct ORBPtr ptr);

extern "C" void ORB_setPatchSize(struct ORBPtr ptr, int patchSize);

extern "C" int ORB_getPatchSize(struct ORBPtr ptr);

extern "C" void ORB_setFastThreshold(struct ORBPtr ptr, int fastThreshold);

extern "C" int ORB_getFastThreshold(struct ORBPtr ptr);










extern "C" struct KeyPointArray AGAST(struct TensorWrapper image, int threshold, bool nonmaxSuppression);