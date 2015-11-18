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







extern "C" struct KeyPointArray AGAST(struct TensorWrapper image, int threshold, bool nonmaxSuppression);