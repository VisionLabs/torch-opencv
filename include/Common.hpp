#pragma once

extern "C" {
#include <TH/TH.h>
}

#include <opencv2/core.hpp>

#ifdef WITH_CUDA
#include <THC/THC.h>
#include <opencv2/core/cuda.hpp>

class GpuMatT;
#endif

#include <iostream>
#include <array>

extern "C" int getIntMax() { return INT_MAX; }
extern "C" float getFloatMax() { return FLT_MAX; }
extern "C" double getDblEpsilon() { return DBL_EPSILON; }

/***************** Tensor <=> Mat conversion *****************/

#define TO_MAT_OR_NOARRAY(mat) (mat.isNull() ? cv::noArray() : mat.toMat())
#define TO_GPUMAT_OR_NOARRAY(mat) (mat.isNull() ? cv::noArray() : mat.toGpuMat())

#define TO_MAT_LIST_OR_NOARRAY(mat) (mat.isNull() ? cv::noArray() : mat.toMatList())
#define TO_GPUMAT_LIST_OR_NOARRAY(mat) (mat.isNull() ? std::vector<cuda::GpuMat>() : mat.toGpuMatList())

extern "C"
void initAllocator();

class MatT {
public:
    cv::Mat mat;
    // The Tensor that `mat` was created from, or nullptr
    THByteTensor *tensor;

    operator cv::_InputOutputArray() { return this->mat; }

    MatT(cv::Mat && mat);
    MatT(cv::Mat & mat);
    MatT();
};

struct TensorWrapper {
    THByteTensor *tensorPtr;
    char typeCode;
    bool definedInLua;

    TensorWrapper();
    TensorWrapper(cv::Mat & mat);
    TensorWrapper(cv::Mat && mat);
    TensorWrapper(MatT & mat);
    TensorWrapper(MatT && mat);

    operator cv::Mat();
    // synonym for operator cv::Mat()
    cv::Mat toMat() { return *this; }
    MatT toMatT();

    #ifdef WITH_CUDA
    TensorWrapper(cv::cuda::GpuMat & mat, THCState *state);
    TensorWrapper(cv::cuda::GpuMat && mat, THCState *state);
    TensorWrapper(GpuMatT & mat, THCState *state);
    TensorWrapper(GpuMatT && mat, THCState *state);

    cv::cuda::GpuMat toGpuMat(int depth = -1);
    GpuMatT toGpuMatT();
    #endif

    bool isNull() { return tensorPtr == nullptr; }
};

struct TensorArray {
    struct TensorWrapper *tensors;
    int size;

    TensorArray();
    TensorArray(std::vector<cv::Mat> & matList);
    TensorArray(std::vector<MatT> & matList);
    explicit TensorArray(short size);

    #ifdef WITH_CUDA
    TensorArray(std::vector<cv::cuda::GpuMat> & matList, THCState *state);
    std::vector<cv::cuda::GpuMat> toGpuMatList();
    #endif

    operator std::vector<cv::Mat>();
    operator std::vector<MatT>();
    // synonyms for operators
    std::vector<cv::Mat> toMatList()  { return *this; }
    std::vector<MatT>    toMatTList() { return *this; }

    bool isNull() { return tensors == nullptr; }
};

inline
std::string typeStr(cv::Mat & mat) {
    switch (mat.depth()) {
        case CV_8U:  return "Byte";
        case CV_8S:  return "Char";
        case CV_16S: return "Short";
        case CV_32S: return "Int";
        case CV_32F: return "Float";
        case CV_64F: return "Double";
        default:     return "Unknown";
    }
}

/***************** Wrappers for small OpenCV classes *****************/

struct SizeWrapper {
    int width, height;

    inline operator cv::Size() { return cv::Size(width, height); }
    SizeWrapper(const cv::Size & other);
    inline SizeWrapper() {}
};

struct Size2fWrapper {
    float width, height;

    inline operator cv::Size2f() { return cv::Size2f(width, height); }
    inline Size2fWrapper() {}
    Size2fWrapper(const cv::Size2f & other);
};

struct TermCriteriaWrapper {
    int type, maxCount;
    double epsilon;

    TermCriteriaWrapper() {}

    inline operator cv::TermCriteria() { return cv::TermCriteria(type, maxCount, epsilon); }
    inline cv::TermCriteria orDefault(cv::TermCriteria defaultVal) {
        return (this->type == 0 ? defaultVal : *this);
    }
    TermCriteriaWrapper(cv::TermCriteria && other);
};

struct ScalarWrapper {
    double v0, v1, v2, v3;

    inline operator cv::Scalar() { return cv::Scalar(v0, v1, v2, v3); }
    inline cv::Scalar orDefault(cv::Scalar defaultVal) {
        return (isnan(this->v0) ? defaultVal : *this);
    }
};

struct Vec2dWrapper {
    double v0, v1;

    inline operator cv::Vec2d() { return cv::Vec2d(v0, v1); }
    inline Vec2dWrapper(const cv::Vec2d & other) {
        this->v0 = other.val[0];
        this->v1 = other.val[1];
    }
};

struct Vec3dWrapper {
    double v0, v1, v2;
    inline operator cv::Vec3d() { return cv::Vec3d(v0, v1, v2); }
    Vec3dWrapper & operator=(cv::Vec3d & other);
    Vec3dWrapper (const cv::Vec3d & other);
    inline Vec3dWrapper() {}
};

struct Vec3fWrapper {
    float v0, v1, v2;
};

struct Vec3iWrapper {
    int v0, v1, v2;
};

struct Vec4iWrapper {
    int v0, v1, v2, v3;
};

struct RectWrapper {
    int x, y, width, height;

    inline operator cv::Rect() { return cv::Rect(x, y, width, height); }
    RectWrapper & operator=(cv::Rect & other);
    RectWrapper(const cv::Rect & other);
    inline RectWrapper() {}
};

struct PointWrapper {
    int x, y;

    inline operator cv::Point() { return cv::Point(x, y); }

    PointWrapper() {}
    PointWrapper(const cv::Point & other);
};

struct Point2fWrapper {
    float x, y;

    inline operator cv::Point2f() { return cv::Point2f(x, y); }
    Point2fWrapper(const cv::Point2f & other);
    inline Point2fWrapper() {}
};

struct Point2dWrapper {
    double x, y;

    inline operator cv::Point2d() { return cv::Point2d(x, y); }
    Point2dWrapper(const cv::Point2d & other);
    inline Point2dWrapper() {}
};

struct RotatedRectWrapper {
    struct Point2fWrapper center;
    struct Size2fWrapper size;
    float angle;

    RotatedRectWrapper() {}
    RotatedRectWrapper(const cv::RotatedRect & other);
    inline operator cv::RotatedRect() { return cv::RotatedRect(center, size, angle); }
};

struct MomentsWrapper {
    double m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
    double mu20, mu11, mu02, mu30, mu21, mu12, mu03;
    double nu20, nu11, nu02, nu30, nu21, nu12, nu03;

    MomentsWrapper(const cv::Moments & other);
    inline operator cv::Moments() {
        return cv::Moments(m00, m10, m01, m20, m11, m02, m30, m21, m12, m03);
    }
};

struct RotatedRectPlusRect {
    struct RotatedRectWrapper rotrect;
    struct RectWrapper rect;
};

struct DMatchWrapper {
    int queryIdx;
    int trainIdx;
    int imgIdx;
    float distance;
};

struct DMatchArray {
    int size;
    struct DMatchWrapper *data;

    DMatchArray() {}
    DMatchArray(const std::vector<cv::DMatch> & other);
    operator std::vector<cv::DMatch>();
};

struct DMatchArrayOfArrays {
    int size;
    struct DMatchArray *data;

    DMatchArrayOfArrays() {}
    DMatchArrayOfArrays(const std::vector<std::vector<cv::DMatch>> & other);
    operator std::vector<std::vector<cv::DMatch>>();
};

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

    KeyPointArray() {}
    KeyPointArray(const std::vector<cv::KeyPoint> & v);
    operator std::vector<cv::KeyPoint>();
};

/***************** Helper wrappers for [OpenCV class + some primitive] *****************/

struct TensorPlusDouble {
    struct TensorWrapper tensor;
    double val;
};

struct TensorPlusFloat {
    struct TensorWrapper tensor;
    float val;
};

struct TensorPlusInt {
    struct TensorWrapper tensor;
    int val;
};

struct TensorPlusBool {
    struct TensorWrapper tensor;
    bool val;
};

struct TensorPlusRect {
    struct TensorWrapper tensor;
    struct RectWrapper rect;
};

struct TensorPlusPoint {
    struct TensorWrapper tensor;
    struct PointWrapper point;
};

struct TensorArrayPlusFloat {
    struct TensorArray tensors;
    float val;
};

struct TensorArrayPlusDouble {
    struct TensorArray tensors;
    double val;
};

struct TensorArrayPlusInt {
    struct TensorArray tensors;
    int val;
};

struct TensorArrayPlusBool {
    struct TensorArray tensors;
    bool val;
};

struct TensorArrayPlusVec3d {
    struct TensorArray tensors;
    struct Vec3dWrapper vec3d;
};

struct TensorArrayPlusRect {
    struct TensorArray tensors;
    struct RectWrapper rect;
};

struct RectPlusInt {
    struct RectWrapper rect;
    int val;
};

struct RectPlusBool {
    struct RectWrapper rect;
    bool val;
};

struct ScalarPlusBool {
    struct ScalarWrapper scalar;
    bool val;
};

struct SizePlusInt {
    struct SizeWrapper size;
    int val;
};

struct Point2fPlusInt {
    struct Point2fWrapper point;
    int val;
};

/***************** Other helper structs *****************/

// Arrays

struct StringArray {
    char **data;
    int size;

    StringArray(int size):
        size(size),
        data(static_cast<char **>(malloc(size * sizeof(char*)))) {}

    operator std::vector<cv::String>();
};

struct UCharArray {
    unsigned char *data;
    int size;

    UCharArray() {}
    UCharArray(const std::vector<unsigned char> vec);
};

struct FloatArray {
    float *data;
    int size;

    FloatArray() {}
    FloatArray(const std::vector<float> vec);
};

struct DoubleArray {
    double *data;
    int size;

    DoubleArray() {}
    DoubleArray(const std::vector<double> vec);
};

struct PointArray {
    struct PointWrapper *data;
    int size;

    PointArray() {}
    PointArray(const std::vector<cv::Point> & vec);
    operator std::vector<cv::Point>();
};

struct RectArray {
    struct RectWrapper *data;
    int size;

    RectArray() {}
    RectArray(const std::vector<cv::Rect> & vec);
    operator std::vector<cv::Rect>();
};

struct SizeArray {
    struct SizeWrapper *data;
    int size;

    operator std::vector<cv::Size>();
};

struct TensorPlusRectArray {
    struct TensorWrapper tensor;
    struct RectArray rects;

    TensorPlusRectArray() {}
};

struct TensorArrayPlusRectArray {
    struct TensorArray tensors;
    struct RectArray rects;
};

struct TensorArrayPlusRectArrayPlusFloat {
    struct TensorArray tensors;
    struct RectArray rects;
    float val;
};

struct TensorPlusPointArray {
    struct TensorWrapper tensor;
    struct PointArray points;
};

struct TensorPlusKeyPointArray {
    struct TensorWrapper tensor;
    struct KeyPointArray keypoints;
};

// Arrays of arrays

struct FloatArrayOfArrays {
    float **pointers;
    float *realData;
    int dims;
};

struct PointArrayOfArrays {
    struct PointWrapper **pointers;
    struct PointWrapper *realData;
    int dims;
    int *sizes;
};

/***************** Helper functions *****************/

std::vector<MatT> get_vec_MatT(std::vector<cv::Mat> vec_mat);

std::vector<cv::UMat> get_vec_UMat(std::vector<cv::Mat> vec_mat);

std::vector<cv::Mat> get_vec_Mat(std::vector<cv::UMat> vec_umat);
