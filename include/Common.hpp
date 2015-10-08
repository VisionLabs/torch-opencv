#include <opencv2/core/core.hpp>

extern "C" {
#include <TH/TH.h>
}

#include <iostream>
#include <array>

extern "C" int getIntMax() { return INT_MAX; }

/***************** Tensor <=> Mat conversion *****************/

#define TO_MAT_OR_NOARRAY(mat) (mat.isNull() ? cv::noArray() : mat.toMat())

struct TensorWrapper {
    void *tensorPtr;
    char typeCode;

    TensorWrapper();
    TensorWrapper(cv::Mat & mat);
    TensorWrapper(cv::Mat && mat);
    operator cv::Mat();
    // synonym for operator cv::Mat()
    inline cv::Mat toMat() { return *this; }

    inline bool isNull() { return tensorPtr == nullptr; }
};

struct MultipleTensorWrapper {
    struct TensorWrapper *tensors;
    short size;

    MultipleTensorWrapper();
    MultipleTensorWrapper(std::vector<cv::Mat> & matList);
    operator std::vector<cv::Mat>();
    // synonym for operator std::vector<cv::Mat>()
    inline std::vector<cv::Mat> toMatList() { return *this; }

    inline bool isNull() { return tensors == nullptr; }
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
        default: ; // TODO: raise an error
    }
}

/***************** Wrappers for small OpenCV classes *****************/

struct SizeWrapper {
    int width, height;

    inline operator cv::Size() { return cv::Size(width, height); }
};

struct Size2fWrapper {
    float width, height;

    inline operator cv::Size2f() { return cv::Size2f(width, height); }
};

struct TermCriteriaWrapper {
    int type, maxCount;
    double epsilon;

    inline operator cv::TermCriteria() { return cv::TermCriteria(type, maxCount, epsilon); }
    inline cv::TermCriteria orDefault(cv::TermCriteria defaultVal) {
        return (this->type == 0 ? defaultVal : *this);
    }
};

struct ScalarWrapper {
    double v0, v1, v2, v3;

    inline operator cv::Scalar() { return cv::Scalar(v0, v1, v2, v3); }
    inline cv::Scalar orDefault(cv::Scalar defaultVal) {
        return (isnan(this->v0) ? defaultVal : *this);
    }
};

struct Vec3dWrapper {
    double v0, v1, v2;
};

struct RectWrapper {
    int x, y, width, height;

    inline operator cv::Rect() { return cv::Rect(x, y, width, height); }
    RectWrapper & operator=(cv::Rect & other);
};

struct PointWrapper {
    int x, y;

    inline operator cv::Point() { return cv::Point(x, y); }
};

struct Point2fWrapper {
    float x, y;

    inline operator cv::Point2f() { return cv::Point2f(x, y); }
};

struct RotatedRectWrapper {
    struct Point2fWrapper center;
    struct Size2fWrapper size;
    float angle;

    inline operator cv::RotatedRect() { return cv::RotatedRect(center, size, angle); }
};

struct MomentsWrapper {
    double m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
    double mu20, mu11, mu02, mu30, mu21, mu12, mu03;
    double nu20, nu11, nu02, nu30, nu21, nu12, nu03;
    
    inline operator cv::Moments() {
        return cv::Moments(m00, m10, m01, m20, m11, m02, m30, m21, m12, m03);
    }
};

/***************** Helper wrappers for [OpenCV class + some primitive] *****************/

struct TWPlusDouble {
    struct TensorWrapper tensor;
    double val;
};

struct MTWPlusFloat {
    struct MultipleTensorWrapper tensors;
    float val;
};

struct RectPlusInt {
    struct RectWrapper rect;
    int val;
};

/***************** Other helper structs *****************/

struct IntArray {
    int *data;
    int size;
};

struct FloatArray {
    float *data;
    int size;
};

struct PointArray {
    struct PointWrapper *data;
    int size;
};

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
