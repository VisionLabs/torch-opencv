#include <opencv2/core/core.hpp>

extern "C" {
#include <TH/TH.h>
}

#include <iostream>
#include <array>

/***************** Tensor <=> Mat conversion *****************/

#define TO_MAT_OR_NOARRAY(mat) (mat.isNull() ? cv::noArray() : mat.toMat())

struct TensorWrapper {
    void *tensorPtr;
    char typeCode;

    TensorWrapper();
    TensorWrapper(cv::Mat & mat);
    TensorWrapper(cv::Mat && mat);
    cv::Mat toMat();

    inline bool isNull() { return tensorPtr == nullptr; }
};

struct MultipleTensorWrapper {
    struct TensorWrapper *tensors;
    short size;

    MultipleTensorWrapper();
    MultipleTensorWrapper(std::vector<cv::Mat> & matList);
    std::vector<cv::Mat> toMatList();

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

struct TermCriteriaWrapper {
    int type, maxCount;
    double epsilon;

    inline cv::TermCriteria toCV() {
        return cv::TermCriteria(type, maxCount, epsilon);
    }
    inline cv::TermCriteria toCVorDefault(cv::TermCriteria defaultVal) {
        return (this->type == -1 ? defaultVal : this->toCV());
    }
};

struct ScalarWrapper {
    double v0, v1, v2, v3;

    inline cv::Scalar toCV() {
        return cv::Scalar(v0, v1, v2, v3);
    }
    inline cv::Scalar toCVorDefault(cv::Scalar defaultVal) {
        return (isnan(this->v0) ? defaultVal : this->toCV());
    }
};

struct Vec3dWrapper {
    double v0, v1, v2;
};

/***************** Helper wrappers for [OpenCV class + some primitive] *****************/

struct TWPlusDouble {
    TensorWrapper tensor;
    double val;
};

struct MTWPlusFloat {
    MultipleTensorWrapper tensors;
    float val;
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

struct FloatArrayOfArrays {
    float **pointers;
    float *realData;
    int dims;
};