#include <opencv2/core/core.hpp>

extern "C" {
#include <TH/TH.h>
}

#include <iostream>
#include <array>

struct TensorWrapper {
    void *tensorPtr;
    char typeCode;

    TensorWrapper();
    TensorWrapper(cv::Mat & mat);
    TensorWrapper(cv::Mat && mat);
    cv::Mat toMat();
};

struct MultipleTensorWrapper {
    struct TensorWrapper *tensors;
    short size;

    MultipleTensorWrapper(std::vector<cv::Mat> & matList);
    std::vector<cv::Mat> toMat();
};

struct TermCriteriaWrapper {
    int type, maxCount;
    double epsilon;

    cv::TermCriteria toCVTermCriteria();
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