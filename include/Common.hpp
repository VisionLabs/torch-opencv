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
    cv::Mat toMat();
};

struct MultipleTensorWrapper {
    struct TensorWrapper *tensors;
    short size;

    MultipleTensorWrapper(std::vector<cv::Mat> & matList):
            tensors(static_cast<TensorWrapper *>(malloc(matList.size() * sizeof(TensorWrapper)))),
            size(matList.size())
    {
        for (size_t i = 0; i < matList.size(); ++i) {
            // invoke the constructor, memory is already allocated
            new (tensors + i) TensorWrapper(matList[i]);
        }
    }
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