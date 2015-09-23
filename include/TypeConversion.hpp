#include <opencv2/core/core.hpp>

extern "C" {
#include <TH/TH.h>
}

#include <iostream>
#include <array>

struct TensorWrapper {
    void *tensorPtr;
    char typeCode;
};

TensorWrapper matToTensor(cv::Mat & mat);
cv::Mat tensorToMat(TensorWrapper tensor);

inline
std::string typeStr(cv::Mat & mat) {
    switch (mat.depth()) {
        case CV_8U:  return "Byte";
        case CV_8S:  return "Char";
        case CV_16S: return "Short";
        case CV_32S: return "Int";
        case CV_32F: return "Float";
        case CV_64F: return "Double";
    }
}