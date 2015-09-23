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