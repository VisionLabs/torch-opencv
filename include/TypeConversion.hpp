#pragma once

#include <opencv2/core/core.hpp>

extern "C" {
#include <TH/TH.h>
}

#include <iostream>
#include <array>

cv::Mat tensorToMat(THFloatTensor *tensor);
void matToTensor(cv::Mat & mat, THFloatTensor * output);