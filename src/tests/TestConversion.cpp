#include <TypeConversion.hpp>

extern "C" {
#include <TH/TH.h>
}

extern "C"
TensorWrapper test_mat_to_tensor() {
    cv::Mat outputMat = cv::Mat::ones(3, 3, CV_8SC1) * 7.;
    std::cout << "Sending Mat to Torch" << std::endl;
    return matToTensor(outputMat);
}

extern "C"
void test_tensor_to_mat(TensorWrapper tensor) {
    cv::Mat temp = tensorToMat(tensor);
    std::cout << "This is a " << temp.channels() <<
    "-channel Mat of type " << typeStr(temp) << std::endl;
    std::cout << temp * 10. << std::endl;
}