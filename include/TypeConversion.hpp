#include <opencv2/core/core.hpp>

extern "C" {
#include <TH/TH.h>
}

#include <iostream>
#include <array>

struct TensorWrapper {
    void *tensorPtr;
    char tensorType;
};