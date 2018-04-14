#include <core.hpp>


extern "C" {

int getNumThreads()
{
    return cv::getNumThreads();
}

void setNumThreads(int nthreads)
{
    cv::setNumThreads(nthreads);
}

struct TensorWrapper copyMakeBorder(struct TensorWrapper src, struct TensorWrapper dst, int top, 
                                    int bottom, int left, int right, int borderType,
                                    struct ScalarWrapper value)
{
    MatT dstMat = dst.toMatT();
    cv::copyMakeBorder(src.toMat(), dstMat, top, bottom, left, right, borderType, value);
    return TensorWrapper(dstMat);
}

} // extern "C"