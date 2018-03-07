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

} // extern "C"