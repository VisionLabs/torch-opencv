#include <Common.hpp>
#include <opencv2/core.hpp>

extern "C" {

int getNumThreads();

void setNumThreads(int nthreads);

struct TensorWrapper copyMakeBorder(struct TensorWrapper src, struct TensorWrapper dst, int top, 
                                    int bottom, int left, int right, int borderType,
                                    struct ScalarWrapper value);
}