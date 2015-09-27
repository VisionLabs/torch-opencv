#include <Common.hpp>
#include <opencv2/imgproc.hpp>

extern "C" struct TensorWrapper getGaussianKernel(int ksize, double sigma, int ktype);
extern "C" struct MultipleTensorWrapper getDerivKernels(int dx, int dy, int ksize, bool normalize, int ktype);
extern "C" struct TensorWrapper getGaborKernel(
        int ksize_rows, int ksize_cols, double sigma, double theta,
        double lambd, double gamma, double psi, int ktype);