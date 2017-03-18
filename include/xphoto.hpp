#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/xphoto.hpp>

extern "C"
struct TensorWrapper xphoto_bm3dDenoising(
        struct TensorWrapper src, struct TensorWrapper dst,
        float h, int templateWindowSize, int searchWindowSize, int blockMatchingStep1,
        int blockMatchingStep2, int groupSize, int slidingStep, float beta, int normType,
        int step, int transformType);

extern "C"
struct TensorWrapper xphoto_balanceWhite(
        struct TensorWrapper src, struct TensorWrapper dst, int algorithmType);

extern "C"
struct TensorWrapper xphoto_dctDenoising(
        struct TensorWrapper src, struct TensorWrapper dst, double sigma, int psize);

extern "C"
struct TensorWrapper xphoto_inpaint(
        struct TensorWrapper src, struct TensorWrapper mask,
        struct TensorWrapper dst, int algorithmType);