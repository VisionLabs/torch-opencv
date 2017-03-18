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
struct TensorWrapper xphoto_SimpleWB(
        struct TensorWrapper src, struct TensorWrapper dst);

extern "C"
struct TensorWrapper xphoto_GrayworldWB(
        struct TensorWrapper src, struct TensorWrapper dst);

extern "C"
struct TensorWrapper xphoto_LearningBasedWB(
        struct TensorWrapper src, struct TensorWrapper dst);

extern "C"
struct TensorWrapper xphoto_dctDenoising(
        struct TensorWrapper src, struct TensorWrapper dst, double sigma, int psize);

extern "C"
struct TensorWrapper xphoto_inpaint(
        struct TensorWrapper src, struct TensorWrapper mask,
        struct TensorWrapper dst, int algorithmType);