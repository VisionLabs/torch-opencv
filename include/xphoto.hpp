#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/xphoto.hpp>

extern "C"
struct TensorWrapper xphoto_autowbGrayworld(
        struct TensorWrapper src, struct TensorWrapper dst, float thresh);

extern "C"
struct TensorWrapper xphoto_balanceWhite(
        struct TensorWrapper src, struct TensorWrapper dst, int algorithmType,
        float inputMin, float inputMax, float outputMin, float outputMax);

extern "C"
struct TensorWrapper xphoto_dctDenoising(
        struct TensorWrapper src, struct TensorWrapper dst, double sigma, int psize);

extern "C"
struct TensorWrapper xphoto_inpaint(
        struct TensorWrapper src, struct TensorWrapper mask,
        struct TensorWrapper dst, int algorithmType);