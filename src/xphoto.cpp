#include <xphoto.hpp>

extern "C"
struct TensorWrapper xphoto_autowbGrayworld(
        struct TensorWrapper src, struct TensorWrapper dst, float thresh)
{
    MatT dst_mat;
    if(!dst.isNull()) dst_mat = dst.toMatT();
    cv::xphoto::autowbGrayworld(src.toMat(), dst_mat, thresh);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper xphoto_balanceWhite(
        struct TensorWrapper src, struct TensorWrapper dst, int algorithmType,
        float inputMin, float inputMax, float outputMin, float outputMax)
{
    MatT dst_mat;
    if(!dst.isNull()) dst_mat = dst.toMatT();
    cv::xphoto::balanceWhite(src.toMat(), dst_mat.mat, algorithmType, inputMin, inputMax, outputMin, outputMax);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper xphoto_dctDenoising(
        struct TensorWrapper src, struct TensorWrapper dst, double sigma, int psize)
{
    MatT dst_mat;
    if(!dst.isNull()) dst_mat = dst.toMatT();
    cv::xphoto::balanceWhite(src.toMat(), dst_mat.mat, sigma, psize);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper xphoto_inpaint(
        struct TensorWrapper src, struct TensorWrapper mask,
        struct TensorWrapper dst, int algorithmType)
{
    MatT dst_mat;
    if(!dst.isNull()) dst_mat = dst.toMatT();
    cv::xphoto::inpaint(src.toMat(), mask.toMat(), dst_mat.mat, algorithmType);
    return TensorWrapper(dst_mat);
}