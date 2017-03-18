#include <xphoto.hpp>

extern "C"
struct TensorWrapper xphoto_balanceWhite(
        struct TensorWrapper src, struct TensorWrapper dst, int algorithmType)
{   
    cv::Ptr<cv::xphoto::WhiteBalancer> wb;
    MatT dst_mat = dst.toMatT();
    if (algorithmType == 1)
        wb = cv::xphoto::createSimpleWB();
    else if (algorithmType == 2)
        wb = cv::xphoto::createGrayworldWB();
    else if (algorithmType == 3)
        wb = cv::xphoto::createLearningBasedWB();
    else 
    {
        printf("Unknown algorithm type!");
    }
    wb->balanceWhite(src.toMat(), dst_mat.mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper xphoto_bm3dDenoising(
        struct TensorWrapper src, struct TensorWrapper dst,
        float h, int templateWindowSize, int searchWindowSize, int blockMatchingStep1,
        int blockMatchingStep2, int groupSize, int slidingStep, float beta, int normType,
        int step, int transformType)
{   

    MatT dst_mat = dst.toMatT();
    cv::xphoto::bm3dDenoising(src.toMat(), dst_mat.mat, h, templateWindowSize, searchWindowSize,
        blockMatchingStep1, blockMatchingStep2, groupSize, slidingStep, beta, normType, step, transformType);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper xphoto_dctDenoising(
        struct TensorWrapper src, struct TensorWrapper dst, double sigma, int psize)
{
    MatT dst_mat = dst.toMatT();
    cv::xphoto::dctDenoising(src.toMat(), dst_mat.mat, sigma, psize);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper xphoto_inpaint(
        struct TensorWrapper src, struct TensorWrapper mask,
        struct TensorWrapper dst, int algorithmType)
{
    MatT dst_mat = dst.toMatT();
    cv::xphoto::inpaint(src.toMat(), mask.toMat(), dst_mat.mat, algorithmType);
    return TensorWrapper(dst_mat);
}
