#include <optflow.hpp>

extern "C"
struct TensorWrapper calcOpticalFlowSF(
        struct TensorWrapper from, struct TensorWrapper to, struct TensorWrapper flow,
        int layers, int averaging_block_size, int max_flow)
{
    MatT flow_mat;
    if(!flow.isNull()) flow_mat = flow.toMatT();
    optflow::calcOpticalFlowSF(
                from.toMat(), to.toMat(), flow_mat, layers, averaging_block_size, max_flow);
    return TensorWrapper(flow_mat);
}

extern "C"
struct TensorWrapper calcOpticalFlowSF_expanded(
        struct TensorWrapper from, struct TensorWrapper to, struct TensorWrapper flow,
        int layers, int averaging_block_size, int max_flow,
        double sigma_dist, double sigma_color, int postprocess_window,
        double sigma_dist_fix, double sigma_color_fix, double occ_thr,
        int upscale_averaging_radius, double upscale_sigma_dist,
        double upscale_sigma_color, double speed_up_thr)
{
    MatT flow_mat;
    if(!flow.isNull()) flow_mat = flow.toMatT();
    optflow::calcOpticalFlowSF(
                from.toMat(), to.toMat(), flow_mat, layers, averaging_block_size,
                max_flow, sigma_dist, sigma_color, postprocess_window, sigma_dist_fix,
                sigma_color_fix, occ_thr, upscale_averaging_radius,
                upscale_sigma_dist, upscale_sigma_color, speed_up_thr);
    return TensorWrapper(flow_mat);
}

extern "C"
struct TensorWrapper readOpticalFlow(const char *path)
{
    return TensorWrapper(MatT(optflow::readOpticalFlow(path)));
}

extern "C"
bool writeOpticalFlow(const char *path, struct TensorWrapper flow)
{
    return optflow::writeOpticalFlow(path, flow.toMat());
}

extern "C"
void updateMotionHistory(
        struct TensorWrapper silhouette, struct TensorWrapper mhi,
        double timestamp, double duration)
{
    motempl::updateMotionHistory(silhouette.toMat(), mhi.toMat(), timestamp, duration);
}

extern "C"
struct TensorArray calcMotionGradient(
        struct TensorWrapper mhi, struct TensorWrapper mask, struct TensorWrapper orientation,
        double delta1, double delta2, int apertureSize)
{
    std::vector<MatT> retval(2);
    if (!mask.isNull()) retval[0] = mask.toMatT();
    if (!orientation.isNull()) retval[1] = orientation.toMatT();
    motempl::calcMotionGradient(mhi.toMat(), retval[0], retval[1], delta1, delta2, apertureSize);
    return TensorArray(retval);
}

extern "C"
double calcGlobalOrientation(
        struct TensorWrapper orientation, struct TensorWrapper mask,
        struct TensorWrapper mhi, double timestamp, double duration)
{
    return motempl::calcGlobalOrientation(
            orientation.toMat(), mask.toMat(), mhi.toMat(), timestamp, duration);
}

extern "C"
struct TensorPlusRectArray segmentMotion(
        struct TensorWrapper mhi, struct TensorWrapper segmask,
        double timestamp, double segThresh)
{
    struct TensorPlusRectArray retval;
    std::vector<cv::Rect> rects;

    MatT segmask_mat;
    if(!segmask.isNull()) segmask_mat = segmask.toMatT();
    motempl::segmentMotion(mhi.toMat(), segmask_mat, rects, timestamp, segThresh);
    new(&retval.tensor) TensorWrapper(segmask_mat);
    new (&retval.rects) RectArray(rects);
    return retval;
}

// DenseOpticalFlowPtr is from video.hpp

extern "C"
struct DenseOpticalFlowPtr createOptFlow_DeepFlow_optflow()
{
    return rescueObjectFromPtr(optflow::createOptFlow_DeepFlow());
}

extern "C"
struct DenseOpticalFlowPtr createOptFlow_SimpleFlow_optflow()
{
    return rescueObjectFromPtr(optflow::createOptFlow_SimpleFlow());
}

extern "C"
struct DenseOpticalFlowPtr createOptFlow_Farneback_optflow()
{
    return rescueObjectFromPtr(optflow::createOptFlow_Farneback());
}

#if CV_MAJOR_VERSION >= 3 && CV_MINOR_VERSION >= 1

extern "C"
struct TensorWrapper calcOpticalFlowSparseToDense(
        struct TensorWrapper from, struct TensorWrapper to, struct TensorWrapper flow,
        int grid_step, int k, float sigma, bool use_post_proc, float fgs_lambda, float fgs_sigma)
{
    MatT flow_mat;
    if(!flow.isNull()) flow_mat = flow.toMatT();
    optflow::calcOpticalFlowSparseToDense(
                from.toMat(), to.toMat(), flow_mat, grid_step, k,
                sigma, use_post_proc, fgs_lambda, fgs_sigma);
    return TensorWrapper(flow_mat);
}

extern "C"
struct DenseOpticalFlowPtr createOptFlow_SparseToDense_optflow()
{
    return rescueObjectFromPtr(optflow::createOptFlow_SparseToDense());
}

#endif
