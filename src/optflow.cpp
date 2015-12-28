#include <optflow.hpp>

extern "C"
struct TensorWrapper calcOpticalFlowSF(
        struct TensorWrapper from, struct TensorWrapper to, struct TensorWrapper flow,
        int layers, int averaging_block_size, int max_flow)
{
    if (flow.isNull()) {
        cv::Mat retval;
        optflow::calcOpticalFlowSF(
                from.toMat(), to.toMat(), retval, layers, averaging_block_size, max_flow);
        return TensorWrapper(retval);
    } else {
        optflow::calcOpticalFlowSF(
                from.toMat(), to.toMat(), flow.toMat(), layers, averaging_block_size, max_flow);
        return flow;
    }
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
    if (flow.isNull()) {
        cv::Mat retval;
        optflow::calcOpticalFlowSF(
                from.toMat(), to.toMat(), retval, layers, averaging_block_size,
                max_flow, sigma_dist, sigma_color, postprocess_window, sigma_dist_fix,
                sigma_color_fix, occ_thr, upscale_averaging_radius,
                upscale_sigma_dist, upscale_sigma_color, speed_up_thr);
        return TensorWrapper(retval);
    } else {
        optflow::calcOpticalFlowSF(
                from.toMat(), to.toMat(), flow.toMat(), layers, averaging_block_size,
                max_flow, sigma_dist, sigma_color, postprocess_window, sigma_dist_fix,
                sigma_color_fix, occ_thr, upscale_averaging_radius,
                upscale_sigma_dist, upscale_sigma_color, speed_up_thr);
        return flow;
    }
}

extern "C"
struct TensorWrapper readOpticalFlow(const char *path)
{
    return TensorWrapper(optflow::readOpticalFlow(path));
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
    std::vector<cv::Mat> retval(2);
    if (!mask.isNull())        retval[0] = mask;
    if (!orientation.isNull()) retval[1] = orientation;
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

    if (segmask.isNull()) {
        cv::Mat segMaskMat;
        motempl::segmentMotion(mhi.toMat(), segMaskMat, rects, timestamp, segThresh);
        new (&retval.tensor) TensorWrapper(segMaskMat);
    } else {
        motempl::segmentMotion(mhi.toMat(), segmask.toMat(), rects, timestamp, segThresh);
        retval.tensor = segmask;
    }

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
    if (flow.isNull()) {
        cv::Mat retval;
        optflow::calcOpticalFlowSparseToDense(
                from.toMat(), to.toMat(), retval, grid_step, k,
                sigma, use_post_proc, fgs_lambda, fgs_sigma);
        return TensorWrapper(retval);
    } else {
        optflow::calcOpticalFlowSparseToDense(
                from.toMat(), to.toMat(), flow.toMat(), grid_step, k,
                sigma, use_post_proc, fgs_lambda, fgs_sigma);
        return flow;
    }
}

extern "C"
struct DenseOpticalFlowPtr createOptFlow_SparseToDense_optflow()
{
    return rescueObjectFromPtr(optflow::createOptFlow_SparseToDense());
}

#endif
