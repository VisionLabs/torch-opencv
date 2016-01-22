#include <cudastereo.hpp>

extern "C"
struct StereoBMPtr createStereoBM(int numDisparities, int blockSize)
{
    return rescueObjectFromPtr(cuda::createStereoBM(numDisparities, blockSize));
}

extern "C"
struct StereoBMPtr StereoBM_compute(struct cutorchInfo info, struct StereoBMPtr ptr,
        struct TensorWrapper left, struct TensorWrapper right, struct TensorWrapper disparity)
{
    cuda::GpuMat retval;
    if (!disparity.isNull()) retval = disparity.toGpuMat();
    ptr->compute(left.toGpuMat(), right.toGpuMat(), retval, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct StereoBeliefPropagationPtr createStereoBeliefPropagation(
        int ndisp, int iters, int levels, int msg_type)
{
    return rescueObjectFromPtr(cuda::createStereoBeliefPropagation(
            ndisp, iters, levels, msg_type));
}

extern "C"
struct TensorWrapper StereoBeliefPropagation_compute(struct cutorchInfo info,
        struct StereoBeliefPropagationPtr ptr, struct TensorWrapper left,
        struct TensorWrapper right, struct TensorWrapper disparity)
{
    cuda::GpuMat retval;
    if (!disparity.isNull()) retval = disparity.toGpuMat();
    ptr->compute(left.toGpuMat(), right.toGpuMat(), retval, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper StereoBeliefPropagation_compute_data(struct cutorchInfo info,
        struct StereoBeliefPropagationPtr ptr, struct TensorWrapper data,
        struct TensorWrapper disparity)
{
    cuda::GpuMat retval;
    if (!disparity.isNull()) retval = disparity.toGpuMat();
    ptr->compute(data.toGpuMat(), retval, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
void StereoBeliefPropagation_setNumIters(struct StereoBeliefPropagationPtr ptr, int val)
{
    ptr->setNumIters(val);
}

extern "C"
int StereoBeliefPropagation_getNumIters(struct StereoBeliefPropagationPtr ptr)
{
    return ptr->getNumIters();
}

extern "C"
void StereoBeliefPropagation_setNumLevels(struct StereoBeliefPropagationPtr ptr, int val)
{
    ptr->setNumLevels(val);
}

extern "C"
int StereoBeliefPropagation_getNumLevels(struct StereoBeliefPropagationPtr ptr)
{
    return ptr->getNumLevels();
}

extern "C"
void StereoBeliefPropagation_setMaxDataTerm(struct StereoBeliefPropagationPtr ptr, double val)
{
    ptr->setMaxDataTerm(val);
}

extern "C"
double StereoBeliefPropagation_getMaxDataTerm(struct StereoBeliefPropagationPtr ptr)
{
    return ptr->getMaxDataTerm();
}

extern "C"
void StereoBeliefPropagation_setDataWeight(struct StereoBeliefPropagationPtr ptr, double val)
{
    ptr->setDataWeight(val);
}

extern "C"
double StereoBeliefPropagation_getDataWeight(struct StereoBeliefPropagationPtr ptr)
{
    return ptr->getDataWeight();
}

extern "C"
void StereoBeliefPropagation_setMaxDiscTerm(struct StereoBeliefPropagationPtr ptr, double val)
{
    ptr->setMaxDiscTerm(val);
}

extern "C"
double StereoBeliefPropagation_getMaxDiscTerm(struct StereoBeliefPropagationPtr ptr)
{
    return ptr->getMaxDiscTerm();
}

extern "C"
void StereoBeliefPropagation_setDiscSingleJump(struct StereoBeliefPropagationPtr ptr, double val)
{
    ptr->setDiscSingleJump(val);
}

extern "C"
double StereoBeliefPropagation_getDiscSingleJump(struct StereoBeliefPropagationPtr ptr)
{
    return ptr->getDiscSingleJump();
}

extern "C"
void StereoBeliefPropagation_setMsgType(struct StereoBeliefPropagationPtr ptr, int val)
{
    ptr->setMsgType(val);
}

extern "C"
int StereoBeliefPropagation_getMsgType(struct StereoBeliefPropagationPtr ptr)
{
    return ptr->getMsgType();
}

extern "C"
struct Vec3iWrapper StereoBeliefPropagation_estimateRecommendedParams(int width, int height)
{
    struct Vec3iWrapper retval;
    cuda::StereoBeliefPropagation::estimateRecommendedParams(
            width, height, &retval.v0, &retval.v1, &retval.v2);
    return retval;
}

extern "C"
struct TensorWrapper reprojectImageTo3D(struct cutorchInfo info,
        struct TensorWrapper disparity, struct TensorWrapper _3dImage,
        struct TensorWrapper Q, bool handleMissingValues, int ddepth)
{
    cuda::GpuMat _3dImage_GpuMat;
    if (!_3dImage.isNull()) _3dImage_GpuMat = _3dImage.toGpuMat();
    cuda::reprojectImageTo3D(
            disparity.toGpuMat(), _3dImage_GpuMat, Q.toGpuMat(),
            handleMissingValues, ddepth, prepareStream(info));
    return TensorWrapper(_3dImage_GpuMat, info.state);
}

