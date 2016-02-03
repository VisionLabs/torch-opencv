#include <cudastereo.hpp>

extern "C"
struct StereoBMPtr createStereoBM(int numDisparities, int blockSize)
{
    return rescueObjectFromPtr(cuda::createStereoBM(numDisparities, blockSize));
}

extern "C"
struct TensorWrapper StereoBM_compute(struct cutorchInfo info, struct StereoBMPtr ptr,
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
struct TensorWrapper StereoBeliefPropagation_compute2(struct cutorchInfo info,
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
            width, height, retval.v0, retval.v1, retval.v2);
    return retval;
}

extern "C"
int StereoConstantSpaceBP_getNrPlane(struct StereoConstantSpaceBPPtr ptr)
{
    return ptr->getNrPlane();
}

extern "C"
void StereoConstantSpaceBP_setNrPlane(struct StereoConstantSpaceBPPtr ptr, int val)
{
    ptr->setNrPlane(val);
}

extern "C"
bool StereoConstantSpaceBP_getUseLocalInitDataCost(struct StereoConstantSpaceBPPtr ptr)
{
    return ptr->getUseLocalInitDataCost();
}

extern "C"
void StereoConstantSpaceBP_setUseLocalInitDataCost(struct StereoConstantSpaceBPPtr ptr, bool val)
{
    ptr->setUseLocalInitDataCost(val);
}

extern "C"
struct Vec4iWrapper StereoConstantSpaceBP_estimateRecommendedParams(int width, int height)
{
    Vec4iWrapper retval;
    cuda::StereoConstantSpaceBP::estimateRecommendedParams(
            width, height, retval.v0, retval.v1, retval.v2, retval.v3);
    return retval;
}

extern "C"
struct StereoConstantSpaceBPPtr createStereoConstantSpaceBP(
        int ndisp, int iters, int levels, int nr_plane, int msg_type)
{
    return rescueObjectFromPtr(cuda::createStereoConstantSpaceBP(
            ndisp, iters, levels, nr_plane, msg_type));
}

extern "C"
struct TensorWrapper reprojectImageTo3D(
        struct cutorchInfo info, struct TensorWrapper disp,
        struct TensorWrapper xyzw, struct TensorWrapper Q, int dst_cn)
{
    cuda::GpuMat retval;
    if (!xyzw.isNull()) retval = xyzw.toGpuMat();
    cuda::reprojectImageTo3D(disp.toGpuMat(), retval, Q.toGpuMat(), dst_cn, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper drawColorDisp(
        struct cutorchInfo info, struct TensorWrapper src_disp,
        struct TensorWrapper dst_disp, int ndisp)
{
    cuda::GpuMat retval;
    if (!dst_disp.isNull()) retval = dst_disp.toGpuMat();
    cuda::drawColorDisp(src_disp.toGpuMat(), retval, ndisp, prepareStream(info));
    return TensorWrapper(retval, info.state);
}
