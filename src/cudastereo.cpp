#include <cudastereo.hpp>

extern "C"
struct StereoBMPtr createStereoBMCuda(int numDisparities, int blockSize)
{
    return rescueObjectFromPtr(cuda::createStereoBM(numDisparities, blockSize));
}

extern "C"
struct TensorWrapper StereoBM_computeCuda(struct cutorchInfo info, struct StereoBMPtr ptr,
        struct TensorWrapper left, struct TensorWrapper right, struct TensorWrapper disparity)
{
    cuda::GpuMat retval = disparity.toGpuMat();
    ptr->compute(left.toGpuMat(), right.toGpuMat(), retval, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct StereoBeliefPropagationPtr createStereoBeliefPropagationCuda(
        int ndisp, int iters, int levels, int msg_type)
{
    return rescueObjectFromPtr(cuda::createStereoBeliefPropagation(
            ndisp, iters, levels, msg_type));
}

extern "C"
struct TensorWrapper StereoBeliefPropagation_computeCuda(struct cutorchInfo info,
        struct StereoBeliefPropagationPtr ptr, struct TensorWrapper left,
        struct TensorWrapper right, struct TensorWrapper disparity)
{
    cuda::GpuMat retval = disparity.toGpuMat();
    ptr->compute(left.toGpuMat(), right.toGpuMat(), retval, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper StereoBeliefPropagation_compute2Cuda(struct cutorchInfo info,
        struct StereoBeliefPropagationPtr ptr, struct TensorWrapper data,
        struct TensorWrapper disparity)
{
    cuda::GpuMat retval = disparity.toGpuMat();
    ptr->compute(data.toGpuMat(), retval, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
void StereoBeliefPropagation_setNumItersCuda(struct StereoBeliefPropagationPtr ptr, int val)
{
    ptr->setNumIters(val);
}

extern "C"
int StereoBeliefPropagation_getNumItersCuda(struct StereoBeliefPropagationPtr ptr)
{
    return ptr->getNumIters();
}

extern "C"
void StereoBeliefPropagation_setNumLevelsCuda(struct StereoBeliefPropagationPtr ptr, int val)
{
    ptr->setNumLevels(val);
}

extern "C"
int StereoBeliefPropagation_getNumLevelsCuda(struct StereoBeliefPropagationPtr ptr)
{
    return ptr->getNumLevels();
}

extern "C"
void StereoBeliefPropagation_setMaxDataTermCuda(struct StereoBeliefPropagationPtr ptr, double val)
{
    ptr->setMaxDataTerm(val);
}

extern "C"
double StereoBeliefPropagation_getMaxDataTermCuda(struct StereoBeliefPropagationPtr ptr)
{
    return ptr->getMaxDataTerm();
}

extern "C"
void StereoBeliefPropagation_setDataWeightCuda(struct StereoBeliefPropagationPtr ptr, double val)
{
    ptr->setDataWeight(val);
}

extern "C"
double StereoBeliefPropagation_getDataWeightCuda(struct StereoBeliefPropagationPtr ptr)
{
    return ptr->getDataWeight();
}

extern "C"
void StereoBeliefPropagation_setMaxDiscTermCuda(struct StereoBeliefPropagationPtr ptr, double val)
{
    ptr->setMaxDiscTerm(val);
}

extern "C"
double StereoBeliefPropagation_getMaxDiscTermCuda(struct StereoBeliefPropagationPtr ptr)
{
    return ptr->getMaxDiscTerm();
}

extern "C"
void StereoBeliefPropagation_setDiscSingleJumpCuda(struct StereoBeliefPropagationPtr ptr, double val)
{
    ptr->setDiscSingleJump(val);
}

extern "C"
double StereoBeliefPropagation_getDiscSingleJumpCuda(struct StereoBeliefPropagationPtr ptr)
{
    return ptr->getDiscSingleJump();
}

extern "C"
void StereoBeliefPropagation_setMsgTypeCuda(struct StereoBeliefPropagationPtr ptr, int val)
{
    ptr->setMsgType(val);
}

extern "C"
int StereoBeliefPropagation_getMsgTypeCuda(struct StereoBeliefPropagationPtr ptr)
{
    return ptr->getMsgType();
}

extern "C"
struct Vec3iWrapper StereoBeliefPropagation_estimateRecommendedParamsCuda(int width, int height)
{
    struct Vec3iWrapper retval;
    cuda::StereoBeliefPropagation::estimateRecommendedParams(
            width, height, retval.v0, retval.v1, retval.v2);
    return retval;
}

extern "C"
int StereoConstantSpaceBP_getNrPlaneCuda(struct StereoConstantSpaceBPPtr ptr)
{
    return ptr->getNrPlane();
}

extern "C"
void StereoConstantSpaceBP_setNrPlaneCuda(struct StereoConstantSpaceBPPtr ptr, int val)
{
    ptr->setNrPlane(val);
}

extern "C"
bool StereoConstantSpaceBP_getUseLocalInitDataCostCuda(struct StereoConstantSpaceBPPtr ptr)
{
    return ptr->getUseLocalInitDataCost();
}

extern "C"
void StereoConstantSpaceBP_setUseLocalInitDataCostCuda(struct StereoConstantSpaceBPPtr ptr, bool val)
{
    ptr->setUseLocalInitDataCost(val);
}

extern "C"
struct Vec4iWrapper StereoConstantSpaceBP_estimateRecommendedParamsCuda(int width, int height)
{
    Vec4iWrapper retval;
    cuda::StereoConstantSpaceBP::estimateRecommendedParams(
            width, height, retval.v0, retval.v1, retval.v2, retval.v3);
    return retval;
}

extern "C"
struct StereoConstantSpaceBPPtr createStereoConstantSpaceBPCuda(
        int ndisp, int iters, int levels, int nr_plane, int msg_type)
{
    return rescueObjectFromPtr(cuda::createStereoConstantSpaceBP(
            ndisp, iters, levels, nr_plane, msg_type));
}

extern "C"
struct TensorWrapper reprojectImageTo3DCuda(
        struct cutorchInfo info, struct TensorWrapper disp,
        struct TensorWrapper xyzw, struct TensorWrapper Q, int dst_cn)
{
    cuda::GpuMat retval = xyzw.toGpuMat();
    cuda::reprojectImageTo3D(disp.toGpuMat(), retval, Q.toGpuMat(), dst_cn, prepareStream(info));
    return TensorWrapper(retval, info.state);
}

extern "C"
struct TensorWrapper drawColorDispCuda(
        struct cutorchInfo info, struct TensorWrapper src_disp,
        struct TensorWrapper dst_disp, int ndisp)
{
    cuda::GpuMat retval = dst_disp.toGpuMat();
    cuda::drawColorDisp(src_disp.toGpuMat(), retval, ndisp, prepareStream(info));
    return TensorWrapper(retval, info.state);
}
