local cv = require 'cv._env'
require 'cutorch'
require 'cv.calib3d'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or require 'cv._env_cuda'

local ffi = require 'ffi'

ffi.cdef[[
struct PtrWrapper createStereoBM(int numDisparities, int blockSize);

struct PtrWrapper StereoBM_compute(struct cutorchInfo info, struct PtrWrapper ptr,
        struct TensorWrapper left, struct TensorWrapper right, struct TensorWrapper disparity);

struct PtrWrapper createStereoBeliefPropagation(
        int ndisp, int iters, int levels, int msg_type);

struct TensorWrapper StereoBeliefPropagation_compute(struct cutorchInfo info,
        struct PtrWrapper ptr, struct TensorWrapper left,
        struct TensorWrapper right, struct TensorWrapper disparity);

struct TensorWrapper StereoBeliefPropagation_compute2(struct cutorchInfo info,
        struct PtrWrapper ptr, struct TensorWrapper data,
        struct TensorWrapper disparity);

void StereoBeliefPropagation_setNumIters(struct PtrWrapper ptr, int val);

int StereoBeliefPropagation_getNumIters(struct PtrWrapper ptr);

void StereoBeliefPropagation_setNumLevels(struct PtrWrapper ptr, int val);

int StereoBeliefPropagation_getNumLevels(struct PtrWrapper ptr);

void StereoBeliefPropagation_setMaxDataTerm(struct PtrWrapper ptr, double val);

double StereoBeliefPropagation_getMaxDataTerm(struct PtrWrapper ptr);

void StereoBeliefPropagation_setDataWeight(struct PtrWrapper ptr, double val);

double StereoBeliefPropagation_getDataWeight(struct PtrWrapper ptr);

void StereoBeliefPropagation_setMaxDiscTerm(struct PtrWrapper ptr, double val);

double StereoBeliefPropagation_getMaxDiscTerm(struct PtrWrapper ptr);

void StereoBeliefPropagation_setDiscSingleJump(struct PtrWrapper ptr, double val);

double StereoBeliefPropagation_getDiscSingleJump(struct PtrWrapper ptr);

void StereoBeliefPropagation_setMsgType(struct PtrWrapper ptr, int val);

int StereoBeliefPropagation_getMsgType(struct PtrWrapper ptr);

struct Vec3iWrapper StereoBeliefPropagation_estimateRecommendedParams(int width, int height);

int StereoConstantSpaceBP_getNrPlane(struct PtrWrapper ptr);

void StereoConstantSpaceBP_setNrPlane(struct PtrWrapper ptr, int val);

bool StereoConstantSpaceBP_getUseLocalInitDataCost(struct PtrWrapper ptr);

void StereoConstantSpaceBP_setUseLocalInitDataCost(struct PtrWrapper ptr, bool val);

struct Vec4iWrapper StereoConstantSpaceBP_estimateRecommendedParams(int width, int height);

struct PtrWrapper createStereoConstantSpaceBP(
        int ndisp, int iters, int levels, int nr_plane, int msg_type);

struct TensorWrapper reprojectImageTo3D(
        struct cutorchInfo info, struct TensorWrapper disp,
        struct TensorWrapper xyzw, struct TensorWrapper Q, int dst_cn);

struct TensorWrapper drawColorDisp(
        struct cutorchInfo info, struct TensorWrapper src_disp,
        struct TensorWrapper dst_disp, int ndisp);
]]

local C = ffi.load(cv.libPath('cudastereo'))

require 'cv.Classes'
local Classes = ffi.load(cv.libPath('Classes'))

-- StereoBM

do
	local StereoBM = torch.class('cuda.StereoBM', 'cv.StereoBM', cv.cuda)

	function StereoBM:compute(t)
		local argRules = {
			{"left", required = true, operator = cv.wrap_tensor},
			{"right", required = true, operator = cv.wrap_tensor},
			{"disparity", default = nil, operator = cv.wrap_tensor}
		}
		return cv.unwrap_tensors(C.StereoBM_compute(cv.cuda._info(), self.ptr,
			cv.argcheck(t, argRules)))
	end
end

function cv.cuda.createStereoBM(t)
	local argRules = {
		{"numDisparities", default = 64},
		{"blockSize", default = 19}
	}
	local retval = torch.factory('cuda.StereoBM')()
	retval.ptr = C.createStereoBM(cv.argcheck(t, argRules))
	return retval
end

-- StereoBeliefPropagation

do
	local StereoBeliefPropagation = 
		torch.class('cuda.StereoBeliefPropagation', 'cv.StereoMatcher', cv.cuda)

	function StereoBeliefPropagation:compute(t)
		local argRules = {
			{"left", required = true, operator = cv.wrap_tensor},
			{"right", required = true, operator = cv.wrap_tensor},
			{"disparity", default = nil, operator = cv.wrap_tensor}
		}
		return cv.unwrap_tensors(C.StereoBeliefPropagation_compute(
			cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
	end

	function StereoBeliefPropagation:compute2(t)
		local argRules = {
			{"data", required = true, operator = cv.wrap_tensor},
			{"disparity", default = nil, operator = cv.wrap_tensor}
		}
		return cv.unwrap_tensors(C.StereoBeliefPropagation_compute2(
			cv.cuda._info(), self.ptr, cv.argcheck(t, argRules)))
	end

	function StereoBeliefPropagation:setNumIters(t)
	    local argRules = {
	        {"val", required = true}
	    }
	    local val = cv.argcheck(t, argRules)
	    
	    C.StereoBeliefPropagation_setNumIters(self.ptr, val)
	end

	function StereoBeliefPropagation:getNumIters()
	    return C.StereoBeliefPropagation_getNumIters(self.ptr)
	end

	function StereoBeliefPropagation:setNumLevels(t)
	    local argRules = {
	        {"val", required = true}
	    }
	    local val = cv.argcheck(t, argRules)
	    
	    C.StereoBeliefPropagation_setNumLevels(self.ptr, val)
	end

	function StereoBeliefPropagation:getNumLevels()
	    return C.StereoBeliefPropagation_getNumLevels(self.ptr)
	end

	function StereoBeliefPropagation:setMaxDataTerm(t)
	    local argRules = {
	        {"val", required = true}
	    }
	    local val = cv.argcheck(t, argRules)
	    
	    C.StereoBeliefPropagation_setMaxDataTerm(self.ptr, val)
	end

	function StereoBeliefPropagation:getMaxDataTerm()
	    return C.StereoBeliefPropagation_getMaxDataTerm(self.ptr)
	end

	function StereoBeliefPropagation:setDataWeight(t)
	    local argRules = {
	        {"val", required = true}
	    }
	    local val = cv.argcheck(t, argRules)
	    
	    C.StereoBeliefPropagation_setDataWeight(self.ptr, val)
	end

	function StereoBeliefPropagation:getDataWeight()
	    return C.StereoBeliefPropagation_getDataWeight(self.ptr)
	end

	function StereoBeliefPropagation:setMaxDiscTerm(t)
	    local argRules = {
	        {"val", required = true}
	    }
	    local val = cv.argcheck(t, argRules)
	    
	    C.StereoBeliefPropagation_setMaxDiscTerm(self.ptr, val)
	end

	function StereoBeliefPropagation:getMaxDiscTerm()
	    return C.StereoBeliefPropagation_getMaxDiscTerm(self.ptr)
	end

	function StereoBeliefPropagation:setDiscSingleJump(t)
	    local argRules = {
	        {"val", required = true}
	    }
	    local val = cv.argcheck(t, argRules)
	    
	    C.StereoBeliefPropagation_setDiscSingleJump(self.ptr, val)
	end

	function StereoBeliefPropagation:getDiscSingleJump()
	    return C.StereoBeliefPropagation_getDiscSingleJump(self.ptr)
	end

	function StereoBeliefPropagation:setMsgType(t)
	    local argRules = {
	        {"val", required = true}
	    }
	    local val = cv.argcheck(t, argRules)
	    
	    C.StereoBeliefPropagation_setMsgType(self.ptr, val)
	end

	function StereoBeliefPropagation:getMsgType()
	    return C.StereoBeliefPropagation_getMsgType(self.ptr)
	end

	function StereoBeliefPropagation.estimateRecommendedParams(t)
		local argRules = {
			{"width", required = true},
			{"height", required = true}
		}
		local retval = 
			C.StereoBeliefPropagation_estimateRecommendedParams(cv.argcheck(t, argRules))
		return retval.v0, retval.v1, retval.v2
	end
end

function cv.cuda.createStereoBeliefPropagation(t)
	local argRules = {
		{"ndisp", default = 64},
		{"iters", default = 5},
		{"levels", default = 5},
		{"msg_type", default = cv.CV_32F}
	}
	local retval = torch.factory('cuda.StereoBeliefPropagation')()
	retval.ptr = C.createStereoBeliefPropagation(cv.argcheck(t, argRules))
	return retval
end

do
	local StereoConstantSpaceBP = 
		torch.class('cuda.StereoConstantSpaceBP', 'cuda.StereoBeliefPropagation', cv.cuda)

	function StereoConstantSpaceBP:getNrPlane(t)
		return C.StereoConstantSpaceBP_getNrPlane(self.ptr)
	end

	function StereoConstantSpaceBP_setNrPlane(t)
		local argRules = {
			{"val", required = true}
		}
		C.StereoConstantSpaceBP_setNrPlane(self.ptr, cv.argcheck(t, argRules))
	end

	function StereoConstantSpaceBP:getUseLocalInitDataCost(t)
		return C.StereoConstantSpaceBP_getUseLocalInitDataCost(self.ptr)
	end

	function StereoConstantSpaceBP_setUseLocalInitDataCost(t)
		local argRules = {
			{"val", required = true}
		}
		C.StereoConstantSpaceBP_setUseLocalInitDataCost(self.ptr, cv.argcheck(t, argRules))
	end

	function StereoConstantSpaceBP.estimateRecommendedParams(t)
		local argRules = {
			{"width", required = true},
			{"height", required = true}
		}
		local retval = 
			C.StereoConstantSpaceBP_estimateRecommendedParams(cv.argcheck(t, argRules))
		return retval.v0, retval.v1, retval.v2, retval.v3
	end
end

function cv.cuda.createStereoConstantSpaceBP(t)
	local argRules = {
		{"ndisp", default = 128},
		{"iters", default = 8},
		{"levels", default = 4},
		{"nr_plane", default = 4},
		{"msg_type", default = cv.CV_32F}
	}
	local retval = torch.factory('cuda.StereoConstantSpaceBP')()
	retval.ptr = C.createStereoConstantSpaceBP(cv.argcheck(t, argRules))
	return retval
end

function cv.cuda.reprojectImageTo3D(t)
	local argRules = {
		{"disp", required = true, operator = cv.wrap_tensor},
		{"xyzw", default = nil, operator = cv.wrap_tensor},
		{"Q", required = true, operator = cv.wrap_tensor},
		{"dst_cn", default = 4}
	}
	return cv.unwrap_tensors(C.reprojectImageTo3D(cv.cuda._info(), cv.argcheck(t, argRules)))
end

function cv.cuda.drawColorDisp(t)
	local argRules = {
		{"src_disp", required = true, operator = cv.wrap_tensor},
		{"dst_disp", default = nil, operator = cv.wrap_tensor},
		{"ndisp", required = true}
	}
	return cv.unwrap_tensors(C.drawColorDisp(cv.cuda._info(), cv.argcheck(t, argRules)))
end

return cv.cuda
