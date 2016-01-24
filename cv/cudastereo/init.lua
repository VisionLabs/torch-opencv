local cv = require 'cv._env'
require 'cutorch'
require 'cv.calib3d'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or require 'cv._env_cuda'

local ffi = require 'ffi'

ffi.cdef[[

]]

local C = ffi.load(cv.libPath('cudafilters'))

require 'cv.Classes'
local Classes = ffi.load(cv.libPath('Classes'))

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

return cv.cuda
