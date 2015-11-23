local cv = require 'cv._env'

local ffi = require 'ffi'

local optflow = {}

ffi.cdef[[

]]

local C = ffi.load(cv.libPath('optflow'))

function optflow.calcOpticalFlowSF(t)
    if t[7] or t.sigma_dist then
        local argRules = {
            {"from", required = true},
            {"to", required = true},
            {"flow", default = nil},
            {"layers", required = true},
            {"averaging_block_size", required = true},
            {"max_flow", required = true},
            {"sigma_dist", required = true},
            {"sigma_color", required = true},
            {"postprocess_window", required = true},
            {"sigma_dist_fix", required = true},
            {"sigma_color_fix", required = true},
            {"occ_thr", required = true},
            {"upscale_averaging_radius", required = true},
            {"upscale_sigma_dist", required = true},
            {"upscale_sigma_color", required = true},
            {"speed_up_thr", required = true}
        }
        local from, to, flow, layers, averaging_block_size, max_flow, sigma_dist, sigma_color, 
            postprocess_window, sigma_dist_fix, sigma_color_fix, occ_thr, upscale_averaging_radius,
            upscale_sigma_dist, upscale_sigma_color, speed_up_thr = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.calcOpticalFlowSF_expanded(
            cv.wrap_tensor(from), cv.wrap_tensor(to), cv.wrap_tensor(flow),
            layers, averaging_block_size, max_flow, sigma_dist, sigma_color, 
            postprocess_window, sigma_dist_fix, sigma_color_fix, occ_thr, upscale_averaging_radius,
            upscale_sigma_dist, upscale_sigma_color, speed_up_thr))
    else
        local argRules = {
            {"from", required = true},
            {"to", required = true},
            {"flow", default = nil},
            {"layers", required = true},
            {"averaging_block_size", required = true},
            {"max_flow", required = true}

        }
        local from, to, flow, layers, averaging_block_size, max_flow = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.calcOpticalFlowSF(
            cv.wrap_tensor(from), cv.wrap_tensor(to), cv.wrap_tensor(flow),
            layers, averaging_block_size, max_flow))
    end
end

function optflow.calcOpticalFlowSparseToDense(t)
    local argRules = {
        {"from", required = true},
        {"to", required = true},
        {"flow", default = nil},
        {"grid_step", default = 8},
        {"k", default = 128},
        {"sigma", default = 0.05},
        {"use_post_proc", default = true},
        {"fgs_lambda", default = 500.0},
        {"fgs_sigma", default = 1.5}
    }
    local from, to, flow, grid_step, k, sigma, use_post_proc, fgs_lambda, fgs_sigma = 
        cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.calcOpticalFlowSparseToDense(
        cv.wrap_tensor(from), cv.wrap_tensor(to), cv.wrap_tensor(flow), grid_step, k,
        sigma, use_post_proc, fgs_lambda, fgs_sigma))
end

function optflow.readOpticalFlow(t)
    local argRules = {
        {"path", required = true}
    }
    local path = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.readOpticalFlow(path))
end

function optflow.writeOpticalFlow(t)
    local argRules = {
        {"path", required = true},
        {"flow", required = true}
    }
    local path, flow = cv.argcheck(t, argRules)

    return C.writeOpticalFlow(path, cv.wrap_tensor(flow))
end

function optflow.updateMotionHistory(t)
    local argRules = {
        {"silhouette", requred = true},
        {"mhi", required = true},
        {"timestamp", required = true},
        {"duration", required = true}
    }
    local silhouette, mhi, timestamp, duration = cv.argcheck(t, argRules)

    C.updateMotionHistory(cv.wrap_tensor(silhouette), cv.wrap_tensor(mhi), timestamp, duration)
end

function optflow.calcMotionGradient(t)
    local argRules = {
        {"mhi", required = true},
        {"mask", default = nil},
        {"orientation", default = nil},
        {"delta1", required = true},
        {"delta2", required = true},
        {"apertureSize", default = 3}
    }
    local mhi, mask, orientation, delta1, delta2, apertureSize = cv.argcheck(t, argRules)

    return cv.unwrap_tensors(C.calcMotionGradient(
        cv.wrap_tensor(mhi), cv.wrap_tensor(mask), cv.wrap_tensor(orientation), 
        delta1, delta2, apertureSize))
end

function optflow.calcGlobalOrientation(t)
    local argRules = {
        {"orientation", required = true},
        {"mask", required = true},
        {"mhi", required = true},
        {"timestamp", required = true},
        {"duration", required = true}
    }
    local orientation, mask, mhi, timestamp, duration = cv.argcheck(t, argRules)

    return C.calcGlobalOrientation(
        cv.wrap_tensor(orientation), cv.wrap_tensor(mask), 
        cv.wrap_tensor(mhi), timestamp, duration)
end

-- TODO: currently it's very slow. To be optimized in the future
function optflow.segmentMotion(t)
    local argRules = {
        {"mhi", required = true},
        {"segmask", default = nil},
        {"timestamp", required = true},
        {"segThresh", required = true}
    }
    local mhi, segmask, timestamp, segThresh = cv.argcheck(t, argRules)

    local result = C.segmentMotion(
        cv.wrap_tensor(mhi), cv.wrap_tensor(segmask), timestamp, segThresh)
    return cv.unwrap_tensors(result.tensor), result.rects
end

return optflow
