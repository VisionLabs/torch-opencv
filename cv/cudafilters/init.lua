local cv = require 'cv._env'
require 'cutorch'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or require 'cv._env_cuda'

local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper Filter_applyCuda(struct cutorchInfo info,
    struct PtrWrapper ptr, struct TensorWrapper src, struct TensorWrapper dst);

struct PtrWrapper createBoxFilterCuda(
        int srcType, int dstType, struct SizeWrapper ksize, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal);

struct PtrWrapper createLinearFilterCuda(
        int srcType, int dstType, struct TensorWrapper kernel, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal);

struct PtrWrapper createLaplacianFilterCuda(
        int srcType, int dstType, int ksize, double scale,
        int borderMode, struct ScalarWrapper borderVal);

struct PtrWrapper createSeparableLinearFilterCuda(
        int srcType, int dstType, struct TensorWrapper rowKernel,
        struct TensorWrapper columnKernel, struct PointWrapper anchor,
        int rowBorderMode, int columnBorderMode);

struct PtrWrapper createDerivFilterCuda(
        int srcType, int dstType, int dx, int dy, int ksize, bool normalize,
        double scale, int rowBorderMode, int columnBorderMode);

struct PtrWrapper createSobelFilterCuda(
        int srcType, int dstType, int dx, int dy, int ksize,
        double scale, int rowBorderMode, int columnBorderMode);

struct PtrWrapper createScharrFilterCuda(
        int srcType, int dstType, int dx, int dy,
        double scale, int rowBorderMode, int columnBorderMode);

struct PtrWrapper createGaussianFilterCuda(
        int srcType, int dstType, struct SizeWrapper ksize,
        double sigma1, double sigma2, int rowBorderMode, int columnBorderMode);

struct PtrWrapper createMorphologyFilterCuda(
        int op, int srcType, struct TensorWrapper kernel,
        struct PointWrapper anchor, int iterations);

struct PtrWrapper createBoxMaxFilterCuda(
        int srcType, struct SizeWrapper ksize, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal);

struct PtrWrapper createBoxMinFilterCuda(
        int srcType, struct SizeWrapper ksize, struct PointWrapper anchor,
        int borderMode, struct ScalarWrapper borderVal);

struct PtrWrapper createRowSumFilterCuda(
        int srcType, int dstType, int ksize, int anchor,
        int borderMode, struct ScalarWrapper borderVal);

struct PtrWrapper createColumnSumFilterCuda(
        int srcType, int dstType, int ksize, int anchor,
        int borderMode, struct ScalarWrapper borderVal);
]]

local C = ffi.load(cv.libPath('cudafilters'))

require 'cv.Classes'
local Classes = ffi.load(cv.libPath('Classes'))

do
    local Filter = torch.class('cuda.Filter', 'cv.Algorithm', cv.cuda)

    function Filter:apply(t)
        local argRules = {
            {"src", required = true},
            {"dst", default = nil}
        }
        local src, dst = cv.argcheck(t, argRules)

        return cv.unwrap_tensors(C.Filter_applyCuda(
            cv.cuda._info(), self.ptr, cv.wrap_tensor(src), cv.wrap_tensor(dst)))
    end
end

function cv.cuda.createBoxFilter(t)
    local argRules = {
        {"srcType", required = true},
        {"dstType", required = true},
        {"ksize", required = true, operator = cv.Size},
        {"anchor", default = {-1, -1}, operator = cv.Point},
        {"borderMode", default = cv.BORDER_DEFAULT},
        {"borderVal", default = {0, 0, 0, 0}, operator = cv.Scalar}
    }
    local srcType, dstType, ksize, anchor, borderMode, borderVal = cv.argcheck(t, argRules)

    local retval = torch.factory('cuda.Filter')()
    retval.ptr = ffi.gc(C.createBoxFilterCuda(
        srcType, dstType, ksize, anchor, borderMode, borderVal),
        Classes.Algorithm_dtor)
    return retval
end

function cv.cuda.createLinearFilter(t)
    local argRules = {
        {"srcType", required = true},
        {"dstType", required = true},
        {"kernel", required = true},
        {"anchor", default = {-1, -1}, operator = cv.Point},
        {"borderMode", default = cv.BORDER_DEFAULT},
        {"borderVal", default = {0, 0, 0, 0}, operator = cv.Scalar}
    }
    local srcType, dstType, kernel, anchor, borderMode, borderVal = cv.argcheck(t, argRules)

    local retval = torch.factory('cuda.Filter')()
    retval.ptr = ffi.gc(C.createLinearFilterCuda(
        srcType, dstType, cv.wrap_tensor(kernel), anchor, borderMode, borderVal),
        Classes.Algorithm_dtor)
    return retval
end

function cv.cuda.createLaplacianFilter(t)
    local argRules = {
        {"srcType", required = true},
        {"dstType", required = true},
        {"ksize", default = 1},
        {"scale", default = 1},
        {"borderMode", default = cv.BORDER_DEFAULT},
        {"borderVal", default = {0, 0, 0, 0}, operator = cv.Scalar}
    }
    local srcType, dstType, ksize, scale, borderMode, borderVal = cv.argcheck(t, argRules)

    local retval = torch.factory('cuda.Filter')()
    retval.ptr = ffi.gc(C.createLaplacianFilterCuda(
        srcType, dstType, ksize, scale, borderMode, borderVal),
        Classes.Algorithm_dtor)
    return retval
end

function cv.cuda.createSeparableLinearFilter(t)
    local argRules = {
        {"srcType", required = true},
        {"dstType", required = true},
        {"rowKernel", required = true},
        {"columnKernel", required = true},
        {"anchor", default = {-1, -1}, operator = cv.Point},
        {"rowBorderMode", default = cv.BORDER_DEFAULT},
        {"columnBorderMode", default = -1}
    }
    local srcType, dstType, rowKernel, columnKernel, anchor,
        rowBorderMode, columnBorderMode = cv.argcheck(t, argRules)

    local retval = torch.factory('cuda.Filter')()
    retval.ptr = ffi.gc(C.createSeparableLinearFilterCuda(
        srcType, dstType, rowKernel, columnKernel, anchor, rowBorderMode, columnBorderMode),
        Classes.Algorithm_dtor)
    return retval
end

function cv.cuda.createDerivFilter(t)
    local argRules = {
        {"srcType", required = true},
        {"dstType", required = true},
        {"dx", required = true},
        {"dy", required = true},
        {"ksize", required = true},
        {"normalize", default = false},
        {"rowBorderMode", default = cv.BORDER_DEFAULT},
        {"columnBorderMode", default = -1}
    }
    local srcType, dstType, dx, dy, ksize, normalize, 
        rowBorderMode, columnBorderMode = cv.argcheck(t, argRules)

    local retval = torch.factory('cuda.Filter')()
    retval.ptr = ffi.gc(C.createDerivFilterCuda(
        srcType, dstType, dx, dy, ksize, normalize, rowBorderMode, columnBorderMode),
        Classes.Algorithm_dtor)
    return retval
end

function cv.cuda.createSobelFilter(t)
    local argRules = {
        {"srcType", required = true},
        {"dstType", required = true},
        {"dx", required = true},
        {"dy", required = true},
        {"ksize", default = 3},
        {"scale", default = 1},
        {"rowBorderMode", default = cv.BORDER_DEFAULT},
        {"columnBorderMode", default = -1}
    }
    local srcType, dstType, dx, dy, ksize, scale, 
        rowBorderMode, columnBorderMode = cv.argcheck(t, argRules)

    local retval = torch.factory('cuda.Filter')()
    retval.ptr = ffi.gc(C.createSobelFilterCuda(
        srcType, dstType, dx, dy, ksize, scale, rowBorderMode, columnBorderMode),
        Classes.Algorithm_dtor)
    return retval
end

function cv.cuda.createScharrFilter(t)
    local argRules = {
        {"srcType", required = true},
        {"dstType", required = true},
        {"dx", required = true},
        {"dy", required = true},
        {"scale", default = 1},
        {"rowBorderMode", default = cv.BORDER_DEFAULT},
        {"columnBorderMode", default = -1}
    }
    local srcType, dstType, dx, dy, scale, 
        rowBorderMode, columnBorderMode = cv.argcheck(t, argRules)

    local retval = torch.factory('cuda.Filter')()
    retval.ptr = ffi.gc(C.createScharrFilterCuda(
        srcType, dstType, dx, dy, scale, rowBorderMode, columnBorderMode),
        Classes.Algorithm_dtor)
    return retval
end

function cv.cuda.createGaussianFilter(t)
    local argRules = {
        {"srcType", required = true},
        {"dstType", required = true},
        {"ksize", required = true, operator = cv.Size},
        {"sigma1", required = true},
        {"sigma2", default = 0},
        {"rowBorderMode", default = cv.BORDER_DEFAULT},
        {"columnBorderMode", default = -1}
    }
    local srcType, dstType, ksize, sigma1, sigma2, 
        rowBorderMode, columnBorderMode = cv.argcheck(t, argRules)

    local retval = torch.factory('cuda.Filter')()
    retval.ptr = ffi.gc(C.createGaussianFilterCuda(
        srcType, dstType, ksize, sigma1, sigma2, rowBorderMode, columnBorderMode),
        Classes.Algorithm_dtor)
    return retval
end

function cv.cuda.createMorphologyFilter(t)
    local argRules = {
        {"op", required = true},
        {"srcType", required = true},
        {"kernel", required = true},
        {"anchor", default = {-1, -1}, operator = cv.Point},
        {"iterations", default = 1}
    }
    local op, srcType, kernel, anchor, iterations = cv.argcheck(t, argRules)

    local retval = torch.factory('cuda.Filter')()
    retval.ptr = ffi.gc(C.createMorphologyFilterCuda(
        op, srcType, kernel, anchor, iterations),
        Classes.Algorithm_dtor)
    return retval
end

function cv.cuda.createBoxMaxFilter(t)
    local argRules = {
        {"srcType", required = true},
        {"ksize", required = true, operator = cv.Size},
        {"anchor", default = {-1, -1}, operator = cv.Point},
        {"borderMode", default = cv.BORDER_DEFAULT},
        {"borderVal", default = {0, 0, 0, 0}, operator = cv.Scalar}
    }
    local srcType, ksize, anchor, borderMode, borderVal = cv.argcheck(t, argRules)

    local retval = torch.factory('cuda.Filter')()
    retval.ptr = ffi.gc(C.createBoxMaxFilterCuda(
        srcType, ksize, anchor, borderMode, borderVal),
        Classes.Algorithm_dtor)
    return retval
end

function cv.cuda.createBoxMinFilter(t)
    local argRules = {
        {"srcType", required = true},
        {"ksize", required = true, operator = cv.Size},
        {"anchor", default = {-1, -1}, operator = cv.Point},
        {"borderMode", default = cv.BORDER_DEFAULT},
        {"borderVal", default = {0, 0, 0, 0}, operator = cv.Scalar}
    }
    local srcType, ksize, anchor, borderMode, borderVal = cv.argcheck(t, argRules)

    local retval = torch.factory('cuda.Filter')()
    retval.ptr = ffi.gc(C.createBoxMinFilterCuda(
        srcType, ksize, anchor, borderMode, borderVal),
        Classes.Algorithm_dtor)
    return retval
end

function cv.cuda.createRowSumFilter(t)
    local argRules = {
        {"srcType", required = true},
        {"dstType", required = true},
        {"ksize", required = true, operator = cv.Size},
        {"anchor", default = -1},
        {"borderMode", default = cv.BORDER_DEFAULT},
        {"borderVal", default = {0, 0, 0, 0}, operator = cv.Scalar}
    }
    local srcType, dstType, ksize, anchor, borderMode, borderVal = cv.argcheck(t, argRules)

    local retval = torch.factory('cuda.Filter')()
    retval.ptr = ffi.gc(C.createRowSumFilterCuda(
        srcType, dstType, ksize, anchor, borderMode, borderVal),
        Classes.Algorithm_dtor)
    return retval
end

function cv.cuda.createColumnSumFilter(t)
    local argRules = {
        {"srcType", required = true},
        {"dstType", required = true},
        {"ksize", required = true, operator = cv.Size},
        {"anchor", default = -1},
        {"borderMode", default = cv.BORDER_DEFAULT},
        {"borderVal", default = {0, 0, 0, 0}, operator = cv.Scalar}
    }
    local srcType, dstType, ksize, anchor, borderMode, borderVal = cv.argcheck(t, argRules)

    local retval = torch.factory('cuda.Filter')()
    retval.ptr = ffi.gc(C.createColumnSumFilterCuda(
        srcType, dstType, ksize, anchor, borderMode, borderVal),
        Classes.Algorithm_dtor)
    return retval
end

return cv.cuda
