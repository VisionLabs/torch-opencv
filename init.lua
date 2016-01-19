-- This file contains common vars and funcs
local cv = require 'cv._env'

local ffi = require 'ffi'

require 'torch'
require 'paths'

local packageRoot = paths.thisfile('')

function cv.libPath(libName)
    if     ffi.os == 'Windows' then
        return packageRoot .. '\\lib\\' .. libName .. '.dll'
    elseif ffi.os == 'OSX' then
        return packageRoot .. '/lib/lib' .. libName .. '.dylib'
    else
        return packageRoot .. '/lib/lib' .. libName .. '.so'
    end
end

ffi.cdef[[
struct TensorWrapper {
    void *tensorPtr;
    char typeCode;
};

struct TensorArray {
    struct TensorWrapper *tensors;
    int size;
};

void *malloc(size_t size);
void free(void *ptr);

void transfer_tensor(void *destination, void *source);
void transfer_tensor_CUDA(void *state, void *destination, void *source);

struct SizeWrapper {
    int width, height;
};

struct Size2fWrapper {
    float width, height;
};

struct TermCriteriaWrapper {
    int type, maxCount;
    double epsilon;
};

struct ScalarWrapper {
    double v0, v1, v2, v3;
};

struct Vec2dWrapper {
    double v0;
};

struct Vec3dWrapper {
    double v0, v1, v2;
};

struct Vec3fWrapper {
    float v0, v1, v2;
};

struct RectWrapper {
    int x, y, width, height;
};

struct PointWrapper {
    int x, y;
};

struct Point2fWrapper {
    float x, y;
};

struct RotatedRectWrapper {
    struct Point2fWrapper center;
    struct Size2fWrapper size;
    float angle;
};

struct MomentsWrapper {
    double m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
    double mu20, mu11, mu02, mu30, mu21, mu12, mu03;
    double nu20, nu11, nu02, nu30, nu21, nu12, nu03;
};

struct RotatedRectPlusRect {
    struct RotatedRectWrapper rotrect;
    struct RectWrapper rect;
};

struct DMatchWrapper {
    int queryIdx;
    int trainIdx;
    int imgIdx;
    float distance;
};

struct DMatchArray {
    int size;
    struct DMatchWrapper *data;
};

struct DMatchArrayOfArrays {
    int size;
    struct DMatchArray *data;
};

struct TensorPlusDouble {
    struct TensorWrapper tensor;
    double val;
};

struct TensorPlusFloat {
    struct TensorWrapper tensor;
    float val;
};

struct TensorPlusBool {
    struct TensorWrapper tensor;
    bool val;
};

struct TensorArrayPlusInt {
    struct TensorArray tensors;
    int val;
};

struct TensorArrayPlusFloat {
    struct TensorArray tensors;
    float val;
};

struct TensorArrayPlusDouble {
    struct TensorArray tensors;
    double val;
};

struct TensorArrayPlusBool {
    struct TensorArray tensors;
    bool val;
};

struct RectPlusInt {
    struct RectWrapper rect;
    int val;
};

struct ScalarPlusBool {
    struct ScalarWrapper scalar;
    bool val;
};

struct IntArray {
    int *data;
    int size;
};

struct FloatArray {
    float *data;
    int size;
};

struct DoubleArray {
    double *data;
    int size;
};

struct PointArray {
    struct PointWrapper *data;
    int size;
};

struct RectArray {
    struct RectWrapper *data;
    int size;
};

struct TensorPlusRectArray {
    struct TensorWrapper tensor;
    struct RectArray rects;
};

struct FloatArrayOfArrays {
    float **pointers;
    float *realData;
    int dims;
};

int getIntMax();
float getFloatMax();

struct PointArrayOfArrays {
    struct PointWrapper **pointers;
    struct PointWrapper *realData;
    int dims;
    int *sizes;
};

]]

local C = ffi.load(cv.libPath('Common'))
local _, CUDACommon_C = pcall(ffi.load, cv.libPath('CUDACommon'))

require 'cv.constants'

cv.INT_MAX = C.getIntMax()
cv.FLT_MAT = C.getFloatMax()
cv.NULLPTR = ffi.new('void *', nil)

--- ***************** Argument checking & unpacking *****************

function cv.argcheck(t, rules)
    local retval = {}
    for i, argument in ipairs(rules) do
        local userInputArg = t[argument[1]]
        if userInputArg == nil then
            userInputArg = t[i]
        end

        if userInputArg == nil then
            if argument.required then
                error('Argument #' .. i .. ' ("' .. argument[1] .. '") is required!')
            else
                userInputArg = argument.default
            end
        end

        local
        function identity(...) return ... end
        retval[i] = (argument.operator or identity)(userInputArg)
    end
    return table.unpack(retval, 1, #rules)
end

--- ***************** Tensor <=> Mat conversion *****************

local tensor_CV_code_by_letter = {
    [ 66] = cv.CV_8U  , -- B : Byte
    [ 70] = cv.CV_32F , -- F : Float
    [ 68] = cv.CV_64F , -- D : Double
    [ 73] = cv.CV_32S , -- I : Int
    [ 83] = cv.CV_16S , -- S : Short
    [104] = cv.CV_8S  , -- h : Char
    [117] = cv.CV_CUDA, -- u : Cuda
}

local tensor_type_by_CV_code = {
    [cv.CV_8U  ] = "Byte",
    [cv.CV_32F ] = "Float",
    [cv.CV_64F ] = "Double",
    [cv.CV_32S ] = "Int",
    [cv.CV_16S ] = "Short",
    [cv.CV_8S  ] = "Char",
    [cv.CV_CUDA] = "Cuda"
}

cv.EMPTY_WRAPPER = ffi.new("struct TensorWrapper", nil)
cv.EMPTY_MULTI_WRAPPER = ffi.new("struct TensorArray", nil)

function cv.tensorType(tensor)
    -- get the first letter of Tensor type
    local typeString = tensor:type()
    local letter1, letter2 = typeString:byte(7), typeString:byte(8)

    if letter1 == 76 then
        error("Sorry, LongTensors aren't supported. Consider using IntTensor")
    elseif letter1 == 67 then
        letter1 = letter2
    end

    return tensor_CV_code_by_letter[letter1]
end

local
function empty_tensor_of_type(code)
    assert(code ~= cv.CV_16U, "Sorry, cv::Mats of type CV_16U aren't supported.")
    return torch[tensor_type_by_CV_code[code] .. "Tensor"]()
end

-- torch.RealTensor ---> tensor:cdata(), tensor_type_CV_code
local
function prepare_for_wrapping(tensor)
    return tensor:cdata(), cv.tensorType(tensor)
end

-- torch.RealTensor ---> struct TensorWrapper
function cv.wrap_tensor(tensor)
    if not tensor then
        return cv.EMPTY_WRAPPER
    end

    return ffi.new("struct TensorWrapper", prepare_for_wrapping(tensor))
end

function cv.wrap_tensors(...)
    if not ... then
        return cv.EMPTY_MULTI_WRAPPER
    end

    local args = {...}
    if type(args[1]) == "table" then
        args = args[1]
    end

    local wrapper = ffi.new("struct TensorArray")
    wrapper.size = #args
    wrapper.tensors = ffi.gc(C.malloc(#args * ffi.sizeof("struct TensorWrapper *")), C.free)

    for i, tensor in ipairs(args) do
        wrapper.tensors[i-1] = cv.wrap_tensor(tensor)
    end

    return wrapper
end

-- struct TensorWrapper(s) ---> torch.RealTensor
function cv.unwrap_tensors(wrapper, toTable)
    if ffi.istype(ffi.typeof(wrapper), cv.EMPTY_WRAPPER) then
        -- handle single tensor
        if wrapper.tensorPtr == nil then
            return
        end

        local retval = empty_tensor_of_type(wrapper.typeCode)

        if wrapper.typeCode == cv.CV_CUDA then
            CUDACommon_C.transfer_tensor_CUDA(cutorch._state, retval:cdata(), wrapper.tensorPtr)
        else
            C.transfer_tensor(retval:cdata(), wrapper.tensorPtr)
        end

        return retval
    else
        -- handle multiple tensors
        if wrapper.tensors == nil then
            -- return nothing in case of a nullptr
            return
        end

        local retval = {}
        for i = 0,wrapper.size-1 do
            local tempTensor = empty_tensor_of_type(wrapper.tensors[i].typeCode)
            C.transfer_tensor(tempTensor:cdata(), wrapper.tensors[i].tensorPtr)
            table.insert(retval, tempTensor)
        end

        C.free(wrapper.tensors)
        if toTable then
            return retval
        else
            return table.unpack(retval)
        end
    end
end

-- see filter_depths in opencv2/imgproc.hpp
function cv.checkFilterCombination(src, ddepth)
    local srcType = cv.tensorType(src)
    if srcType == cv.CV_8U then
        return ddepth == cv.CV_16S or ddepth >= cv.CV_32F or ddepth == -1
    elseif srcType == cv.CV_16S or srcType == cv.CV_32F then
        return ddepth >= cv.CV_32F or ddepth == -1
    elseif srcType == cv.CV_64F then
        return ddepth == cv.CV_64F or ddepth == -1
    else
        return false
    end
end

--- ***************** Wrappers for small OpenCV classes *****************
-- Use these for passing into functions. Example:

-- r = cv.Rect(10, 10, 15, 25)
-- OR
-- r = cv.Rect{10, 10, 15, 25}
-- OR
-- r = cv.Rect{x=10, y=10, width=15, height=25}
-- OR
-- r1 = cv.Rect{x=10, y=10, width=15, height=25}
-- r2 = cv.Rect(r1)

-- same with most of the following wrappers (see OpenCV defs)

-- TODO: generate these straight in the code

function cv.Rect(...)
    return ffi.new('struct RectWrapper', ...)
end

function cv.TermCriteria(...)
    return ffi.new('struct TermCriteriaWrapper', ...)
end

function cv.Scalar(...)
    return ffi.new('struct ScalarWrapper', ...)
end

function cv.Moments(...)
    return ffi.new('struct MomentsWrapper', ...)
end

function cv.Size(...)
    return ffi.new('struct SizeWrapper', ...)
end

function cv.Size2f(...)
    return ffi.new('struct Size2fWrapper', ...)
end

function cv.Vec3d(...)
    return ffi.new('struct Vec3dWrapper', ...)
end

function cv.Point(...)
    return ffi.new('struct PointWrapper', ...)
end

function cv.Point2f(...)
    return ffi.new('struct Point2fWrapper', ...)
end

function cv.RotatedRect(...)
    return ffi.new('struct RotatedRectWrapper', ...)
end

--- ***************** Other helper structs *****************

--[[
Makes an <IntArray, FloatArray, ...> from a table of numbers.
Generally, if you're calling a function that uses Array many
times, consider reusing the retval of this function.

Example (not very realistic):

cv.calcHist{images=im, channels={3,3,1,3,4}, ......}

OR

ch = cv.newArray('Int', {3,3,1,3,4})
for i = 1,1e8 do
    cv.calcHist{images=im, channels=ch, ...}
    ......
--]]
function cv.newArray(elemType, data)
    local retval
    local fullTypeName
    local shortTypeName

    if elemType:byte(3) == 46 then
        -- there's a period after 2 symbols: likely "cv.Something"
        shortTypeName = elemType:sub(4)
        fullTypeName = 'struct ' .. shortTypeName .. 'Wrapper'
        retval = ffi.new('struct ' .. shortTypeName .. 'Array')
    else
        -- C primitive type, such as 'Int' or 'Float'
        fullTypeName = elemType:lower()
        retval = ffi.new('struct ' .. elemType .. 'Array')
    end

    if not data then
        -- create an array with no data
        -- here, we assume that our C function will resize the array
        retval.data = ffi.gc(cv.NULLPTR, C.free)
        retval.size = 0
        return retval
    end

    if type(data) == 'number' then
        retval.data = ffi.gc(C.malloc(data * ffi.sizeof(fullTypeName)), C.free)
        retval.size = data
        return retval
    end

    retval.data = ffi.gc(C.malloc(#data * ffi.sizeof(fullTypeName)), C.free)
    retval.size = #data


    if elemType:byte(3) == 46 then
        for i, value in ipairs(data) do
            retval.data[i-1] = cv[shortTypeName](data[i])
        end
    else
        for i, value in ipairs(data) do
            retval.data[i-1] = data[i]
        end
    end

    return retval
end

-- TODO: function cv.newArrayOfArrays(elemType, data)

-- example: table of tables of numbers ---> struct FloatArrayOfArrays
function cv.numberArrayOfArrays(elemType, data)
    local retval = ffi.new('struct ' .. elemType .. 'ArrayOfArrays')

    -- first, compute relative addresses
    retval.pointers = ffi.gc(C.malloc(#data * ffi.sizeof(elemType:lower() .. '*') + 1), C.free)
    retval.pointers[0] = nil

    for i, row in ipairs(data) do
        data[i] = data[i-1] + #row
    end
    retval.realData = ffi.gc(C.malloc(totalElemSize * ffi.sizeof(elemType:lower())), C.free)

    retval.pointers[0] = retval.realData
    local counter = 0
    for i, row in ipairs(data) do
        -- fill data
        for j, elem in ipairs(row) do
            retval.realData[counter] = elem
            counter = counter + 1
        end
        -- transform relative addresses to absolute
        retval.pointers[i] = retval.pointers[i] + retval.realData
    end
end


function cv.arrayToLua(array, outputType, output)
    local retval

    if output then
        for i = 1,array.size do
            output[i] = array.data[i-1]
        end

        C.free(array.data)
        return output
    end

    if     outputType == 'table' then
        retval = {}
    elseif outputType == 'Tensor' then
        -- ctype has the form 'ctype<struct IntArray>'
        local ctype = tostring(ffi.typeof(array))
        local typeStart = 14
        local typeEnd = ctype:find('Arr') - 1
        retval = torch[ctype:sub(typeStart, typeEnd) .. 'Tensor'](array.size)
    end

    for i = 1,array.size do
        retval[i] = array.data[i-1]
    end

    C.free(array.data)
    return retval
end

-- make an array that has come from C++ garbage-collected
function cv.gcarray(array)
    array.data = ffi.gc(array.data, C.free)
    return array
end

return cv
