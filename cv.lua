-- This file contains common vars and funcs

local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper {
    void *tensorPtr;
    char typeCode;
};

struct MultipleTensorWrapper {
    struct TensorWrapper *tensors;
    short size;
};

void *malloc(size_t size);
void free(void *ptr);

void transfer_tensor(void *destination, void *source);

struct Algorithm;
struct Algorithm *createAlgorithm();
void destroyAlgorithm(struct Algorithm *ptr);
]]

local C = ffi.load 'lib/libCommon.so'

cv = {}

require 'cv.constants'

--- ***************** Tensor <=> Mat conversion *****************

local tensor_CV_code_by_letter = {
    [66] = cv.CV_8U , -- Byte
    [70] = cv.CV_32F, -- Float
    [68] = cv.CV_64F, -- Double
    [73] = cv.CV_32S, -- Int
    [83] = cv.CV_16S, -- Short
    [67] = cv.CV_8S , -- Char
}

local tensor_type_by_CV_code = {
    [cv.CV_8U ] = "Byte",
    [cv.CV_32F] = "Float",
    [cv.CV_64F] = "Double",
    [cv.CV_32S] = "Int",
    [cv.CV_16S] = "Short",
    [cv.CV_8S ] = "Char"
}

local
function empty_tensor_of_type(code)
    return torch[tensor_type_by_CV_code[code] .. "Tensor"]()
end

-- torch.RealTensor ---> tensor:cdata(), tensor_type_CV_code
local 
function prepare_for_wrapping(tensor)
    -- get the first letter of Tensor type
    local tensor_type = tensor:type():byte(7)

    if tensor_type == 76 then
        error("Mats of type long are not supported. Consider using int")
    end

    return tensor:cdata(), tensor_CV_code_by_letter[tensor_type]
end

-- torch.RealTensor ---> struct TensorWrapper/struct MultipleTensorWrapper
function cv.wrap_tensors(...)
    local args = {...}

    if #args == 1 then
        return ffi.new("struct TensorWrapper", prepare_for_wrapping(args[1]))
    else
        wrapper = ffi.new("struct MultipleTensorWrapper")
        wrapper.size = #args
        wrapper.tensors = C.malloc(#args * ffi.sizeof("struct TensorWrapper *"))

        for i, tensor in ipairs(args) do
            wrapper.tensors[i-1] = cv.wrap_tensors(tensor)
        end

        return wrapper
    end
end

-- struct TensorWrapper(s) ---> torch.RealTensor
function cv.unwrap_tensors(wrapper)
    if ffi.typeof(wrapper) == ffi.typeof("struct TensorWrapper") then
        -- handle single tensor
        retval = empty_tensor_of_type(wrapper.typeCode)
        C.transfer_tensor(retval:cdata(), wrapper.tensorPtr)
        return retval
    else
        -- handle multiple tensors
        retval = {}
        for i = 0,wrapper.size-1 do
            temp_tensor = empty_tensor_of_type(wrapper.tensors[i].typeCode)
            C.transfer_tensor(temp_tensor:cdata(), wrapper.tensors[i].tensorPtr)
            table.insert(retval, temp_tensor)
        end

        C.free(wrapper.tensors)
        return unpack(retval)
    end
end

--- ***************** Common base classes *****************

do
    -- TODO this: how to RAII?
    local Algorithm = torch.class('cv.Algorithm')
    
    function Algorithm:destroy()
        C.destroyAlgorithm(self.ptr);
    end

    function Algorithm:__init()
        --self.ptr = C.createAlgorithm();
    end
end

return cv