local ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper {
    void *tensorPtr;
    char typeCode;
};

void transfer_tensor(void *destination, void *source);
]]

local C = ffi.load 'lib/libTypeConversion.so'

local tensor_CV_code_by_letter = {
    [66] = 0, -- Byte   -> CV_8U
    [70] = 5, -- Float  -> CV_32F
    [68] = 6, -- Double -> CV_64F
    [73] = 4, -- Int    -> CV_32S
    [83] = 3, -- Short  -> CV_16S
    [67] = 1, -- Char   -> CV_8S
}

local tensor_type_by_CV_code = {
    [0] = "Byte",
    [5] = "Float",
    [6] = "Double",
    [4] = "Int",
    [3] = "Short",
    [1] = "Char"
}

function empty_tensor_of_type(code)
    return torch[tensor_type_by_CV_code[code] .. "Tensor"]()
end

-- torch.RealTensor ---> struct TensorWrapper
function wrap_tensor(tensor)
    -- get the first letter of Tensor type
    local tensor_type = tensor:type():byte(7)
    
    if tensor_type == 76 then
        error("Mats of type long are not supported. Consider using int")
    end

    local tensor_type_CV_code = tensor_CV_code_by_letter[tensor_type]
    return ffi.new("struct TensorWrapper", tensor:cdata(), tensor_type_CV_code)
end

-- struct TensorWrapper ---> torch.RealTensor
function unwrap_tensor(tensor_wrapper)
    retval = empty_tensor_of_type(tensor_wrapper.typeCode)
    C.transfer_tensor(
        retval:cdata(),
        tensor_wrapper.tensorPtr
    )
    return retval
end
