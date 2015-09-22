ffi = require 'ffi'

ffi.cdef[[
struct TensorWrapper {
    void *tensorPtr;
    char tensorType;
};

void test_tensor_to_mat(TensorWrapper tensor);
void test_mat_to_tensor(TensorWrapper tensor);
]]

local C = ffi.load 'lib/libTypeConversion.so'

function wrap_tensor(tensor)
    -- TODO: maybe implement a lookup table
    local tensor_type = tensor:type():byte(7)
    local tensor_type_code

    if     tensor_type == 66 then -- Byte
        tensor_type_code = 0 -- CV_8U
    elseif tensor_type == 70 then -- Float
        tensor_type_code = 5 -- CV_32F
    elseif tensor_type == 68 then -- Double
        tensor_type_code = 6 -- CV_64F
    elseif tensor_type == 73 then -- Int
        tensor_type_code = 4 -- CV_32S
    elseif tensor_type == 83 then -- Short
        tensor_type_code = 3 -- CV_16S
    elseif tensor_type == 67 then -- Char
        tensor_type_code = 1 -- CV_8S
    elseif tensor_type == 76 then -- Long
        error("Mats of type long are not supported. Consider using int")
    end

    return ffi.new("struct TensorWrapper", tensor:cdata(), tensor_type_code)
end

torch.setdefaulttensortype("torch.FloatTensor")

function test_tensor_to_mat(tensor)
    assert(tensor:dim() <= 2 or tensor:dim() == 3 and tensor:size(3) <= 4)
    C.test_tensor_to_mat(wrap_tensor(tensor))
end

function test_mat_to_tensor()
    retval = torch.Tensor()
    C.test_mat_to_tensor(retval:cdata())
    return retval
end

tensor_a = torch.Tensor(2, 2, 2)
tensor_a[1][1][1] = 1
tensor_a[1][2][1] = 2
tensor_a[2][1][1] = 3
tensor_a[2][2][1] = 4
tensor_a[1][1][2] = 5
tensor_a[1][2][2] = 6
tensor_a[2][1][2] = 7
tensor_a[2][2][2] = 8
test_tensor_to_mat(tensor_a)

tensor_b = torch.Tensor(2, 3)
tensor_b[1][1] = 1
tensor_b[1][2] = 2
tensor_b[1][3] = 3
tensor_b[2][1] = 1
tensor_b[2][2] = 4
tensor_b[2][3] = 9
test_tensor_to_mat(tensor_b)

--print(test_mat_to_tensor())