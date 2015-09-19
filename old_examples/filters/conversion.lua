local ffi = require 'ffi'

ffi.cdef[[
void test_tensor_to_mat(THFloatTensor *tensor);
void test_mat_to_tensor(THFloatTensor *output);
]]
local C = ffi.load 'lib/libconversion.so'

torch.setdefaulttensortype('torch.FloatTensor')

function test_tensor_to_mat(tensor)
	assert(tensor:dim() <= 2 or tensor:dim() == 3 and tensor:size(3) <= 4)
    C.test_tensor_to_mat(tensor:cdata())
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

print(test_mat_to_tensor())