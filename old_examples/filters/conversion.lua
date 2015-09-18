local ffi = require 'ffi'

ffi.cdef[[
void test_tensor_conversion(THFloatTensor *tensor);
]]
local C = ffi.load 'lib/libconversion.so'

function test_tensor_conversion(tensor)
	assert(tensor:dim() <= 2 or tensor:dim() == 3 and tensor:size(3) <= 4)
    C.test_tensor_conversion(tensor:cdata())
end

tensor_a = torch.FloatTensor(2, 2, 2)
tensor_b = torch.FloatTensor(3, 2)
---[[
tensor_a[1][1][1] = 1
tensor_a[1][2][1] = 2
tensor_a[2][1][1] = 3
tensor_a[2][2][1] = 4
tensor_a[1][1][2] = 5
tensor_a[1][2][2] = 6
tensor_a[2][1][2] = 7
tensor_a[2][2][2] = 8
--]]
---[[
tensor_b[1][1] = 1
tensor_b[1][2] = 2
tensor_b[2][1] = 3
tensor_b[2][2] = 4
--]]
test_tensor_conversion(tensor_a)
test_tensor_conversion(tensor_b)