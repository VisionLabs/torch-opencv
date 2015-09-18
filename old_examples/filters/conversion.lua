local ffi = require 'ffi'

ffi.cdef[[
void test_tensor_conversion(THFloatTensor *tensor);
]]
local C = ffi.load 'lib/libconversion.so'

function test_tensor_conversion(tensor)
    C.test_tensor_conversion(tensor)
end

tensor = torch.FloatTensor(2, 2, 2)
---[[
tensor[1][1][1] = 1
tensor[1][2][1] = 2
tensor[2][1][1] = 3
tensor[2][2][1] = 4
--]]
---[[
tensor[1][1][2] = 5
tensor[1][2][2] = 6
tensor[2][1][2] = 7
tensor[2][2][2] = 8
--]]
--[[
tensor[1][1] = 1
tensor[1][2] = 2
tensor[2][1] = 3
tensor[2][2] = 4
--]]
test_tensor_conversion(tensor:cdata())
