-- Currently no unit tests are present here.
-- Let this be the future site for them

local cv = require 'cv'

local ffi = require 'ffi'

ffi.cdef[[
void test_tensor_to_mat(struct TensorWrapper tensor);
struct TensorWrapper test_mat_to_tensor();
]]

local ok, C = pcall(ffi.load, cv.libPath('Tests'))
if not ok then
    print('torch-opencv is built with BUILD_TESTS = OFF. Aborting.')
    os.exit(-1)
end

-- prints Tensor from OpenCV
function test_tensor_to_mat(tensor)
    assert(tensor:dim() <= 2 or tensor:dim() == 3 and tensor:size(3) <= 4)
    C.test_tensor_to_mat(cv.wrap_tensor(tensor))
end

-- creates a Mat and returns it as a Tensor
function test_mat_to_tensor()
    return cv.unwrap_tensors(C.test_mat_to_tensor())
end

-- *************** testing ***************

local tensor_a = torch.IntTensor(2, 2, 2)
tensor_a[1][1][1] = 1
tensor_a[1][2][1] = 2
tensor_a[2][1][1] = 3
tensor_a[2][2][1] = 4
tensor_a[1][1][2] = 5
tensor_a[1][2][2] = 6
tensor_a[2][1][2] = 7
tensor_a[2][2][2] = 8

test_tensor_to_mat(tensor_a)

local tensor_b = torch.DoubleTensor(3, 3)
for i=1,3 do
    for j=1,3 do
        tensor_b[i][j] = j^i
    end
end

test_tensor_to_mat(tensor_b)

local another_tensor = test_mat_to_tensor()
print(another_tensor)
