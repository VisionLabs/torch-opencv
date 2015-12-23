-- Calculates an elementwise minimum of two random matrices on GPU
local cv = require 'cv'
cv.cuda = require 'cv.cudaarithm'
require 'cutorch'

local n = 5

local a = (torch.rand(n, n) * 10):int():float():cuda()
local b = (torch.rand(n, n) * 10):int():float():cuda()
local min_torch = torch.FloatTensor(n, n)

for i = 1, n do
    for j = 1, n do
        min_torch[i][j] = math.min(a[i][j], b[i][j])
    end
end

local min_cv = cv.cuda.min{a, b}

if n <= 20 then
    print('Matrix of minimums by Torch:')
    print(min_torch)

    print('Matrix of minimums by CUDA-OpenCV:')
    print(min_cv)
end

if (min_torch:eq(min_cv:float()) - 1):sum() == 0 then
    print("OpenCV-CUDA's min() function works fine!")
else
    print("You shouldn't see this message -- something went wrong! Please contact developers.")
end