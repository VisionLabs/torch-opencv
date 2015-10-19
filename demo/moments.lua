require 'cv.imgproc'

points = {
    {10, 23.4},
    {17, 10.1},
    {20, 31.2},
    {65, 43.1},
    {5.67, 11},
    {10, 4.42}
}

pointsTensor = torch.FloatTensor(points)
moments = cv.moments{array=pointsTensor}

Hu_table = {}
Hu_tensor = torch.DoubleTensor(7)

cv.HuMoments{moments=moments, outputType='table', output=Hu_table}
Hu_tensor = cv.HuMoments{moments=moments, outputType='Tensor', output=Hu_tensor}

print(Hu_table)
print(Hu_tensor)