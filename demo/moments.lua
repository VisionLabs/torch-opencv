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

Hu_table = cv.HuMoments{moments=moments, toTable=true}
Hu_tensor = cv.HuMoments{moments=moments, toTable=false}
print(Hu_table)
print(Hu_tensor)
