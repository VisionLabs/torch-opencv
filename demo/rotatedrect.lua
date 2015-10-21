require 'cv.imgproc'

r = cv.RotatedRect{center={50, 120}, size={40, 90}, angle=30}

points = torch.FloatTensor(4, 2)
cv.boxPoints{box=r, points=points}

print(points)