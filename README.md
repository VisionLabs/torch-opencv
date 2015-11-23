OpenCV bindings for LuaJIT+Torch
=====================

For complete info on the project, visit its [Wiki](https://github.com/VisionLabs/torch-opencv/wiki).

See [this page](https://github.com/VisionLabs/torch-opencv/wiki/Trying-it-out) for a watching-demos quickstart.

#Tutorial

This section showcases code snippets to use basic image based [OpenCV](http://opencv.org/) functionalities. Using the famous Lena image for this purpose.

###Requiring torch-opencv
OpenCV provides range of functionalities. You can require them as required.
```lua
local cv = require 'cv'
require 'cv.imgcodecs' -- reading/writing images
require 'cv.imgproc' -- image processing
require 'cv.highgui' -- GUI
require 'cv.ml' -- Machine Learning
require 'cv.videoio' -- Video
```

###Reading/Writing Image
OpenCV reads image in row major format and shape is (height, width, channels) unless the image is loaded as grayscale or it is grayscale and loaded with ```cv.IMREAD_UNCHANGED``` flag, in that case the shape is (height, width). Functions ```cv.imread``` and ```cv.imwrite``` reverses the channel order. If the image is in RGB on disk then after reading it becomes BGR (in memory) and vice-versa for image writing.

To load image as it is on disk use ```cv.IMREAD_UNCHANGED``` flag.
```lua
loadType = cv.IMREAD_UNCHANGED
src = cv.imread{imagePath, loadType}
print(src:size())
 512
 512
   3
[torch.LongStorage of size 3]
```

To load the image as color image use ```cv.IMREAD_COLOR``` flag.
```lua
--loadType: cv.IMREAD_COLOR, loads (always) 3 channel image.
loadType = cv.IMREAD_COLOR
src = cv.imread{imagePath, loadType}
print(src:size())
 512
 512
   3
[torch.LongStorage of size 3]
```

You can use ```cv.IMREAD_GRAYSCALE``` to load image as grayscale. In this case the color is converted to grayscale. For this conversion the channels of the image are assumed to be in RGB order.
```lua
loadType = cv.IMREAD_GRAYSCALE
src = cv.imread{imagePath, loadType}
print(src:size())
 512
 512
[torch.LongStorage of size 2]
```

For saving image to disk use ```cv.imwrite```. Image compression is defined by the extension of the ```imagePath```.
```lua
cv.imwrite{imagePath, src}
```
Third argument to the function can compression specific parameter. E.g if compression is *JPEG* then the parameter is *JPEG* compression quality. If not provided then default values are used.

###Color Conversion
OpenCV provides optimized color conversion functions. Following are couple of examples.

Convert BGR to YUV
```lua
dst = src:clone()
cv.cvtColor{src=src, dst=dst, code=cv.COLOR_BGR2YUV}
print(dst:size())

 512
 512
   3
[torch.LongStorage of size 3]
```

Convert to grayscale
```lua
dst = cv.cvtColor{src=src, code=cv.COLOR_BGR2GRAY}
print(dst:size())

 512
 512
[torch.LongStorage of size 2]
```

###Image Resize
Here is an exmaple to resize an image to fixed size.
```lua
dst = cv.resize{src=src, dsize={1024, 1024}, interpolation=cv.INTER_CUBIC}
print(dst:size())

 1024
 1024
    3
[torch.LongStorage of size 3]
```

We can also resize an image using scaling factot. You can use different scaling factor for height and width.
```lua
scaleX = 0.25
scaleY = 0.35
dst = cv.resize{src=src, fx=scaleX, fy=scaleY, interpolation=cv.INTER_AREA}
print(dst:size())

 179
 128
   3
[torch.LongStorage of size 3]
```

###Affine transformation
Affine transformation is one of the most widely used image processing function. We will go through this functionality using following image as an example.

**Source Image**

![Lena](demo/lena.jpg)

Affine transformation is a two step process.

1) Get affine rotation/scaling matrix.

```lua
height = src:size(1)
width = src:size(2)

-- rotate counter clockwise about center (in image coordinate system)
center = cv.Point2f{width/2, height/2}
angle = 45 -- in degrees
scale = 0.5

-- get rotation matrix
M = cv.getRotationMatrix2D{center=center, angle=angle, scale=scale}
print(M:size())
 2
 3
[torch.LongStorage of size 2]
```
Transformation matrix M provided by OpenCV has only rotation and scaling. You can add translation by adding [translationX translationY] to the last column of M.


2) Transforming Image (Affine Warp)
```lua
dsize = cv.Size{width, height} -- if not provided or zero then uses source image's size
dst = cv.warpAffine{src=src, M=M, dsize=dsize, flags=cv.INTER_LINEAR}
print(dst:size())

 512
 512
   3
[torch.LongStorage of size 3]
```

**Affine Transformed Image**

![Transformed image](demo/lenaTrans.jpg)
