OpenCV bindings for LuaJIT+Torch
=====================

For complete info on the project, visit its [Wiki](https://github.com/VisionLabs/torch-opencv/wiki).

See [this page](https://github.com/VisionLabs/torch-opencv/wiki/Trying-it-out) for a watching-demos quickstart.

#Tutorial

This document showcases code snippets to use basic image based Opencv functionalities.

###Requiring torch-opencv
```lua
local cv = require 'cv'
require 'cv.imgcodecs' -- reading/writing images
require 'cv.imgproc' -- image processing
```

###Color Conversion
cv.imread and cv.imwrite reverses color order. If the image is in RGB (on disk) format then after reading it becomes BGR (in memory) and vice-versa for imwrite.
```lua
src = cv.imread{imagePath, -1} -- loads image in row-major format
dst = tensor:clone()
cv.cvtColor{src=src, dst=dst, code=cv.COLOR_BGR2RGB}
```

###Resize image
```lua
src = cv.imread{imagePath, -1} -- loads image in row-major format
print(src:size())

 2448
 3264
    3
[torch.LongStorage of size 3]

-- resize to fixed size
dst = cv.resize{src=src, dsize={1632, 1224}, interpolation=cv.INTER_CUBIC}
print(dst:size())

 1224
 1632
    3
[torch.LongStorage of size 3]

-- resize by scale
scale = 0.25
dst = cv.resize{src=src, fx=scale, fy=scale, interpolation=cv.INTER_AREA}
print(dst:size())

 612
 816
   3
[torch.LongStorage of size 3]
```
