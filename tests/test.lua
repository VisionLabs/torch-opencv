require 'image'
local cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.imgproc'

local mytester = torch.Tester()
local cvtest = torch.TestSuite()

local lena_path = 'demo/data/lena.jpg'

local function HWCBGR_2_CHWRGB(im)
   return im:permute(3,1,2):index(1,torch.LongTensor{3,2,1})
end

-- test against torch/image
-- assuming that test is called from root path
function cvtest.imread()
   local ref_image = image.load(lena_path, 3, 'byte')
   local cv_image = HWCBGR_2_CHWRGB(cv.imread{lena_path})
   mytester:asserteq((ref_image - cv_image):float():abs():max(), 0, 'imread byte difference')
end

function cvtest.resize()
   local torch_image = image.load(lena_path, 3, 'byte')
   local cv_image = cv.imread{lena_path}

   local M = 256

   local interp_methods = {
      {torch='simple', cv=cv.INTER_NEAREST, error = 0},
      {torch='bilinear', cv=cv.INTER_LINEAR, error = 0.5}, -- implementations are different
   }

   for i,method in ipairs(interp_methods) do
      local torch_resized = image.scale(torch_image, M, M, method.torch)
      local cv_resized = cv.resize{src = cv_image, dsize = {M,M}, interpolation = method.cv}
      local diff = torch_resized - HWCBGR_2_CHWRGB(cv_resized)
      mytester:assertle(diff:float():abs():mean(), method.error, 'imread byte difference')
   end
end

function cvtest.flip()
   local torch_image = image.load(lena_path, 3, 'byte')
   local cv_image = cv.imread{lena_path}

   function image.hvflip(src)
      return image.vflip(image.hflip(src))
   end
   
   for i,v in ipairs{
      {f='hflip', code=1},
      {f='vflip', code=0},
      {f='hvflip', code=-1},
   } do
      local torch_hflipped = image[v.f](torch_image)
      local cv_hflipped = cv.flip{src = cv_image, flipCode = v.code}
      local diff = torch_hflipped - HWCBGR_2_CHWRGB(cv_hflipped)
      mytester:asserteq(diff:float():abs():max(), 0, 'imread byte difference')
   end
end

mytester:add(cvtest)
mytester:run()
