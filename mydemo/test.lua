local cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.stitching'
require 'cv.features2d'

local images = {}
images[1] = cv.imread{'demo/lena.jpg' }
images[2] = cv.imread{'demo/lena.jpg' }

local K = torch.FloatTensor(3,3)
local R = torch.FloatTensor(3,3)
local T = torch.FloatTensor(1,3)
local src = torch.FloatTensor(30,30)

local mask = {}
mask[1] = cv.imread{'demo/lena.jpg'}
mask[2] = cv.imread{'demo/lena.jpg'}

local points = {}
points[1] = {1,1}
points[2] = {2,2}
local points_array = cv.newArray('cv.Point', points)

local size = {};
size[1] = {512,512}
size[2] = {512,512}
local size_array = cv.newArray('cv.Size', size)

local chr ={}
chr[1] = 255
chr[2] = 255
local chr_array = cv.newArray('UChar', chr)

text = {}
text[1] = "123"
text[2] = "456"
text[3] = "789"

local b = cv.Blender{0 }
local exposure_comp = cv.ExposureCompensator{0 }
local seam_finder = cv.SeamFinder{}

-------------------------data------------------------------------------

local temp = cv.Stitcher{0}

local v1, v2 , v3 = temp:featuresFinder{
                b = b,
                kind = 0,
                resol_mpx = 3,
                index = 2,
                corner = {1,1},
                conf_thresh = 3,
                resol_mpx = 4,
                flag = false,
                creator = seam_finder,
                seam_finder = seam_finder,
                exposure_comp = exposure_comp,
                corners = points_array,
                weight_maps = images,
                sizes = size_array,
                img = images[1],
                pyr = images,
                sharpness = 4,
                num_levels = 3,
                src = images[1],
                image = images[1],
                weight = images[1],
                dst = images[1],
                dst_mask = mask[2],
                mask = mask[2],
                corners = points_array,
                pano = images[1],
                images = images,
                masks = mask,
                mat = mask,
                chr = chr_array,
                tl = {1,1},
                dst_roi = {1, 1, 1, 1} }

-------------------------------------------------------------------
--[[
local dst = cv.test{img = images[1]}
cv.imshow{"test", images[1]}
cv.imshow{"test2", dst}
cv.waitKey{0}
]]

print(v1)
print(v2)
print(v3)
print(v4)