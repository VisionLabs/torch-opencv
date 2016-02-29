local cv = require 'cv'
require 'cv.stitching'
require 'cv.features2d'

local K = torch.FloatTensor(3,3)
local R = torch.FloatTensor(3,3)
local T = torch.FloatTensor(1,3)
local src = torch.FloatTensor(30,30)

local temp = cv.detail_SphericalWarper{1}

temp:setScale{5}
local x = temp:getScale()

local v1, v2, v3, v4  = temp:warp{ src_size = {1,1},
                                        src = src,
                                        pt = {1,1},
                                        K = K,
                                        R = R,
                                        T = T,
                                        interp_mode = 1,
                                        border_mode = 1 }

print(x)
print(v1)
print(v2)
print(v3)
print(v4)

--[[

text = {}
text[1] = "123"
text[2] = "456"
text[3] = "789"

local retval = cv.test(text)

print(retval)


local elemtype = {}

elemtype[1] = cv.MatchesInfo{}
elemtype[2] = cv.MatchesInfo{}
elemtype[3] = cv.MatchesInfo{}

local retval = cv.newArray("Class", elemtype)

local result = cv.test(retval)

print("__lua__")

print(#result)


local num_images = 10
local pairwise_matches = {}
pairwise_matches[1] = cv.MatchesInfo{}
pairwise_matches[2] = cv.MatchesInfo{}
pairwise_matches[3] = cv.MatchesInfo{}

local span_tree, centers = cv.detail.findMaxSpanningTree{num_images = num_images,
                                                         pairwise_matches = pairwise_matches }

print(span_tree.ptr)
print(centers)
]]