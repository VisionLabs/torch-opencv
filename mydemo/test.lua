local cv = require 'cv'
require 'cv.stitching'
require 'cv.features2d'

text = {}
text[1] = "123"
text[2] = "456"
text[3] = "789"

local retval = cv.test(text)

print(#retval)
print(retval[3])

--[[
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