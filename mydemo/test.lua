local cv = require 'cv'
require 'cv.stitching'
require 'cv.features2d'

local ffi = require 'ffi'

local elemtype = {}

elemtype[1] = cv.MatchesInfo{}
elemtype[2] = cv.MatchesInfo{}
--elemtype[3] = cv.MatchesInfo{}

local retval = cv.newArray("Class", elemtype)

local result = cv.test(retval)

print("__lua__")
print("size of output vector = " .. result.size)
--print(result.data)
--print(result.data[0])