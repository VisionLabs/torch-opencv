require 'image'
local ffi = require 'ffi'

ffi.cdef[[
void bilateralFilter(THFloatTensor *in, THFloatTensor *out, int d, double sigmaColor, double sigmaSpace);
void medianFilter(THFloatTensor *in, THFloatTensor *out, int d);
void createVideoWriter(const char* name, int w, int h, float framerate);
void addFrame(THFloatTensor *t);
void releaseVideoWriter();
]]
local C = ffi.load'build/libfilters.so'

function bilateralFilter(im, d, sigmaColor, sigmaSpace)
  assert(im:nDimension() == 3)
  assert(im:size(1) == 3)
  local a = im:permute(2,3,1):float():contiguous()
  print(#a)
  local ret = torch.FloatTensor()
  C.bilateralFilter(a:cdata(), ret:cdata(), d, sigmaColor, sigmaSpace)
  return ret:permute(3,1,2):float()
end

function medianFilter(im, k)
  assert(im:nDimension() == 3)
  assert(im:size(1) == 3)
  local a = im:permute(2,3,1):float():contiguous()
  print(#a)
  local ret = torch.FloatTensor()
  C.medianFilter(a:cdata(), ret:cdata(), k)
  return ret:permute(3,1,2):float()
end

function createVideoWriter(filename, w, h, framerate)
  C.createVideoWriter(filename, w, h, framerate or 30);
end

function addFrame(t)
  C.addFrame(t:cdata())
end

function releaseVideoWriter()
  C.releaseVideoWriter()
end
