require 'cutorch'
local cuda = {}

cuda.SF_UYVY = 0
cuda.SF_YUY2 = 1
cuda.SF_YV12 = 2
cuda.SF_NV12 = 3
cuda.SF_IYUV = 4
cuda.SF_BGR = 5
cuda.SF_GRAY = 5

cuda.CODEC_MPEG1 = 0
cuda.CODEC_MPEG2 = 1
cuda.CODEC_MPEG4 = 2
cuda.CODEC_VC1 = 3
cuda.CODEC_H264 = 4
cuda.CODEC_JPEG = 5
cuda.CODEC_H264_SVC = 6
cuda.CODEC_H264_MVC = 7
cuda.CODEC_Uncompressed_YUV420 = 1230591318
cuda.CODEC_Uncompressed_YV12 = 1498820914
cuda.CODEC_Uncompressed_NV12 = 1314271538
cuda.CODEC_Uncompressed_YUYV = 1498765654
cuda.CODEC_Uncompressed_UYVY = 1431918169

cuda.CHROMA_FORMAT_Monochrome = 0
cuda.CHROMA_FORMAT_YUV420 = 1
cuda.CHROMA_FORMAT_YUV422 = 2
cuda.CHROMA_FORMAT_YUV444 = 3

local ffi = require 'ffi'

ffi.cdef[[
struct cutorchInfo {
    int deviceID;
    struct THCState *state;
};
]]

function cuda._info()
    return ffi.new('struct cutorchInfo', cutorch.getDevice(), cutorch._state)
end

return cuda
