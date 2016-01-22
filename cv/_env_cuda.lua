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

cuda.COLOR_BayerBG2BGR_MHT = 256
cuda.COLOR_BayerGB2BGR_MHT = 257
cuda.COLOR_BayerRG2BGR_MHT = 258
cuda.COLOR_BayerGR2BGR_MHT = 259

cuda.COLOR_BayerBG2RGB_MHT = COLOR_BayerRG2BGR_MHT
cuda.COLOR_BayerGB2RGB_MHT = COLOR_BayerGR2BGR_MHT
cuda.COLOR_BayerRG2RGB_MHT = COLOR_BayerBG2BGR_MHT
cuda.COLOR_BayerGR2RGB_MHT = COLOR_BayerGB2BGR_MHT

cuda.COLOR_BayerBG2GRAY_MHT = 260
cuda.COLOR_BayerGB2GRAY_MHT = 261
cuda.COLOR_BayerRG2GRAY_MHT = 262
cuda.COLOR_BayerGR2GRAY_MHT = 263

cuda.ALPHA_OVER = 0
cuda.ALPHA_IN = 1
cuda.ALPHA_OUT = 2
cuda.ALPHA_ATOP = 3
cuda.ALPHA_XOR = 4
cuda.ALPHA_PLUS = 5
cuda.ALPHA_OVER_PREMUL = 6
cuda.ALPHA_IN_PREMUL = 7
cuda.ALPHA_OUT_PREMUL = 8
cuda.ALPHA_ATOP_PREMUL = 9
cuda.ALPHA_XOR_PREMUL = 10
cuda.ALPHA_PLUS_PREMUL = 11
cuda.ALPHA_PREMUL = 12

local ffi = require 'ffi'

ffi.cdef[[
struct cutorchInfo {
    int deviceID;
    struct THCState *state;
};
]]

function cuda._info()
    return ffi.new('struct cutorchInfo', cutorch.getDevice(), cutorch.getState())
end

return cuda
