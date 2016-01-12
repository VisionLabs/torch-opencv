local cv = require 'cv._env'

-- TODO: remove this after gathering all CUDA packages in a single submodule
cv.cuda = cv.cuda or {}

local ffi = require 'ffi'

ffi.cdef[[
struct EncoderParams {
    int P_Interval;      //!< NVVE_P_INTERVAL,
    int IDR_Period;      //!< NVVE_IDR_PERIOD,
    int DynamicGOP;      //!< NVVE_DYNAMIC_GOP,
    int RCType;          //!< NVVE_RC_TYPE,
    int AvgBitrate;      //!< NVVE_AVG_BITRATE,
    int PeakBitrate;     //!< NVVE_PEAK_BITRATE,
    int QP_Level_Intra;  //!< NVVE_QP_LEVEL_INTRA,
    int QP_Level_InterP; //!< NVVE_QP_LEVEL_INTER_P,
    int QP_Level_InterB; //!< NVVE_QP_LEVEL_INTER_B,
    int DeblockMode;     //!< NVVE_DEBLOCK_MODE,
    int ProfileLevel;    //!< NVVE_PROFILE_LEVEL,
    int ForceIntra;      //!< NVVE_FORCE_INTRA,
    int ForceIDR;        //!< NVVE_FORCE_IDR,
    int ClearStat;       //!< NVVE_CLEAR_STAT,
    int DIMode;          //!< NVVE_SET_DEINTERLACE,
    int Presets;         //!< NVVE_PRESETS,
    int DisableCabac;    //!< NVVE_DISABLE_CABAC,
    int NaluFramingType; //!< NVVE_CONFIGURE_NALU_FRAMING_TYPE
    int DisableSPSPPS;   //!< NVVE_DISABLE_SPS_PPS
};

struct EncoderParams EncoderParams_ctor_default();

struct EncoderParams EncoderParams_ctor(const char *configFile);

void EncoderParams_save(struct EncoderParams params, const char *configFile);
]]

local C = ffi.load(cv.libPath('cudacodec'))

require 'cv.Classes'
local Classes = ffi.load(cv.libPath('Classes'))

-- TODO test this on Windows
do
    local EncoderParams = torch.class('cuda.EncoderParams', cv.cuda)

    local paramNames = {
        P_Interval = true,
        IDR_Period = true,
        DynamicGOP = true,
        RCType = true,
        AvgBitrate = true,
        PeakBitrate = true,
        QP_Level_Intra = true,
        QP_Level_InterP = true,
        QP_Level_InterB = true,
        DeblockMode = true,
        ProfileLevel = true,
        ForceIntra = true,
        ForceIDR = true,
        ClearStat = true,
        DIMode = true,
        Presets = true,
        DisableCabac = true,
        NaluFramingType = true,
        DisableSPSPPS = true
    }

    function EncoderParams:__init(t)
        local argRules = {
            {"configFile", default = nil}
        }
        local configFile = cv.argcheck(t, argRules)

        if configFile then
            self.object = C.EncoderParams_ctor(configFile)
        else
            self.object = C.EncoderParams_ctor_default()
        end
    end

    function EncoderParams:__index__(key)
        if paramNames[key] ~= nil then
            return rawget(self, object)[key]
        else
            return rawget(self, key)
        end
    end

    function EncoderParams:load(t)
        local argRules = {
            {"configFile", required = true}
        }
        local configFile = cv.argcheck(t, argRules)

        self.object = C.EncoderParams_ctor(configFile)
    end

    function EncoderParams:save(t)
        local argRules = {
            {"configFile", required = true}
        }
        local configFile = cv.argcheck(t, argRules)

        C.EncoderParams_save(self.object, configFile)
    end
end

return cv
