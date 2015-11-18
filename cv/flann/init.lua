local cv = require 'cv._env'
local ffi = require 'ffi'

ffi.cdef[[

]]

local C = ffi.load(cv.libPath('flann'))

do
    local IndexParams = cv.newTorchClass('cv.IndexParams')
end

do
    local KDTreeIndexParams = cv.newTorchClass('cv.KDTreeIndexParams', 'cv.IndexParams')

    function KDTreeIndexParams:__init(t)
        local argRules = {
            {"trees", default = 4}
        }
        local trees = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.KDTreeIndexParams_ctor(trees), C.IndexParams_dtor)
    end
end

do
    local LinearIndexParams = cv.newTorchClass('cv.LinearIndexParams', 'cv.IndexParams')

    function LinearIndexParams:__init(t)
        self.ptr = ffi.gc(C.LinearIndexParams_ctor(), C.IndexParams_dtor)
    end
end

do
    local CompositeIndexParams = cv.newTorchClass('cv.CompositeIndexParams', 'cv.IndexParams')

    function CompositeIndexParams:__init(t)
        local argRules = {
            {"trees", default = 4},
            {"branching", default = 32},
            {"iterations", default = 11},
            {"centers_init", default = FLANN_CENTERS_RANDOM},
            {"cb_index", default = 0.2}
        }
        local trees, branching, iterations, centers_init, cb_index = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.CompositeIndexParams_ctor(
            trees, branching, iterations, centers_init, cb_index), C.IndexParams_dtor)
    end
end

do
    local AutotunedIndexParams = cv.newTorchClass('cv.AutotunedIndexParams', 'cv.IndexParams')

    function AutotunedIndexParams:__init(t)
        local argRules = {
            {"target_precision", default = 0.8},
            {"build_weight", default = 0.01},
            {"memory_weight", default = 0},
            {"sample_fraction", default = 0.1}
        }
        local target_precision, build_weight, memory_weight, sample_fraction = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.AutotunedIndexParams_ctor(
            target_precision, build_weight, memory_weight, sample_fraction), C.IndexParams_dtor)
    end
end

do
    local HierarchicalClusteringIndexParams = cv.newTorchClass('cv.HierarchicalClusteringIndexParams', 'cv.IndexParams')

    function HierarchicalClusteringIndexParams:__init(t)
        local argRules = {
            {"branching", default = 32},
            {"centers_init", default = FLANN_CENTERS_RANDOM},
            {"trees", default = 4},
            {"leaf_size", default = 100},
        }
        local branching, centers_init, trees, leaf_size = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.HierarchicalClusteringIndexParams_ctor(
            branching, centers_init, trees, leaf_size), C.IndexParams_dtor)
    end
end

do
    local KMeansIndexParams = cv.newTorchClass('cv.KMeansIndexParams', 'cv.IndexParams')

    function KMeansIndexParams:__init(t)
        local argRules = {
            {"branching", default = 32},
            {"iterations", default = 11},
            {"centers_init", default = FLANN_CENTERS_RANDOM},
            {"cb_index", default = 0.2}
        }
        local branching, iterations, centers_init, cb_index = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.KMeansIndexParams_ctor(
            branching, iterations, centers_init, cb_index), C.IndexParams_dtor)
    end
end

do
    local LshIndexParams = cv.newTorchClass('cv.LshIndexParams', 'cv.IndexParams')

    function LshIndexParams:__init(t)
        local argRules = {
            {"table_number", required = true},
            {"key_size", required = true},
            {"multi_probe_level", required = true}
        }
        local table_number, key_size, multi_probe_level = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.LshIndexParams_ctor(
            table_number, key_size, multi_probe_level), C.IndexParams_dtor)
    end
end

do
    local SavedIndexParams = cv.newTorchClass('cv.SavedIndexParams', 'cv.IndexParams')

    function SavedIndexParams:__init(t)
        local argRules = {
            {"filename", required = true}
        }
        local filename = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.SavedIndexParams_ctor(filename), C.IndexParams_dtor)
    end
end

do
    local SearchIndexParams = cv.newTorchClass('cv.SearchParams', 'cv.IndexParams')

    function SearchParams:__init(t)
        local argRules = {
            {"checks", default = 32},
            {"eps", default = 0},
            {"sorted", default = true}
        }
        local checks, eps, sorted = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.SearchParams_ctor(
            checks, eps, sorted), C.IndexParams_dtor)
    end
end

do
    local Index = cv.newTorchClass('cv.Index')

    function Index:__init(t)
        if not (t[1] or t.features) then
            self.ptr = ffi.gc(C.Index_ctor_default(), C.Index_dtor)
        else
            local argRules = {
                {"features", required = true},
                {"params", required = true},
                {"distType", default = cv.FLANN_DIST_L2}
            }
            local features, params, distType = cv.argcheck(t, argRules)

            self.ptr = ffi.gc(C.Index_ctor(cv.wrap_tensor(features), params, distType), C.Index_dtor)
        end
    end
end

return cv
