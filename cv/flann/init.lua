local cv = require 'cv._env'
require 'cv.Classes'

local flann = {}

local ffi = require 'ffi'

ffi.cdef[[
void IndexParams_dtor(struct PtrWrapper ptr);

struct PtrWrapper KDTreeIndexParams_ctor(int trees);

struct PtrWrapper LinearIndexParams_ctor();

struct PtrWrapper CompositeIndexParams_ctor(
        int trees, int branching, int iterations,
        int centers_init, float cb_index);

struct PtrWrapper AutotunedIndexParams_ctor(
        float target_precision, float build_weight,
        float memory_weight, float sample_fraction);

struct PtrWrapper HierarchicalClusteringIndexParams_ctor(
        int branching, int centers_init, int trees, int leaf_size);

struct PtrWrapper KMeansIndexParams_ctor(
        int branching, int iterations, int centers_init, float cb_index);

struct PtrWrapper LshIndexParams_ctor(
        int table_number, int key_size, int multi_probe_level);

struct PtrWrapper SavedIndexParams_ctor(const char *filename);

struct PtrWrapper SearchParams_ctor(int checks, float eps, bool sorted);

struct PtrWrapper Index_ctor_default();

struct PtrWrapper Index_ctor(
        struct TensorWrapper features, struct PtrWrapper params,
        int distType);

void Index_dtor(struct PtrWrapper ptr);

void Index_build(
        struct PtrWrapper ptr, struct TensorWrapper features,
        struct PtrWrapper params, int distType);

struct TensorArray Index_knnSearch(
        struct PtrWrapper ptr, struct TensorWrapper query, int knn, struct TensorWrapper indices,
        struct TensorWrapper dists, struct PtrWrapper params);

struct TensorArrayPlusInt Index_radiusSearch(
        struct PtrWrapper ptr, struct TensorWrapper query, double radius, int maxResults,
        struct TensorWrapper indices, struct TensorWrapper dists, struct PtrWrapper params);

void Index_save(struct PtrWrapper ptr, const char *filename);

bool Index_load(struct PtrWrapper ptr, struct TensorWrapper features, const char *filename);

void Index_release(struct PtrWrapper ptr);

int Index_getDistance(struct PtrWrapper ptr);

int Index_getAlgorithm(struct PtrWrapper ptr);
]]

local C = ffi.load(cv.libPath('flann'))

do
    local IndexParams = torch.class('flann.IndexParams', flann)
end

do
    local KDTreeIndexParams = torch.class('flann.KDTreeIndexParams', 'flann.IndexParams', flann)

    function KDTreeIndexParams:__init(t)
        local argRules = {
            {"trees", default = 4}
        }
        local trees = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.KDTreeIndexParams_ctor(trees), C.IndexParams_dtor)
    end
end

do
    local LinearIndexParams = torch.class('flann.LinearIndexParams', 'flann.IndexParams', flann)

    function LinearIndexParams:__init(t)
        self.ptr = ffi.gc(C.LinearIndexParams_ctor(), C.IndexParams_dtor)
    end
end

do
    local CompositeIndexParams = torch.class('flann.CompositeIndexParams', 'flann.IndexParams', flann)

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
    local AutotunedIndexParams = torch.class('flann.AutotunedIndexParams', 'flann.IndexParams', flann)

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
    local HierarchicalClusteringIndexParams = torch.class('flann.HierarchicalClusteringIndexParams', 'flann.IndexParams', flann)

    function HierarchicalClusteringIndexParams:__init(t)
        local argRules = {
            {"branching", default = 32},
            {"centers_init", default = FLANN_CENTERS_RANDOM},
            {"trees", default = 4},
            {"leaf_size", default = 100}
        }
        local branching, centers_init, trees, leaf_size = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.HierarchicalClusteringIndexParams_ctor(
            branching, centers_init, trees, leaf_size), C.IndexParams_dtor)
    end
end

do
    local KMeansIndexParams = torch.class('flann.KMeansIndexParams', 'flann.IndexParams', flann)

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
    local LshIndexParams = torch.class('flann.LshIndexParams', 'flann.IndexParams', flann)

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
    local SavedIndexParams = torch.class('flann.SavedIndexParams', 'flann.IndexParams', flann)

    function SavedIndexParams:__init(t)
        local argRules = {
            {"filename", required = true}
        }
        local filename = cv.argcheck(t, argRules)

        self.ptr = ffi.gc(C.SavedIndexParams_ctor(filename), C.IndexParams_dtor)
    end
end

do
    local SearchParams = torch.class('flann.SearchParams', 'flann.IndexParams', flann)

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

local defaultSearchParams = flann.SearchParams{}

do
    local Index = torch.class('flann.Index', flann)

    function Index:__init(t)
        if not t or not (t[1] or t.features) then
            self.ptr = ffi.gc(C.Index_ctor_default(), C.Index_dtor)
        else
            local argRules = {
                {"features", required = true},
                {"params", required = true},
                {"distType", default = cv.FLANN_DIST_L2}
            }
            local features, params, distType = cv.argcheck(t, argRules)

            self.ptr = ffi.gc(C.Index_ctor(cv.wrap_tensor(features), params.ptr, distType), C.Index_dtor)
        end
    end

    function Index:build(t)
        local argRules = {
            {"features", required = true},
            {"params", required = true},
            {"distType", default = cv.FLANN_DIST_L2}
        }
        local features, params, distType = cv.argcheck(t, argRules)

        C.Index_build(self.ptr, cv.wrap_tensor(features), params.ptr, distType)
    end

    function Index:knnSearch(t)
        local argRules = {
            {"query", required = true},
            {"knn", required = true},
            {"indices", default = nil},
            {"dists", default = nil},
            {"params", default = defaultSearchParams}
        }
        local query, knn, indices, dists, params = cv.argcheck(t, argRules)

        local indices, dists = cv.unwrap_tensors(C.Index_knnSearch(self.ptr,
            cv.wrap_tensor(query), knn, cv.wrap_tensor(indices),
            cv.wrap_tensor(dists), params.ptr))
        return indices + 1, dists
    end

    function Index:radiusSearch(t)
        local argRules = {
            {"query", required = true},
            {"radius", required = true},
            {"maxResults", required = true},
            {"indices", default = nil},
            {"dists", default = nil},
            {"params", default = defaultSearchParams}
        }
        local query, radius, maxResults, indices, dists, params = cv.argcheck(t, argRules)

        local result = C.Index_radiusSearch(self.ptr,
            cv.wrap_tensor(query), radius, maxResults,
            cv.wrap_tensor(indices), cv.wrap_tensor(dists), params.ptr)
        local indices, dists = cv.unwrap_tensors(result.tensors)
        return result.val, indices + 1, dists
    end

    function Index:save(t)
        local argRules = {
            {"filename", required = true}
        }
        local filename = cv.argcheck(t, argRules)

        C.Index_save(self.ptr, filename)
    end

    function Index:load(t)
        local argRules = {
            {"features", required = true},
            {"filename", required = true}
        }
        local features, filename = cv.argcheck(t, argRules)

        return C.Index_load(self.ptr, cv.wrap_tensor(features), filename)
    end

    function Index:release()
        C.Index_release(self.ptr)
    end

    function Index:getDistance()
        return C.Index_getDistance(self.ptr)
    end

    function Index:getAlgorithm()
        return C.Index_getAlgorithm(self.ptr)
    end
end

return flann
