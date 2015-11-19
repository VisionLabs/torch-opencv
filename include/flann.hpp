#include <Common.hpp>
#include <opencv2/flann.hpp>

namespace flann = cv::flann;

struct IndexPtr {
    void *ptr;

    inline flann::Index * operator->() { return static_cast<flann::Index *>(ptr); }
    inline IndexPtr(flann::Index *ptr) { this->ptr = ptr; }
};

// Various IndexParams

struct IndexParamsPtr {
    void *ptr;

    inline flann::IndexParams * operator->() { return static_cast<flann::IndexParams *>(ptr); }
    inline IndexParamsPtr(flann::IndexParams *ptr) { this->ptr = ptr; }
    inline operator const flann::IndexParams &() { return *static_cast<flann::IndexParams *>(ptr); }
};

struct KDTreeIndexParamsPtr {
    void *ptr;

    inline flann::KDTreeIndexParams * operator->() { return static_cast<flann::KDTreeIndexParams *>(ptr); }
    inline KDTreeIndexParamsPtr(flann::KDTreeIndexParams *ptr) { this->ptr = ptr; }
};

struct LinearIndexParamsPtr {
    void *ptr;

    inline flann::LinearIndexParams * operator->() { return static_cast<flann::LinearIndexParams *>(ptr); }
    inline LinearIndexParamsPtr(flann::LinearIndexParams *ptr) { this->ptr = ptr; }
};

struct CompositeIndexParamsPtr {
    void *ptr;

    inline flann::CompositeIndexParams * operator->() { return static_cast<flann::CompositeIndexParams *>(ptr); }
    inline CompositeIndexParamsPtr(flann::CompositeIndexParams *ptr) { this->ptr = ptr; }
};

struct AutotunedIndexParamsPtr {
    void *ptr;

    inline flann::AutotunedIndexParams * operator->() { return static_cast<flann::AutotunedIndexParams *>(ptr); }
    inline AutotunedIndexParamsPtr(flann::AutotunedIndexParams *ptr) { this->ptr = ptr; }
};

struct HierarchicalClusteringIndexParamsPtr {
    void *ptr;

    inline flann::HierarchicalClusteringIndexParams * operator->() { return static_cast<flann::HierarchicalClusteringIndexParams *>(ptr); }
    inline HierarchicalClusteringIndexParamsPtr(flann::HierarchicalClusteringIndexParams *ptr) { this->ptr = ptr; }
};

struct KMeansIndexParamsPtr {
    void *ptr;

    inline flann::KMeansIndexParams * operator->() { return static_cast<flann::KMeansIndexParams *>(ptr); }
    inline KMeansIndexParamsPtr(flann::KMeansIndexParams *ptr) { this->ptr = ptr; }
};

struct LshIndexParamsPtr {
    void *ptr;

    inline flann::LshIndexParams * operator->() { return static_cast<flann::LshIndexParams *>(ptr); }
    inline LshIndexParamsPtr(flann::LshIndexParams *ptr) { this->ptr = ptr; }
};

struct SavedIndexParamsPtr {
    void *ptr;

    inline flann::SavedIndexParams * operator->() { return static_cast<flann::SavedIndexParams *>(ptr); }
    inline SavedIndexParamsPtr(flann::SavedIndexParams *ptr) { this->ptr = ptr; }
};

struct SearchParamsPtr {
    void *ptr;

    inline flann::SearchParams * operator->() { return static_cast<flann::SearchParams *>(ptr); }
    inline SearchParamsPtr(flann::SearchParams *ptr) { this->ptr = ptr; }
    inline operator const flann::SearchParams &() { return *static_cast<const flann::SearchParams *>(ptr); }
};

extern "C"
void IndexParams_dtor(struct IndexParamsPtr ptr);

extern "C"
struct KDTreeIndexParamsPtr KDTreeIndexParams_ctor(int trees);

extern "C"
struct LinearIndexParamsPtr LinearIndexParams_ctor();

extern "C"
struct CompositeIndexParamsPtr CompositeIndexParams_ctor(
        int trees, int branching, int iterations,
        cvflann::flann_centers_init_t centers_init, float cb_index);

extern "C"
struct AutotunedIndexParamsPtr AutotunedIndexParams_ctor(
        float target_precision, float build_weight,
        float memory_weight, float sample_fraction);

extern "C"
struct HierarchicalClusteringIndexParamsPtr HierarchicalClusteringIndexParams_ctor(
        int branching, cvflann::flann_centers_init_t centers_init, int trees, int leaf_size);

extern "C"
struct KMeansIndexParamsPtr KMeansIndexParams_ctor(
        int branching, int iterations, cvflann::flann_centers_init_t centers_init, float cb_index);

extern "C"
struct LshIndexParamsPtr LshIndexParams_ctor(
        int table_number, int key_size, int multi_probe_level);

extern "C"
struct SavedIndexParamsPtr SavedIndexParams_ctor(const char *filename);

extern "C"
struct SearchParamsPtr SearchParams_ctor(int checks, float eps, bool sorted);

extern "C"
struct IndexPtr Index_ctor_default();

extern "C"
struct IndexPtr Index_ctor(
        struct TensorWrapper features, struct IndexParamsPtr params,
        cvflann::flann_distance_t distType);

extern "C"
void Index_dtor(struct IndexPtr ptr);

extern "C"
void Index_build(
        struct IndexPtr ptr, struct TensorWrapper features,
        struct IndexParamsPtr params, cvflann::flann_distance_t distType);

extern "C"
struct TensorArray Index_knnSearch(
        struct IndexPtr ptr, struct TensorWrapper query, int knn, struct TensorWrapper indices,
        struct TensorWrapper dists, struct SearchParamsPtr params);

extern "C"
struct TensorArrayPlusInt Index_radiusSearch(
        struct IndexPtr ptr, struct TensorWrapper query, double radius, int maxResults,
        struct TensorWrapper indices, struct TensorWrapper dists, struct SearchParamsPtr params);

extern "C"
void Index_save(struct IndexPtr ptr, const char *filename);

extern "C"
bool Index_load(struct IndexPtr ptr, struct TensorWrapper features, const char *filename);

extern "C"
void Index_release(struct IndexPtr ptr);

extern "C"
int Index_getDistance(struct IndexPtr ptr);

extern "C"
int Index_getAlgorithm(struct IndexPtr ptr);
