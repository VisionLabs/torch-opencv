#include <flann.hpp>

struct IndexPtr {
    void *ptr;

    inline flann::Index * operator->() { return static_cast<flann::Index *>(ptr); }
    inline IndexPtr(flann::Index *ptr) { this->ptr = ptr; }
};


struct IndexParamsPtr {
    void *ptr;

    inline flann::IndexParams * operator->() { return static_cast<flann::IndexParams *>(ptr); }
    inline IndexParamsPtr(flann::IndexParams *ptr) { this->ptr = ptr; }
    inline operator const flann::IndexParams &() { return *static_cast<const flann::IndexParams *>(ptr); }
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
};


struct IndexPtr Index_ctor_default()
{
    return new flann::Index();
}

struct IndexPtr Index_ctor(
        struct TensorWrapper features, struct IndexParamsPtr params,
        cvflann::flann_distance_t distType)
{
    return new flann::Index(features.toMat(), params, distType);
}

void Index_dtor(struct IndexPtr ptr)
{
    delete static_cast<flann::Index *>(ptr.ptr);
}