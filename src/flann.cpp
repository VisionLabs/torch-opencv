#include <flann.hpp>

// Various IndexParams

extern "C"
void IndexParams_dtor(struct IndexParamsPtr ptr)
{
    delete static_cast<flann::IndexParams *>(ptr.ptr);
}

extern "C"
struct KDTreeIndexParamsPtr KDTreeIndexParams_ctor(int trees)
{
    return new flann::KDTreeIndexParams(trees);
}

extern "C"
struct LinearIndexParamsPtr LinearIndexParams_ctor()
{
    return new flann::LinearIndexParams();
}

extern "C"
struct CompositeIndexParamsPtr CompositeIndexParams_ctor(
        int trees, int branching, int iterations,
        cvflann::flann_centers_init_t centers_init, float cb_index)
{
    return new flann::CompositeIndexParams(
            trees, branching, iterations, centers_init, cb_index);
}

extern "C"
struct AutotunedIndexParamsPtr AutotunedIndexParams_ctor(
        float target_precision, float build_weight,
        float memory_weight, float sample_fraction)
{
    return new flann::AutotunedIndexParams(
            target_precision, build_weight, memory_weight, sample_fraction);
}

extern "C"
struct HierarchicalClusteringIndexParamsPtr HierarchicalClusteringIndexParams_ctor(
        int branching, cvflann::flann_centers_init_t centers_init, int trees, int leaf_size)
{
    return new flann::HierarchicalClusteringIndexParams(
            branching, centers_init, trees, leaf_size);
}

extern "C"
struct KMeansIndexParamsPtr KMeansIndexParams_ctor(
        int branching, int iterations, cvflann::flann_centers_init_t centers_init, float cb_index)
{
    return new flann::KMeansIndexParams(branching, iterations, centers_init, cb_index);
}

extern "C"
struct LshIndexParamsPtr LshIndexParams_ctor(
        int table_number, int key_size, int multi_probe_level)
{
    return new flann::LshIndexParams(table_number, key_size, multi_probe_level);
}

extern "C"
struct SavedIndexParamsPtr SavedIndexParams_ctor(const char *filename)
{
    return new flann::SavedIndexParams(filename);
}

extern "C"
struct SearchParamsPtr SearchParams_ctor(int checks, float eps, bool sorted)
{
    return new flann::SearchParams(checks, eps, sorted);
}

// Index

extern "C"
struct IndexPtr Index_ctor_default()
{
    return new flann::Index();
}

extern "C"
struct IndexPtr Index_ctor(
        struct TensorWrapper features, struct IndexParamsPtr params,
        cvflann::flann_distance_t distType)
{
    return new flann::Index(features.toMat(), params, distType);
}

extern "C"
void Index_dtor(struct IndexPtr ptr)
{
    delete static_cast<flann::Index *>(ptr.ptr);
}

extern "C"
void Index_build(
        struct IndexPtr ptr, struct TensorWrapper features,
        struct IndexParamsPtr params, cvflann::flann_distance_t distType)
{
    ptr->build(features.toMat(), params, distType);
}

extern "C"
struct TensorArray Index_knnSearch(
        struct IndexPtr ptr, struct TensorWrapper query, int knn, struct TensorWrapper indices,
        struct TensorWrapper dists, struct SearchParamsPtr params)
{
    std::vector<cv::Mat> retval(2);
    if (!indices.isNull()) retval[0] = indices;
    if (!dists.isNull())   retval[1] = dists;
    
    ptr->knnSearch(query.toMat(), retval[0], retval[1], knn, params);
    
    return TensorArray(retval);
}

extern "C"
struct TensorArrayPlusInt Index_radiusSearch(
        struct IndexPtr ptr, struct TensorWrapper query, double radius, int maxResults,
        struct TensorWrapper indices, struct TensorWrapper dists, struct SearchParamsPtr params)
{
    struct TensorArrayPlusInt retval;
    
    std::vector<cv::Mat> result(2);
    if (!indices.isNull()) result[0] = indices;
    if (!dists.isNull())   result[1] = dists;
    
    retval.val = ptr->radiusSearch(
            query.toMat(), result[0], result[1], radius, maxResults, params);
    
    new (&retval.tensors) TensorArray(result);
    
    return retval;
}

extern "C"
void Index_save(struct IndexPtr ptr, const char *filename)
{
    ptr->save(filename);
}

extern "C"
bool Index_load(struct IndexPtr ptr, struct TensorWrapper features, const char *filename)
{
    return ptr->load(features.toMat(), filename);
}

extern "C"
void Index_release(struct IndexPtr ptr)
{
    ptr->release();
}

extern "C"
int Index_getDistance(struct IndexPtr ptr)
{
    return ptr->getDistance();
}

extern "C"
int Index_getAlgorithm(struct IndexPtr ptr)
{
    return ptr->getAlgorithm();
}
