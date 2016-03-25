#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/ximgproc.hpp>


extern "C"
struct TensorWrapper niBlackThreshold(struct TensorWrapper src, struct TensorWrapper dst, double maxValue, int type, int blockSize, double delta);


// GraphSegmentation
struct GraphSegmentationPtr {
    void *ptr;

    inline cv::ximgproc::segmentation::GraphSegmentation * operator->() { return static_cast<cv::ximgproc::segmentation::GraphSegmentation *>(ptr); }
    inline cv::ximgproc::segmentation::GraphSegmentation * operator*() { return static_cast<cv::ximgproc::segmentation::GraphSegmentation *>(ptr); }
    inline GraphSegmentationPtr(cv::ximgproc::segmentation::GraphSegmentation *ptr) { this->ptr = ptr; }
};

// GraphSegmentation
extern "C"
struct GraphSegmentationPtr GraphSegmentation_ctor(double sigma, float k, int min_size);

extern "C"
struct TensorWrapper GraphSegmentation_processImage(struct GraphSegmentationPtr ptr, struct TensorWrapper);

extern "C"
void GraphSegmentation_setSigma(struct GraphSegmentationPtr ptr, double s);

extern "C"
double GraphSegmentation_getSigma(struct GraphSegmentationPtr ptr);

extern "C"
void GraphSegmentation_setK(struct GraphSegmentationPtr ptr, float k);

extern "C"
float GraphSegmentation_getK(struct GraphSegmentationPtr ptr);

extern "C"
void GraphSegmentation_setMinSize(struct GraphSegmentationPtr ptr, int min_size);

extern "C"
int GraphSegmentation_getMinSize(struct GraphSegmentationPtr ptr);


// SelectiveSearchSegmentationStrategy
struct SelectiveSearchSegmentationStrategyPtr {
    void *ptr;

    inline cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategy * operator->() { return static_cast<cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategy *>(ptr); }
    inline cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategy * operator*() { return static_cast<cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategy *>(ptr); }
    inline SelectiveSearchSegmentationStrategyPtr(cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategy *ptr) { this->ptr = ptr; }
};
//
// extern "C"
// struct SelectiveSearchSegmentationStrategyPtr SelectiveSearchSegmentationStrategyColor_ctor();
//
// extern "C"
// struct SelectiveSearchSegmentationStrategyPtr SelectiveSearchSegmentationStrategySize_ctor();
//
// extern "C"
// struct SelectiveSearchSegmentationStrategyPtr SelectiveSearchSegmentationStrategyTexture_ctor();
//
// extern "C"
// struct SelectiveSearchSegmentationStrategyPtr SelectiveSearchSegmentationStrategyFill_ctor();
//
// extern "C"
// void SelectiveSearchSegmentationStrategy_setImage(struct SelectiveSearchSegmentationStrategyPtr ptr, struct TensorWrapper, struct TensorWrapper, struct TensorWrapper, int);
//
// extern "C"
// float SelectiveSearchSegmentationStrategy_get(int, int);
//
// extern "C"
// void SelectiveSearchSegmentationStrategy_merge(int, int);


// MULTIPLE STRTEGY
//

// See #103
#ifndef APPLE

// SelectiveSearchSegmentation
struct SelectiveSearchSegmentationPtr {
    void *ptr;

    inline cv::ximgproc::segmentation::SelectiveSearchSegmentation * operator->() { return static_cast<cv::ximgproc::segmentation::SelectiveSearchSegmentation *>(ptr); }
    inline cv::ximgproc::segmentation::SelectiveSearchSegmentation * operator*() { return static_cast<cv::ximgproc::segmentation::SelectiveSearchSegmentation *>(ptr); }
    inline SelectiveSearchSegmentationPtr(cv::ximgproc::segmentation::SelectiveSearchSegmentation *ptr) { this->ptr = ptr; }
};

extern "C"
struct SelectiveSearchSegmentationPtr SelectiveSearchSegmentation_ctor();

extern "C"
void SelectiveSearchSegmentation_setBaseImage(struct SelectiveSearchSegmentationPtr ptr, struct TensorWrapper);

extern "C"
void SelectiveSearchSegmentation_switchToSingleStrategy(struct SelectiveSearchSegmentationPtr ptr, int, float);

extern "C"
void SelectiveSearchSegmentation_switchToSelectiveSearchFast(struct SelectiveSearchSegmentationPtr ptr, int, int, float);

extern "C"
void SelectiveSearchSegmentation_switchToSelectiveSearchQuality(struct SelectiveSearchSegmentationPtr ptr, int, int, float);

extern "C"
void SelectiveSearchSegmentation_addImage(struct SelectiveSearchSegmentationPtr ptr, struct TensorWrapper);

extern "C"
void SelectiveSearchSegmentation_clearImages(struct SelectiveSearchSegmentationPtr ptr);

extern "C"
void SelectiveSearchSegmentation_addGraphSegmentation(struct SelectiveSearchSegmentationPtr ptr, struct GraphSegmentationPtr);

extern "C"
void SelectiveSearchSegmentation_clearGraphSegmentations(struct SelectiveSearchSegmentationPtr ptr);

extern "C"
void SelectiveSearchSegmentation_addStrategy(struct SelectiveSearchSegmentationPtr ptr, struct SelectiveSearchSegmentationStrategyPtr);

extern "C"
void SelectiveSearchSegmentation_clearStrategies(struct SelectiveSearchSegmentationPtr ptr);

extern "C"
struct RectArray SelectiveSearchSegmentation_process(struct SelectiveSearchSegmentationPtr ptr);

#endif
