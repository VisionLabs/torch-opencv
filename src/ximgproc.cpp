#include <ximgproc.hpp>

extern "C"
struct TensorWrapper niBlackThreshold(struct TensorWrapper src, struct TensorWrapper dst, double maxValue, int type, int blockSize, double delta) {

    MatT dst_mat = dst.toMatT();
    cv::ximgproc::niBlackThreshold(src.toMat(), dst_mat, maxValue, type, blockSize, delta);
    return TensorWrapper(dst_mat);
}

// GraphSegmentation

extern "C"
struct GraphSegmentationPtr GraphSegmentation_ctor(double sigma, float k, int min_size) {
    return rescueObjectFromPtr(cv::ximgproc::segmentation::createGraphSegmentation(sigma, k, min_size));
}

extern "C"
void GraphSegmentation_dtor(struct GraphSegmentationPtr ptr) {
    delete static_cast<cv::ximgproc::segmentation::GraphSegmentation *>(ptr.ptr);
}

extern "C"
struct TensorWrapper GraphSegmentation_processImage(struct GraphSegmentationPtr ptr, struct TensorWrapper img) {

    cv::Mat result;
    ptr->processImage(img.toMat(), result);
    return TensorWrapper(result);
}

extern "C"
void GraphSegmentation_setSigma(struct GraphSegmentationPtr ptr, double s) {
    ptr->setSigma(s);
}

extern "C"
double GraphSegmentation_getSigma(struct GraphSegmentationPtr ptr) {
    return ptr->getSigma();
}

extern "C"
void GraphSegmentation_setK(struct GraphSegmentationPtr ptr, float k) {
    ptr->setK(k);
}

extern "C"
float GraphSegmentation_getK(struct GraphSegmentationPtr ptr) {
    return ptr->getK();
}

extern "C"
void GraphSegmentation_setMinSize(struct GraphSegmentationPtr ptr, int min_size) {
    ptr->setMinSize(min_size);
}

extern "C"
int GraphSegmentation_getMinSize(struct GraphSegmentationPtr ptr) {
    return ptr->getMinSize();
}

// SLIC

extern "C"
struct SuperpixelSLICPtr SuperpixelSLIC_ctor(
        struct TensorWrapper image, int algorithm,
        int region_size, float ruler) {
    return rescueObjectFromPtr(cv::ximgproc::createSuperpixelSLIC(
            image.toMat(), algorithm, region_size, ruler));
}

extern "C"
int SuperpixelSLIC_getNumberOfSuperpixels(struct SuperpixelSLICPtr ptr) {
    return ptr->getNumberOfSuperpixels();
}

extern "C"
void SuperpixelSLIC_iterate(struct SuperpixelSLICPtr ptr, int num_iterations) {
    ptr->iterate(num_iterations);
}

extern "C"
struct TensorWrapper SuperpixelSLIC_getLabels(
        struct SuperpixelSLICPtr ptr, struct TensorWrapper labels_out) {
    MatT labels_outMat = labels_out.toMatT();
    ptr->getLabels(labels_outMat);
    return labels_outMat;
}

extern "C"
struct TensorWrapper SuperpixelSLIC_getLabelContourMask(
        struct SuperpixelSLICPtr ptr, struct TensorWrapper image, bool thick_line) {
    MatT imageMat = image.toMatT();
    ptr->getLabelContourMask(imageMat, thick_line);
    return imageMat;
}

extern "C"
void SuperpixelSLIC_enforceLabelConnectivity(
        struct SuperpixelSLICPtr ptr, int min_element_size) {
    ptr->enforceLabelConnectivity(min_element_size);
}

// See #103 and #95
/*

// SelectiveSearchSegmentation

extern "C"
struct SelectiveSearchSegmentationPtr SelectiveSearchSegmentation_ctor() {
    return rescueObjectFromPtr(cv::ximgproc::segmentation::createSelectiveSearchSegmentation());
}

extern "C"
void SelectiveSearchSegmentationPtr_dtor(struct SelectiveSearchSegmentationPtr ptr) {
    delete static_cast<cv::ximgproc::segmentation::SelectiveSearchSegmentation *>(ptr.ptr);
}

extern "C"
void SelectiveSearchSegmentation_setBaseImage(struct SelectiveSearchSegmentationPtr ptr, struct TensorWrapper img) {
    ptr->setBaseImage(img.toMat());
}

extern "C"
void SelectiveSearchSegmentation_switchToSingleStrategy(struct SelectiveSearchSegmentationPtr ptr, int k, float sigma) {
    ptr->switchToSingleStrategy(k, sigma);
}

extern "C"
void SelectiveSearchSegmentation_switchToSelectiveSearchFast(struct SelectiveSearchSegmentationPtr ptr, int k, int inc_k, float sigma) {
    ptr->switchToSelectiveSearchFast(k, inc_k, sigma);
}

extern "C"
void SelectiveSearchSegmentation_switchToSelectiveSearchQuality(struct SelectiveSearchSegmentationPtr ptr, int k, int inc_k, float sigma) {
    ptr->switchToSelectiveSearchQuality(k, inc_k, sigma);
}

extern "C"
void SelectiveSearchSegmentation_addImage(struct SelectiveSearchSegmentationPtr ptr, struct TensorWrapper img) {
    ptr->setBaseImage(img.toMat());
}

extern "C"
void SelectiveSearchSegmentation_clearImages(struct SelectiveSearchSegmentationPtr ptr) {
    ptr->clearImages();
}

extern "C"
void SelectiveSearchSegmentation_addGraphSegmentation(struct SelectiveSearchSegmentationPtr ptr, struct GraphSegmentationPtr gs) {
    ptr->addGraphSegmentation(*gs);
}

extern "C"
void SelectiveSearchSegmentation_clearGraphSegmentations(struct SelectiveSearchSegmentationPtr ptr) {
    ptr->clearGraphSegmentations();
}

extern "C"
void SelectiveSearchSegmentation_addStrategy(struct SelectiveSearchSegmentationPtr ptr, struct SelectiveSearchSegmentationStrategyPtr s) {
    ptr->addStrategy(*s);
}

extern "C"
void SelectiveSearchSegmentation_clearStrategies(struct SelectiveSearchSegmentationPtr ptr) {
    ptr->clearStrategies();
}

extern "C"
struct RectArray SelectiveSearchSegmentation_process(struct SelectiveSearchSegmentationPtr ptr) {

    std::vector<cv::Rect> result;
    ptr->process(result);

    return RectArray(result);
}
*/