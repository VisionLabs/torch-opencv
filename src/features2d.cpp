#include <features2d.hpp>

KeyPointWrapper::KeyPointWrapper(const cv::KeyPoint & other) {
    this->pt = other.pt;
    this->size = other.size;
    this->angle = other.angle;
    this->response = other.response;
    this->octave = other.octave;
    this->class_id = other.class_id;
}

KeyPointArray::KeyPointArray(const std::vector<cv::KeyPoint> & v)
{
    // TODO: IMPORTANT! Prevent memory leak here
    this->size = v.size();
    this->data = static_cast<KeyPointWrapper *>(
            malloc(sizeof(KeyPointWrapper) * this->size));
    for (int i = 0; i < this->size; ++i) {
        this->data[i] = v[i];
    }
}

KeyPointArray::operator std::vector<cv::KeyPoint>()
{
    std::vector<cv::KeyPoint> retval(this->size);
    for (int i = 0; i < this->size; ++i) {
        retval[i] = this->data[i];
    }
    return retval;
}

KeyPointMat::KeyPointMat(const std::vector<std::vector<cv::KeyPoint> > & v)
{
    // TODO: this function
    this->size1 = v.size();
    this->size2 = v[0].size();
}

KeyPointMat::operator std::vector<std::vector<cv::KeyPoint> >()
{
    // TODO: this function
    std::vector<std::vector<cv::KeyPoint> > retval;
    return retval;
}

// KeyPointsFilter

extern "C" struct KeyPointsFilterPtr KeyPointsFilter_ctor()
{
    return new cv::KeyPointsFilter();
}

extern "C" void KeyPointsFilter_dtor(struct KeyPointsFilterPtr ptr)
{
    delete static_cast<cv::KeyPointsFilter *>(ptr.ptr);
}

extern "C" struct KeyPointArray KeyPointsFilter_runByImageBorder(struct KeyPointArray keypoints,
                    struct SizeWrapper imageSize, int borderSize)
{
    std::vector<cv::KeyPoint> keypointsVector(keypoints);
    cv::KeyPointsFilter::runByImageBorder(keypointsVector, imageSize, borderSize);
    return KeyPointArray(keypointsVector);
}

extern "C" struct KeyPointArray KeyPointsFilter_runByKeypointSize(struct KeyPointArray keypoints,
                        float minSize, float maxSize)
{
    std::vector<cv::KeyPoint> keypointsVector(keypoints);
    cv::KeyPointsFilter::runByKeypointSize(keypointsVector, minSize, maxSize);
    return KeyPointArray(keypointsVector);
}

extern "C" struct KeyPointArray KeyPointsFilter_runByPixelsMask(struct KeyPointArray keypoints,
                        struct TensorWrapper mask)
{
    std::vector<cv::KeyPoint> keypointsVector(keypoints);
    cv::KeyPointsFilter::runByPixelsMask(keypointsVector, mask.toMat());
    return KeyPointArray(keypointsVector);
}

extern "C" struct KeyPointArray KeyPointsFilter_removeDuplicated(struct KeyPointArray keypoints)
{
    std::vector<cv::KeyPoint> keypointsVector(keypoints);
    cv::KeyPointsFilter::removeDuplicated(keypointsVector);
    return KeyPointArray(keypointsVector);
}

extern "C" struct KeyPointArray KeyPointsFilter_retainBest(struct KeyPointArray keypoints, int npoints)
{
    std::vector<cv::KeyPoint> keypointsVector(keypoints);
    cv::KeyPointsFilter::retainBest(keypointsVector, npoints);
    return KeyPointArray(keypointsVector);
}

// Feature2D

extern "C" struct KeyPointArray Feature2D_detect(struct Feature2DPtr ptr, struct TensorWrapper image,
                        struct KeyPointArray keypoints, struct TensorWrapper mask)
{
    std::vector<cv::KeyPoint> keypointsVector(keypoints);
    ptr->detect(image.toMat(), keypointsVector, mask.toMat());
    return KeyPointArray(keypointsVector);
}

extern "C" struct KeyPointMat Feature2D_detect2(struct Feature2DPtr ptr, struct TensorArray images,
                        struct KeyPointMat keypoints, struct TensorArray masks)
{
    std::vector<std::vector<cv::KeyPoint> > keypointsMat(keypoints);
    ptr->detect(images.toMatList(), keypointsMat, masks.toMatList());
    return KeyPointMat(keypointsMat);
}

extern "C" struct KeyPointArray Feature2D_compute(struct Feature2DPtr ptr, struct TensorWrapper image,
                        struct KeyPointArray keypoints, struct TensorWrapper descriptors)
{
    std::vector<cv::KeyPoint> keypointsVector(keypoints);
    ptr->compute(image.toMat(), keypointsVector, descriptors.toMat());
    return KeyPointArray(keypointsVector);
}

extern "C" struct KeyPointMat Feature2D_compute2(struct Feature2DPtr ptr, struct TensorArray images,
                        struct KeyPointMat keypoints, struct TensorArray descriptors)
{
    std::vector<std::vector<cv::KeyPoint> > keypointsMat(keypoints);
    ptr->compute(images.toMatList(), keypointsMat, descriptors.toMatList());
    return KeyPointMat(keypointsMat);
}

extern "C" struct KeyPointArray Feature2D_detectAndCompute(struct Feature2DPtr ptr, struct TensorWrapper image,
                        struct TensorWrapper mask, struct KeyPointArray keypoints,
                        struct TensorWrapper descriptors, bool useProvidedKeypoints)
{
    std::vector<cv::KeyPoint> keypointsVector(keypoints);
    ptr->detectAndCompute(image.toMat(), mask.toMat(), keypointsVector, descriptors.toMat(), useProvidedKeypoints);
    return KeyPointArray(keypointsVector);
}

extern "C" int Feature2D_descriptorSize(struct Feature2DPtr ptr)
{
    return ptr->descriptorSize();
}

extern "C" int Feature2D_descriptorType(struct Feature2DPtr ptr)
{
    return ptr->descriptorType();
}

extern "C" int Feature2D_defaultNorm(struct Feature2DPtr ptr)
{
    return ptr->defaultNorm();
}

extern "C" bool Feature2D_empty(struct Feature2DPtr ptr)
{
    return ptr->empty();
}





extern "C" struct KeyPointArray AGAST(struct TensorWrapper image, int threshold, bool nonmaxSuppression)
{
    std::vector<cv::KeyPoint> result;
    cv::AGAST(image.toMat(), result, threshold, nonmaxSuppression);
    return KeyPointArray(result);
}