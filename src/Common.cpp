#include <Common.hpp>
#include <stitching.hpp>

using namespace std;

/***************** Tensor <=> Mat conversion *****************/

// https://github.com/torch/torch7/blob/master/lib/TH/THGeneral.c#L8
#if (defined(__unix) || defined(_WIN32))
#include <malloc.h>
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#endif

static long getAllocSize(void *ptr) {
#if defined(__unix)
    return malloc_usable_size(ptr);
#elif defined(__APPLE__)
    return malloc_size(ptr);
#elif defined(_WIN32)
    return _msize(ptr);
#else
    return 0;
#endif
}

static void *OpenCVMalloc(void */*allocatorContext*/, long size) {
    return cv::fastMalloc(size);
}

static void *OpenCVRealloc(void */*allocatorContext*/, void *ptr, long size) {
    // https://github.com/Itseez/opencv/blob/master/modules/core/src/alloc.cpp#L62
    void *oldMem = ((unsigned char**)ptr)[-1];
    void *newMem = cv::fastMalloc(size);
    memcpy(newMem, oldMem, getAllocSize(oldMem));
    return newMem;
}

static int c = 0;
static void OpenCVFree(void */*allocatorContext*/, void *ptr) {
    std::cout << "Free " << c++ << std::endl;
    cv::fastFree(ptr);
}

static THAllocator OpenCVCompatibleAllocator;

extern "C"
void initAllocator() {
    OpenCVCompatibleAllocator.malloc = OpenCVMalloc;
    OpenCVCompatibleAllocator.realloc = OpenCVRealloc;
    OpenCVCompatibleAllocator.free = OpenCVFree;
}

// for debugging

extern "C"
void refcount(THByteTensor * x) {
    std::cout << "Tensor refcount: " << x->refcount << std::endl;
    std::cout << "Storage address: " << x->storage << std::endl;
    std::cout << "Storage refcount: " << x->storage->refcount << std::endl;
}

MatT::MatT(cv::Mat & mat) {
    this->mat = mat;
    this->tensor = nullptr;
}

MatT::MatT(cv::Mat && mat) {
    new (this) MatT(mat);
}

MatT::MatT() {
    this->tensor = nullptr;
}

TensorWrapper::TensorWrapper(): tensorPtr(nullptr) {}

TensorWrapper::TensorWrapper(cv::Mat & mat) {

    if (mat.empty()) {
        this->tensorPtr = nullptr;
        return;
    }

    this->typeCode = static_cast<char>(mat.depth());

    this->definedInLua = false;

    THByteTensor *outputPtr = THByteTensor_new();

    // Build new storage on top of the Mat

    outputPtr->storage = THByteStorage_newWithDataAndAllocator(
                mat.data,
                mat.step[0] * mat.rows,
                &OpenCVCompatibleAllocator,
                nullptr
        );

    int sizeMultiplier;
    if (mat.channels() == 1) {
        outputPtr->nDimension = mat.dims;
        sizeMultiplier = cv::getElemSize(mat.depth());
    } else {
        outputPtr->nDimension = mat.dims + 1;
        sizeMultiplier = mat.elemSize1();
    }

    outputPtr->size = static_cast<long *>(THAlloc(sizeof(long) * outputPtr->nDimension));
    outputPtr->stride = static_cast<long *>(THAlloc(sizeof(long) * outputPtr->nDimension));

    if (mat.channels() > 1) {
        outputPtr->size[outputPtr->nDimension - 1] = mat.channels();
        outputPtr->stride[outputPtr->nDimension - 1] = 1; //cv::getElemSize(returnValue.typeCode);
    }

    for (int i = 0; i < mat.dims; ++i) {
        outputPtr->size[i] = mat.size[i];
        outputPtr->stride[i] = mat.step[i] / sizeMultiplier;
    }

    // Prevent OpenCV from deallocating Mat memory
    if (mat.u) {
        mat.addref();
    }

    this->tensorPtr = outputPtr;
}

TensorWrapper::TensorWrapper(cv::Mat && mat) {
    // invokes TensorWrapper(cv::Mat & mat)
    new (this) TensorWrapper(mat);
}

TensorWrapper::TensorWrapper(MatT & matT) {

    if (matT.mat.empty()) {
        this->tensorPtr = nullptr;
        return;
    }

    this->typeCode = static_cast<char>(matT.mat.depth());

    if (matT.tensor != nullptr) {
        // Mat is already constructed on another Tensor, so return that
        this->tensorPtr = matT.tensor;
        this->definedInLua = true;
        THAtomicIncrementRef(&this->tensorPtr->storage->refcount);
        return;
    }

    this->definedInLua = false;

    cv::Mat & mat = matT.mat;

    THByteTensor *outputPtr = THByteTensor_new();

    // Build new Storage on top of the Mat
    outputPtr->storage = THByteStorage_newWithDataAndAllocator(
            mat.data,
            mat.step[0] * mat.rows,
            &OpenCVCompatibleAllocator,
            nullptr);

    int sizeMultiplier;
    if (mat.channels() == 1) {
        outputPtr->nDimension = mat.dims;
        sizeMultiplier = cv::getElemSize(mat.depth());
    } else {
        outputPtr->nDimension = mat.dims + 1;
        sizeMultiplier = mat.elemSize1();
    }

    outputPtr->size   = static_cast<long *>(THAlloc(sizeof(long) * outputPtr->nDimension));
    outputPtr->stride = static_cast<long *>(THAlloc(sizeof(long) * outputPtr->nDimension));

    if (mat.channels() > 1) {
        outputPtr->size[outputPtr->nDimension - 1] = mat.channels();
        outputPtr->stride[outputPtr->nDimension - 1] = 1; //cv::getElemSize(returnValue.typeCode);
    }

    for (int i = 0; i < mat.dims; ++i) {
        outputPtr->size[i] = mat.size[i];
        outputPtr->stride[i] = mat.step[i] / sizeMultiplier;
    }

    // Prevent OpenCV from deallocating Mat memory
    if (mat.u) {
        mat.addref();
    }

    this->tensorPtr = outputPtr;
}

TensorWrapper::TensorWrapper(MatT && mat) {
    new (this) TensorWrapper(mat);
}

TensorWrapper::operator cv::Mat() {

    if (this->tensorPtr == nullptr) {
        return cv::Mat();
    }

    int numberOfDims = tensorPtr->nDimension;
    // THTensor stores its dimensions sizes under long *.
    // In a constructor for cv::Mat, we need const int *.
    // We can't guarantee int and long to be equal.
    // So we somehow need to static_cast THTensor sizes.
    // TODO: get rid of array allocation

    std::array<int, 3> size;
    std::copy(tensorPtr->size, tensorPtr->size + tensorPtr->nDimension, size.begin());

    // Same thing for stride values.
    std::array<size_t, 3> stride;
    std::copy(tensorPtr->stride, tensorPtr->stride + tensorPtr->nDimension, stride.begin());

    int depth = this->typeCode;

    // Determine the number of channels.
    int numChannels;
    // cv::Mat() takes stride values in bytes, so we have to multiply by the element size:
    size_t sizeMultiplier = cv::getElemSize(depth);

    if (tensorPtr->nDimension <= 2) {
        // If such tensor is passed, assume that it is single-channel:
        numChannels = 1;
    } else {
        // Otherwise depend on the 3rd dimension:
        numChannels = tensorPtr->size[2];
        numberOfDims = 2;
    }

    std::for_each(stride.begin(),
                  stride.end(),
                  [sizeMultiplier] (size_t & x) { x *= sizeMultiplier; });

    return cv::Mat(
            numberOfDims,
            size.data(),
            CV_MAKE_TYPE(depth, numChannels),
            tensorPtr->storage->data,
            stride.data()
    );
}

MatT TensorWrapper::toMatT() {
    MatT retval;

    if (this->tensorPtr) {
        retval.mat = this->toMat();
        retval.tensor = tensorPtr;
    } else {
        retval.tensor = nullptr;
    }

    return retval;
}

TensorArray::TensorArray(): tensors(nullptr) {}

TensorArray::TensorArray(std::vector<cv::Mat> & matList):
        tensors(static_cast<TensorWrapper *>(malloc(matList.size() * sizeof(TensorWrapper)))),
        size(matList.size())
{
    for (size_t i = 0; i < matList.size(); ++i) {
        // invoke the constructor, memory is already allocated
        new (tensors + i) TensorWrapper(matList[i]);
    }
}

TensorArray::TensorArray(std::vector<MatT> & matList):
        tensors(static_cast<TensorWrapper *>(malloc(matList.size() * sizeof(TensorWrapper)))),
        size(matList.size())
{
    for (size_t i = 0; i < matList.size(); ++i) {
        // invoke the constructor, memory is already allocated
        new (tensors + i) TensorWrapper(matList[i]);
    }
}

TensorArray::operator std::vector<cv::Mat>() {
    std::vector<cv::Mat> retval(this->size);
    for (int i = 0; i < this->size; ++i) {
        // TODO: avoid temporary object
        retval[i] = this->tensors[i].toMat();
    }
    return retval;
}

TensorArray::operator std::vector<MatT>() {
    std::vector<MatT> retval(this->size);
    for (int i = 0; i < this->size; ++i) {
        // TODO: avoid temporary object
        retval[i] = this->tensors[i].toMatT();
    }
    return retval;
}

// Assign `src` data to `dst`.
// `dst` is always supposed to be an empty Tensor
extern "C"
void transfer_tensor(THByteTensor *dst, struct TensorWrapper srcWrapper) {

    THByteTensor *src = srcWrapper.tensorPtr;

    dst->nDimension = src->nDimension;
    dst->refcount = src->refcount;

    dst->storage = src->storage;

    if (!srcWrapper.definedInLua) {
        // Don't let Torch deallocate size and stride arrays
        dst->size = src->size;
        dst->stride = src->stride;
        src->size = nullptr;
        src->stride = nullptr;
        THAtomicIncrementRef(&src->storage->refcount);
        THByteTensor_free(src);
    } else {
        dst->size   = static_cast<long *>(THAlloc(sizeof(long) * dst->nDimension));
        dst->stride = static_cast<long *>(THAlloc(sizeof(long) * dst->nDimension));
        memcpy(dst->size,   src->size,   src->nDimension * sizeof(long));
        memcpy(dst->stride, src->stride, src->nDimension * sizeof(long));
    }
}

/***************** Wrappers for small classes *****************/

MomentsWrapper::MomentsWrapper(const cv::Moments & other) {
    m00 = other.m00; m01 = other.m01; m02 = other.m02; m03 = other.m03; m10 = other.m10;
    m11 = other.m11; m12 = other.m12; m20 = other.m20; m21 = other.m21; m30 = other.m30;
    mu20 = other.mu20; mu11 = other.mu11; mu02 = other.mu02; mu30 = other.mu30;
    mu21 = other.mu21; mu12 = other.mu12; mu03 = other.mu03;
    nu20 = other.nu20; nu11 = other.nu11; nu02 = other.nu02; nu30 = other.nu30;
    nu21 = other.nu21; nu12 = other.nu12; nu03 = other.nu03;
}

DMatchArray::DMatchArray(const std::vector<cv::DMatch> & other) {
    this->size = other.size();
    this->data = static_cast<DMatchWrapper *>(malloc((this->size + 1) * sizeof(DMatchWrapper)));
    memcpy(this->data + 1, other.data(), this->size * sizeof(DMatchWrapper));
}

DMatchArray::operator std::vector<cv::DMatch>() {
    std::vector<cv::DMatch> retval(this->size);
    memcpy(retval.data(), this->data + 1, this->size * sizeof(DMatchWrapper));
    return retval;
}

DMatchArrayOfArrays::DMatchArrayOfArrays(const std::vector<std::vector<cv::DMatch>> & other) {
    this->size = other.size();
    this->data = static_cast<DMatchArray *>(malloc(this->size * sizeof(DMatchArray)));
    for (int i = 0; i < this->size; ++i) {
        new (this->data + i) DMatchArray(other[i]);
    }
}

DMatchArrayOfArrays::operator std::vector<std::vector<cv::DMatch>>() {
    std::vector<std::vector<cv::DMatch>> retval(this->size);
    for (int i = 0; i < this->size; ++i) {
        retval[i] = this->data[i];
    }
    return retval;
}

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
    this->size = v.size();
    this->data = static_cast<KeyPointWrapper *>(
            malloc(sizeof(KeyPointWrapper) * (this->size + 1)));
    for (int i = 0; i < this->size; ++i) {
        this->data[i + 1] = v[i];
    }
}

KeyPointArray::operator std::vector<cv::KeyPoint>()
{
    std::vector<cv::KeyPoint> retval(this->size);
    for (int i = 0; i < this->size; ++i) {
        retval[i] = this->data[i + 1];
    }
    return retval;
}

Vec3dWrapper & Vec3dWrapper::operator=(cv::Vec3d & other) {
   this->v0 = other[0];
   this->v1 = other[1];
   this->v2 = other[2];
   return *this;
}

Vec3dWrapper::Vec3dWrapper(const cv::Vec3d & other) {
   this->v0 = other[0];
   this->v1 = other[1];
   this->v2 = other[2];
}
   

RectWrapper & RectWrapper::operator=(cv::Rect & other) {
    this->x = other.x;
    this->y = other.y;
    this->width = other.width;
    this->height = other.height;
    return *this;
}

RectWrapper::RectWrapper(const cv::Rect & other) {
    this->x = other.x;
    this->y = other.y;
    this->width = other.width;
    this->height = other.height;
}

RotatedRectWrapper::RotatedRectWrapper(const cv::RotatedRect & other) {
    this->center = other.center;
    this->size = other.size;
    this->angle = other.angle;
}

Size2fWrapper::Size2fWrapper(const cv::Size2f & other) {
    this->height = other.height;
    this->width = other.width;
}

TermCriteriaWrapper::TermCriteriaWrapper(cv::TermCriteria && other) {
    this->epsilon = other.epsilon;
    this->maxCount = other.maxCount;
    this->type = other.type;
}

PointWrapper::PointWrapper(const cv::Point & other) {
    this->x = other.x;
    this->y = other.y;
}

PointArray::PointArray(const std::vector<cv::Point> & vec) {
    this->data = static_cast<PointWrapper *>(malloc((vec.size() + 1) * sizeof(PointWrapper)));
    this->size = vec.size();

    for (int i = 0; i < vec.size(); ++i) {
        this->data[i + 1] = vec[i];
    }
}

PointArray::operator std::vector<cv::Point>() {
    std::vector<cv::Point> retval(this->size);
    memcpy(retval.data(), this->data + 1, this->size * sizeof(PointWrapper));
    return retval;
}

Point2fWrapper::Point2fWrapper(const cv::Point2f & other) {
    this->x = other.x;
    this->y = other.y;
}

Point2dWrapper::Point2dWrapper(const cv::Point2d & other) {
    this->x = other.x;
    this->y = other.y;
}

SizeWrapper::SizeWrapper(const cv::Size & other) {
    this->height = other.height;
    this->width = other.width;
}

RectArray::RectArray(const std::vector<cv::Rect> & vec) {
    this->data = static_cast<RectWrapper *>(malloc((vec.size() + 1) * sizeof(RectWrapper)));
    this->size = vec.size();

    for (int i = 0; i < vec.size(); ++i) {
        this->data[i + 1] = vec[i];
    }
}

RectArray::operator std::vector<cv::Rect>() {
    std::vector<cv::Rect> retval(this->size);
    memcpy(retval.data(), this->data + 1, this->size * sizeof(RectWrapper));
    return retval;
}

SizeArray::SizeArray(const std::vector<cv::Size> & vec) {
    this->data = static_cast<SizeWrapper *>(malloc((vec.size() + 1) * sizeof(SizeWrapper)));
    this->size = vec.size();

    for (int i = 0; i < vec.size(); ++i) {
        this->data[i+1] = vec[i];
    }
}

SizeArray::operator std::vector<cv::Size>()
{
    std::vector<cv::Size> retval(this->size);
    memcpy(retval.data(), this->data+1, this->size * sizeof(SizeWrapper));
    return retval;
}

StringArray::StringArray(const std::vector<cv::String> vec)
{
    this->size = vec.size();

    this->data = static_cast<struct StringWrapper *>(malloc(vec.size() * sizeof(struct StringWrapper)));

    for(int i = 0; i < vec.size(); i++) {
        this->data[i].str = static_cast<char *>(malloc((vec[i].length()+1) * sizeof(char)));
        this->data[i].str = vec[i].c_str();
    }
}

StringArray::operator std::vector<cv::String>()
{
    std::vector<cv::String> retval(this->size);
    for(int i = 0; i < this->size; i++){
        retval[i] = this->data[i].str;
    }
    return retval;
}

IntArray::IntArray(const std::vector<int> vec):
        size(vec.size()),
        data(static_cast<int *>(malloc(vec.size() * sizeof(int))))
{
    for(int i = 0; i < vec.size(); i++){
        data[i] = vec[i];
    }
}

FloatArray::FloatArray(const std::vector<float> vec):
        size(vec.size()),
        data(static_cast<float *>(malloc(vec.size() * sizeof(float))))
{
    for(int i = 0; i < vec.size(); i++){
        data[i] = vec[i];
    }
}

DoubleArray::DoubleArray(const std::vector<double> vec):
        size(vec.size()),
        data(static_cast<double *>(malloc(vec.size() * sizeof(double))))
{
    for(int i = 0; i < vec.size(); i++){
        data[i] = vec[i];
    }
}

/***************** Helper functions *****************/

std::vector<MatT> get_vec_MatT(std::vector<cv::Mat> vec_mat) {
    std::vector<MatT> retval(vec_mat.size());
    for(int i = 0; i < vec_mat.size(); i++) retval[i] = vec_mat[i];
    return retval;
}

std::vector<cv::UMat> get_vec_UMat(std::vector<cv::Mat> vec_mat)
{
    std::vector<cv::UMat> retval(vec_mat.size());
    for(int i = 0; i < vec_mat.size(); i++) retval[i] = vec_mat[i].getUMat(cv::ACCESS_RW);
    return retval;
}

std::vector<cv::Mat> get_vec_Mat(std::vector<cv::UMat> vec_umat)
{
std::vector<cv::Mat> retval(vec_umat.size());
for(int i = 0; i < vec_umat.size(); i++) retval[i] = vec_umat[i].getMat(cv::ACCESS_RW);
return retval;
}