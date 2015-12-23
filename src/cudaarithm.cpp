#include <cudaarithm.hpp>

using namespace std;
#define pr(v) cout << #v << " = " << v << "\n";

extern "C"
struct TensorWrapper min(
        struct THCState *state, struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper dst)
{
    if (dst.isNull()) {
        cuda::GpuMat result;

        std::cout << "here" << std::endl;
        cuda::min(src1.toGpuMat(), src2.toGpuMat(), result);

        cv::Mat temp;
        result.download(temp);
        //std::cout << temp << std::endl;
        cout << "temp step,rows: " << temp.step << " " << temp.rows << endl;
        cout << "result step,rows: " << result.step << " " << result.rows << endl;
        cout << "is result continuous: " << result.isContinuous() << endl;
        cout << "is src1 continuous: " << src1.toGpuMat().isContinuous() << endl;
        cuda::GpuMat res2(1, 160, CV_32F, result.data, 160 * sizeof(float));
        res2.download(temp);
        cout << "is res2 continuous: " << res2.isContinuous() << endl;
        cout << temp << endl;
        cout << "res2 refcount " << res2.refcount << endl;
        result.refcount = nullptr;

        return TensorWrapper(result, state);
    } else {
        cuda::min(src1.toGpuMat(), src2.toGpuMat(), dst.toGpuMat());
        return dst;
    }
}