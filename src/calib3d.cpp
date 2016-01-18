#include <calib3d.hpp>

extern "C"
double calibrateCamera(
	struct TensorArray objectPoints, struct TensorArray imagePoints,
	struct SizeWrapper imageSize, struct TensorWrapper cameraMatrix,
	struct TensorWrapper distCoeffs, struct TensorArray rvecs,
	struct TensorArray tvecs, int flags, struct TermCriteriaWrapper criteria)
{   
    return cv::calibrateCamera(objectPoints.toMatList(), imagePoints.toMatList(),
			       imageSize, cameraMatrix.toMat(), distCoeffs.toMat(),
			       rvecs.toMatList(), tvecs.toMatList(), flags, criteria);
}

extern "C"
struct TensorWrapper calibrationMatrixValues(
	struct TensorWrapper cameraMatrix,
	struct SizeWrapper imageSize,
	double apertureWidth, double apertureHeight)
{
    cv::Mat retval(6,1,CV_64FC1);
    cv::Point2d principalPoint_temp;

    cv::calibrationMatrixValues(cameraMatrix.toMat(), imageSize, apertureWidth,
				apertureHeight, retval.at<double>(0,0),
				retval.at<double>(1,0), retval.at<double>(2,0),
				principalPoint_temp, retval.at<double>(5,0));

    retval.at<double>(3,0) = principalPoint_temp.x;
    retval.at<double>(4,0) = principalPoint_temp.y;

    return TensorWrapper(retval);
}

extern "C"
void composeRT(
	struct TensorWrapper rvec1, struct TensorWrapper tvec1, struct TensorWrapper rvec2,
	struct TensorWrapper tvec2, struct TensorWrapper rvec3, struct TensorWrapper tvec3,
	struct TensorWrapper dr3dr1, struct TensorWrapper dr3dt1, struct TensorWrapper dr3dr2,
	struct TensorWrapper dr3dt2, struct TensorWrapper dt3dr1, struct TensorWrapper dt3dt1,
	struct TensorWrapper dt3dr2, struct TensorWrapper dt3dt2)
{
    cv::composeRT(rvec1.toMat(),tvec1.toMat(), rvec2.toMat(),
		  tvec2.toMat(), rvec3.toMat(), tvec3.toMat(),
		  TO_MAT_OR_NOARRAY(dr3dr1), TO_MAT_OR_NOARRAY(dr3dt1),
 		  TO_MAT_OR_NOARRAY(dr3dr2), TO_MAT_OR_NOARRAY(dr3dt2),
		  TO_MAT_OR_NOARRAY(dt3dr1), TO_MAT_OR_NOARRAY(dt3dt1),
  		  TO_MAT_OR_NOARRAY(dt3dr2), TO_MAT_OR_NOARRAY(dt3dt2));
}

extern "C"
struct TensorWrapper computeCorrespondEpilines(
	struct TensorWrapper points, int whichImage, struct TensorWrapper F)
{
    cv::Mat lines;
    cv::computeCorrespondEpilines(points.toMat(), whichImage, F.toMat(),lines);
    return TensorWrapper(lines);
}

extern "C" 
struct TensorWrapper convertPointsFromHomogeneous(
	struct TensorWrapper src)
{
    cv::Mat dst;
    cv::convertPointsFromHomogeneous(src.toMat(),dst);
    return TensorWrapper(dst);
}  

extern "C"
struct TensorWrapper convertPointsHomogeneous(
	struct TensorWrapper src, struct TensorWrapper dst)
{
    cv::convertPointsHomogeneous(src.toMat(), dst.toMat());
    return dst;
}

extern "C" 
struct TensorWrapper convertPointsToHomogeneous(
	struct TensorWrapper src)
{
    cv::Mat dst;
    cv::convertPointsToHomogeneous(src.toMat(),dst);
    return TensorWrapper(dst);
}

extern "C"
struct TensorArray correctMatches(
	struct TensorWrapper F, struct TensorWrapper points1,
	struct TensorWrapper points2)
{
    std::vector<cv::Mat> vecPoints(2);
    cv::correctMatches(F.toMat(), points1.toMat(), points2.toMat(),
                       vecPoints[0],vecPoints[1]);
    
    return TensorArray(vecPoints);
}

extern "C"
struct TensorArray decomposeEssentialMat(
	struct TensorWrapper E)
{
    std::vector<cv::Mat> vec(3);
    cv::decomposeEssentialMat(E.toMat(), vec[0],  vec[1], vec[2]);  
    return TensorArray(vec);
}

extern "C"
struct TensorArrayPlusInt decomposeHomographyMat(
	struct TensorWrapper H, struct TensorWrapper K)
{
    struct TensorArrayPlusInt result;
    std::vector<cv::Mat> vec(3);
    result.val = cv::decomposeHomographyMat(H.toMat(), K.toMat(),
 					 vec[0], vec[1], vec[2]);

    new(&result.tensors) TensorArray(vec);
    return result;
}

extern "C"
struct TensorArray decomposeProjectionMatrix(
	struct TensorWrapper projMatrix, struct TensorWrapper rotMatrixX,
	struct TensorWrapper rotMatrixY, struct TensorWrapper rotMatrixZ,
	struct TensorWrapper eulerAngles)
{
    std::vector<cv::Mat> vec(3);
    cv::decomposeProjectionMatrix(projMatrix.toMat(), vec[0], vec[1], vec[2],
			          TO_MAT_OR_NOARRAY(projMatrix), TO_MAT_OR_NOARRAY(rotMatrixY),
				  TO_MAT_OR_NOARRAY(rotMatrixZ), TO_MAT_OR_NOARRAY(eulerAngles));
    return TensorArray(vec);
}

extern "C"
void drawChessboardCorners(
	struct TensorWrapper image, struct SizeWrapper patternSize,
	struct TensorWrapper corners, bool patternWasFound)
{
    cv::Mat imgMat(image);
    cv::drawChessboardCorners(imgMat, patternSize, corners.toMat(), patternWasFound);
}

extern "C"
struct TensorArrayPlusInt estimateAffine3D(
	struct TensorWrapper src, struct TensorWrapper dst,
	double ransacThreshold, double confidence)
{
    struct TensorArrayPlusInt result;
    std::vector<cv::Mat> vec(2); 
    result.val = cv::estimateAffine3D(src.toMat(), dst.toMat(), vec[0],
                                      vec[1], ransacThreshold, confidence);
    new(&result.tensors) TensorArray(vec);
    return result;
}

extern "C"
void filterSpeckles(
	struct TensorWrapper img, double newVal, int maxSpeckleSize,
	double maxDiff, struct TensorWrapper buf)
{
    cv::filterSpeckles(img.toMat(), newVal, maxSpeckleSize,
                       maxDiff, TO_MAT_OR_NOARRAY(buf));
}

extern "C"
void find4QuadCornerSubpix(
	struct TensorWrapper img, struct TensorWrapper corners,
	struct SizeWrapper region_size)
{
    cv::find4QuadCornerSubpix(img.toMat(), corners.toMat(), region_size);
}

extern "C"
struct TensorWrapper findChessboardCorners(
	struct TensorWrapper image, struct SizeWrapper patternSize, int flags)
{
    cv::Mat corners;
    cv::findChessboardCorners(image.toMat(), patternSize, corners, flags);
    return TensorWrapper(corners);
}

//TODO const Ptr<FeatureDetector>& blobDetector = SimpleBlobDetector::create()
extern "C"
struct TensorPlusBool findCirclesGrid(
	struct TensorWrapper image, struct SizeWrapper patternSize, int flags)
{
    struct TensorPlusBool result;
    cv::Mat centers;
    result.val = cv::findCirclesGrid(image.toMat(), patternSize, centers, flags);
    new(&result.tensor) TensorWrapper(centers);
    return result;
}

struct TensorWrapper findEssentialMat(
	struct TensorWrapper points1, struct TensorWrapper points2,
	double focal, struct Point2dWrapper pp, int method, double prob,
	double threshold, struct TensorWrapper mask)
{
    return TensorWrapper(cv::findEssentialMat(points1.toMat(), points2.toMat(),
					      focal, pp, method, prob, threshold,
					      TO_MAT_OR_NOARRAY(mask)));
}

extern "C"
struct TensorWrapper findFundamentalMat(
	struct TensorWrapper points1, struct TensorWrapper points2, int method,
	double param1, double param2, struct TensorWrapper mask)
{
    return TensorWrapper(cv::findFundamentalMat(points1.toMat(), points1.toMat(),
						method, param1, param2,
						TO_MAT_OR_NOARRAY(mask)));
}

extern "C"
struct TensorWrapper findFundamentalMat2(
	struct TensorWrapper points1, struct TensorWrapper points2,
	struct TensorWrapper mask, int method, double param1, double param2)
{
    return TensorWrapper(cv::findFundamentalMat(points1.toMat(), points1.toMat(),
						mask.toMat(), method, param1, param2));
}
 
extern "C"
struct TensorWrapper findHomography(
	struct TensorWrapper srcPoints, struct TensorWrapper dstPoints,
	int method, double ransacReprojThreshold, struct TensorWrapper mask,
	const int maxIters, const double confidence)
{
    return TensorWrapper(cv::findHomography(srcPoints.toMat(), dstPoints.toMat(),
					    method, ransacReprojThreshold,
					    TO_MAT_OR_NOARRAY(mask),
					    maxIters, confidence));
}

extern "C"
struct TensorWrapper findHomography2(
	struct TensorWrapper srcPoints, struct TensorWrapper dstPoints,
	struct TensorWrapper mask, int method, double ransacReprojThreshold)
{
    return TensorWrapper(cv::findHomography(srcPoints.toMat(), dstPoints.toMat(),
			                    mask.toMat(), method,
                                            ransacReprojThreshold));
}






