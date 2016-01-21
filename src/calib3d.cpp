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
struct TensorWrapper find4QuadCornerSubpix(
	struct TensorWrapper img, struct TensorWrapper corners,
	struct SizeWrapper region_size)
{
    cv::Mat corners_mat;
    if(!corners.isNull()) corners_mat = corners.toMat();
    cv::find4QuadCornerSubpix(img.toMat(), corners_mat, region_size);
    return TensorWrapper(corners_mat);
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

extern "C"
struct TensorPlusRect getOptimalNewCameraMatrix(
	struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
	struct SizeWrapper imageSize, double alpha, struct SizeWrapper newImgSize,
	bool centerPrincipalPoint)
{
    struct TensorPlusRect result;
    cv::Rect* validPixROI;
    new(&result.tensor) TensorWrapper(
				cv::getOptimalNewCameraMatrix(
						cameraMatrix.toMat(), distCoeffs.toMat(),
	                                        imageSize, alpha, newImgSize,
                                                validPixROI, centerPrincipalPoint));
    result.rect = *validPixROI;
    return result;
}

extern "C"
struct RectWrapper getValidDisparityROI(
	struct RectWrapper roi1, struct RectWrapper roi2,
	int minDisparity, int numberOfDisparities, int SADWindowSize)
{
    return RectWrapper(cv::getValidDisparityROI(roi1, roi2, minDisparity,
                                                numberOfDisparities, SADWindowSize));
}

extern "C"
struct TensorWrapper initCameraMatrix2D(
	struct TensorArray objectPoints, struct TensorArray imagePoints,
   	struct SizeWrapper imageSize, double aspectRatio)
{
    return TensorWrapper(cv::initCameraMatrix2D(
					objectPoints.toMatList(), imagePoints.toMatList(),
					imageSize, aspectRatio));
}

extern "C"
struct TensorArray matMulDeriv(
	struct TensorWrapper A, struct TensorWrapper B)
{
    std::vector<cv::Mat> result(2);
    cv::matMulDeriv(A.toMat(), B.toMat(), result[0], result[1]);
    return TensorArray(result);
}

extern "C"
struct TensorArray projectPoints(
	struct TensorWrapper objectPoints, struct TensorWrapper rvec,
	struct TensorWrapper tvec, struct TensorWrapper cameraMatrix,
	struct TensorWrapper distCoeffs, struct TensorWrapper imagePoints,
	struct TensorWrapper jacobian, double aspectRatio)
{
    std::vector<cv::Mat> result;
    if(!imagePoints.isNull()) result[0] = imagePoints.toMat();
    if(!jacobian.isNull()) result[1] = jacobian.toMat();
    cv::projectPoints(objectPoints.toMat(), rvec.toMat(), tvec.toMat(),
                      cameraMatrix.toMat(), distCoeffs.toMat(), result[0],
                      result[1], aspectRatio);
    return TensorArray(result);
}

extern "C"
struct TensorArrayPlusInt recoverPose(
	struct TensorWrapper E, struct TensorWrapper points1,
	struct TensorWrapper points2, double focal,
	struct Point2dWrapper pp, struct TensorWrapper mask)
{
    struct TensorArrayPlusInt result;
    std::vector<cv::Mat> vec(2);
    result.val = cv::recoverPose(E.toMat(), points1.toMat(), points2.toMat(), vec[0],
                                 vec[1], focal, pp, TO_MAT_OR_NOARRAY(mask));
    new(&result.tensors) TensorArray(vec);
    return result;
}

extern "C"
struct TensorArrayPlusRectArrayPlusFloat rectify3Collinear(
	struct TensorWrapper cameraMatrix1, struct TensorWrapper distCoeffs1,
	struct TensorWrapper cameraMatrix2, struct TensorWrapper distCoeffs2,
	struct TensorWrapper cameraMatrix3, struct TensorWrapper distCoeffs3,
	struct TensorArray imgpt1, struct TensorArray imgpt3,
	struct SizeWrapper imageSize, struct TensorWrapper R12,
	struct TensorWrapper T12, struct TensorWrapper R13,
	struct TensorWrapper T13, double alpha,
	struct SizeWrapper newImgSize, int flags)
{
    struct TensorArrayPlusRectArrayPlusFloat result;
    std::vector<cv::Mat> vec(7);
    std::vector<cv::Rect> rec(2);
    result.val = cv::rectify3Collinear(
			cameraMatrix1.toMat(), distCoeffs1.toMat(), cameraMatrix2.toMat(),
			distCoeffs2.toMat(), cameraMatrix3.toMat(), distCoeffs3.toMat(),
			imgpt1.toMatList(), imgpt3.toMatList(), imageSize, R12.toMat(),
			T12.toMat(), R13.toMat(), T13.toMat(), vec[0], vec[1], vec[2],
			vec[3], vec[4], vec[5], vec[6], alpha, newImgSize,
			&rec[0], &rec[1], flags);
    new(&result.tensors) TensorArray(vec);
    new(&result.rects) RectArray(rec);
    return result;
}

extern "C"
struct TensorWrapper reprojectImageTo3D(
	struct TensorWrapper disparity, struct TensorWrapper _3dImage,
	struct TensorWrapper Q, bool handleMissingValues, int ddepth)
{
    cv::Mat _3dImage_mat;
    if(!_3dImage.isNull()) _3dImage_mat = _3dImage.toMat();
    cv::reprojectImageTo3D(disparity.toMat(), _3dImage_mat, Q.toMat(), handleMissingValues, ddepth);
    return TensorWrapper(_3dImage_mat);
}

extern "C"
struct TensorArray Rodrigues(
	struct TensorWrapper src, struct TensorWrapper dst, struct TensorWrapper jacobian)
{
    std::vector<cv::Mat> result;
    if(!dst.isNull()) result[0] = dst.toMat();
    if(!dst.isNull()) result[1] = jacobian.toMat();
    cv::Rodrigues(src.toMat(), result[0], result[1]);
    return TensorArray(result);
}

extern "C"
struct TensorArrayPlusVec3d RQDecomp3x3(
	struct TensorWrapper src, struct TensorWrapper mtxR, struct TensorWrapper mtxQ,
	struct TensorWrapper Qx, struct TensorWrapper Qy, struct TensorWrapper Qz)
{
    struct TensorArrayPlusVec3d result;
    std::vector<cv::Mat> vec(5);
    if(!mtxR.isNull()) vec[0] = mtxR.toMat();
    if(!mtxQ.isNull()) vec[1] = mtxQ.toMat();
    if(!Qx.isNull()) vec[2] = Qx.toMat();
    if(!Qy.isNull()) vec[3] = Qy.toMat();
    if(!Qz.isNull()) vec[4] = Qz.toMat();
    new(&result.vec3d) Vec3dWrapper(
				cv::RQDecomp3x3(
					src.toMat(), vec[0], vec[1],
					vec[2], vec[3], vec[4]));
    new(&result.tensors) TensorArray(vec);
    return result;
}

extern "C"
struct TensorArrayPlusBool solvePnP(
	struct TensorWrapper objectPoints, struct TensorWrapper imagePoints,
	struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
	struct TensorWrapper rvec, struct TensorWrapper tvec,
	bool useExtrinsicGuess, int flags)
{
    struct TensorArrayPlusBool result;
    std::vector<cv::Mat> vec(2);
    if(!rvec.isNull()) vec[0] = rvec.toMat();
    if(!tvec.isNull()) vec[1] = tvec.toMat();
    result.val = cv::solvePnP(objectPoints.toMat(), imagePoints.toMat(), cameraMatrix.toMat(),
                              distCoeffs.toMat(), vec[0], vec[1], useExtrinsicGuess, flags);
    new(&result.tensors) TensorArray(vec);
    return result;
}

extern "C"
struct TensorArrayPlusBool solvePnPRansac(
	struct TensorWrapper objectPoints, struct TensorWrapper imagePoints,
	struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
	struct TensorWrapper rvec, struct TensorWrapper tvec,
	bool useExtrinsicGuess, int iterationsCount, float reprojectionError,
	double confidence, struct TensorWrapper inliers, int flags)
{
    struct TensorArrayPlusBool result;
    std::vector<cv::Mat> vec(3);
    if(!rvec.isNull()) vec[0] = rvec.toMat();
    if(!tvec.isNull()) vec[1] = tvec.toMat();
    if(!inliers.isNull()) vec[2] = inliers.toMat();
    result.val = cv::solvePnPRansac(objectPoints.toMat(), imagePoints.toMat(),
                 cameraMatrix.toMat(), distCoeffs.toMat(), vec[0], vec[1],
                 useExtrinsicGuess, iterationsCount, reprojectionError,
                 confidence, vec[2], flags);
    new(&result.tensors) TensorArray(vec);
    return result;
}

extern "C"
double stereoCalibrate(
	struct TensorWrapper objectPoints, struct TensorWrapper imagePoints1,
	struct TensorWrapper imagePoints2, struct TensorWrapper cameraMatrix1,
	struct TensorWrapper distCoeffs1, struct TensorWrapper cameraMatrix2,
	struct TensorWrapper distCoeffs2, struct SizeWrapper imageSize,
	struct TensorWrapper R, struct TensorWrapper T,
	struct TensorWrapper E, struct TensorWrapper F,
	int flags, struct TermCriteriaWrapper criteria)
{
    return cv::stereoCalibrate(
		objectPoints.toMat(), imagePoints1.toMat(), imagePoints2.toMat(),
		cameraMatrix1.toMat(), distCoeffs1.toMat(), cameraMatrix2.toMat(),
		distCoeffs2.toMat(), imageSize, R.toMat(), T.toMat(), E.toMat(),
		F.toMat(), flags, criteria);
}

extern "C"
struct RectArray stereoRectify(
	struct TensorWrapper cameraMatrix1, struct TensorWrapper distCoeffs1,
	struct TensorWrapper cameraMatrix2, struct TensorWrapper distCoeffs2,
	struct SizeWrapper imageSize, struct TensorWrapper R,
	struct TensorWrapper T, struct TensorWrapper R1,
	struct TensorWrapper R2, struct TensorWrapper P1,
	struct TensorWrapper P2, struct TensorWrapper Q,
	int flags, double alpha,struct SizeWrapper newImageSize)
{
    std::vector<cv::Rect> rec(2);
    cv::stereoRectify(
		cameraMatrix1.toMat(), distCoeffs1.toMat(),
		cameraMatrix2.toMat(), distCoeffs2.toMat(),
		imageSize, R.toMat(), T.toMat(), R1.toMat(),
		R2.toMat(), P1.toMat(), P2.toMat(), Q.toMat(),
		flags, alpha, newImageSize, &rec[0], &rec[1]);
    return RectArray(rec);
}

extern "C"
struct TensorArrayPlusBool stereoRectifyUncalibrated(
	struct TensorWrapper points1, struct TensorWrapper points2,
	struct TensorWrapper F, struct SizeWrapper imgSize,
	struct TensorWrapper H1, struct TensorWrapper H2, double threshold)
{
    struct TensorArrayPlusBool result;
    std::vector<cv::Mat> vec(2);
    if(!H1.isNull()) vec[0] = H1.toMat();
    if(!H2.isNull()) vec[1] = H2.toMat();
    cv::stereoRectifyUncalibrated(
		points1.toMat(), points2.toMat(), F.toMat(),
		imgSize, vec[0], vec[1], threshold);
    new(&result.tensors) TensorArray(vec);    
    return result;
}

extern "C"
struct TensorWrapper triangulatePoints(
	struct TensorWrapper projMatr1, struct TensorWrapper projMatr2,
	struct TensorWrapper projPoints1, struct TensorWrapper projPoints2)
{
    cv::Mat points4D;
    cv::triangulatePoints(projMatr1.toMat(), projMatr2.toMat(),
                          projPoints1.toMat(), projPoints2.toMat(), points4D);
    return TensorWrapper(points4D);
}

extern "C"
struct TensorWrapper validateDisparity(
	struct TensorWrapper disparity, struct TensorWrapper cost,
        int minDisparity, int numberOfDisparities, int disp12MaxDisp)
{
   cv::validateDisparity(
		disparity.toMat(), cost.toMat(), minDisparity,
		numberOfDisparities, disp12MaxDisp);
    return disparity;
}



