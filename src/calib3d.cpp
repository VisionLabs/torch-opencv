#include <calib3d.hpp>

extern "C"
struct calibrateCameraRetval calibrateCamera(
	struct TensorArray objectPoints, struct TensorArray imagePoints,
	struct SizeWrapper imageSize, struct TensorWrapper cameraMatrix,
	struct TensorWrapper distCoeffs, struct TensorArray rvecs,
	struct TensorArray tvecs, int flags, struct TermCriteriaWrapper criteria)
{   
    struct calibrateCameraRetval result;
    std::vector<cv::Mat> intrinsics(2), rvecs_vec, tvecs_vec;

    if(!cameraMatrix.isNull()) intrinsics[0] = cameraMatrix.toMat();
    if(!distCoeffs.isNull()) intrinsics[1] = distCoeffs.toMat();
    if(!rvecs.isNull()) rvecs_vec = rvecs.toMatList();
    if(!tvecs.isNull()) tvecs_vec = tvecs.toMatList();

    result.retval = cv::calibrateCamera(objectPoints.toMatList(), imagePoints.toMatList(),
			       		imageSize, intrinsics[0], intrinsics[1],
			       		rvecs_vec, tvecs_vec, flags, criteria);
    new(&result.intrinsics) TensorArray(intrinsics);
    new(&result.rvecs) TensorArray(rvecs_vec);
    new(&result.tvecs) TensorArray(tvecs_vec);
    return result;
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
struct TensorArray composeRT(
	struct TensorWrapper rvec1, struct TensorWrapper tvec1, struct TensorWrapper rvec2,
	struct TensorWrapper tvec2, struct TensorWrapper rvec3, struct TensorWrapper tvec3,
	struct TensorWrapper dr3dr1, struct TensorWrapper dr3dt1, struct TensorWrapper dr3dr2,
	struct TensorWrapper dr3dt2, struct TensorWrapper dt3dr1, struct TensorWrapper dt3dt1,
	struct TensorWrapper dt3dr2, struct TensorWrapper dt3dt2)
{
    std::vector<cv::Mat> vec(10);
    if(!rvec3.isNull()) vec[0] = rvec3.toMat();
    if(!tvec3.isNull()) vec[1] = tvec3.toMat();
    if(!dr3dr1.isNull()) vec[2] = dr3dr1.toMat();
    if(!dr3dt1.isNull()) vec[3] = dr3dt1.toMat();
    if(!dr3dr2.isNull()) vec[4] = dr3dr2.toMat();
    if(!dr3dt2.isNull()) vec[5] = dr3dt2.toMat();
    if(!dt3dr1.isNull()) vec[6] = dt3dr1.toMat();
    if(!dt3dt1.isNull()) vec[7] = dt3dt1.toMat();
    if(!dt3dr2.isNull()) vec[8] = dt3dr2.toMat();
    if(!dt3dt2.isNull()) vec[9] = dt3dt2.toMat();

    cv::composeRT(rvec1.toMat(),tvec1.toMat(), rvec2.toMat(),
		  tvec2.toMat(), vec[0], vec[1], vec[2], vec[3],
 		  vec[4], vec[5], vec[6], vec[7], vec[8], vec[9]);
    return TensorArray(vec);
}

extern "C"
struct TensorWrapper computeCorrespondEpilines(
	struct TensorWrapper points, int whichImage, struct TensorWrapper F,
	struct TensorWrapper lines)
{
    cv::Mat lines_mat;
    if(!lines.isNull()) lines_mat = lines.toMat();
    cv::computeCorrespondEpilines(points.toMat(), whichImage, F.toMat(),lines_mat);
    return TensorWrapper(lines_mat);
}

extern "C" 
struct TensorWrapper convertPointsFromHomogeneous(
	struct TensorWrapper src, struct TensorWrapper dst)
{
    cv::Mat dst_mat;
    if(!dst.isNull()) dst_mat = dst.toMat();
    cv::convertPointsFromHomogeneous(src.toMat(),dst_mat);
    return TensorWrapper(dst_mat);
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
	struct TensorWrapper src, struct TensorWrapper dst)
{
    cv::Mat dst_mat;
    if(!dst.isNull()) dst_mat = dst.toMat();
    cv::convertPointsToHomogeneous(src.toMat(),dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorArray correctMatches(
	struct TensorWrapper F, struct TensorWrapper points1,
	struct TensorWrapper points2, struct TensorWrapper newPoints1,
	struct TensorWrapper newPoints2)
{
    std::vector<cv::Mat> vec(2);
    if(!newPoints1.isNull()) vec[0] = newPoints1.toMat();
    if(!newPoints2.isNull()) vec[1] = newPoints2.toMat();
    cv::correctMatches(F.toMat(), points1.toMat(), points2.toMat(),
                       vec[0],vec[1]);
    
    return TensorArray(vec);
}

extern "C"
struct TensorArray decomposeEssentialMat(
	struct TensorWrapper E, struct TensorWrapper R1,
	struct TensorWrapper R2, struct TensorWrapper t)
{
    std::vector<cv::Mat> vec(3);
    if(!R1.isNull()) vec[0] = R1.toMat();
    if(!R2.isNull()) vec[1] = R2.toMat();
    if(!t.isNull()) vec[2] = t.toMat();
    cv::decomposeEssentialMat(E.toMat(), vec[0],  vec[1], vec[2]);  
    return TensorArray(vec);
}

extern "C"
struct decomposeHomographyMatRetval decomposeHomographyMat(
	struct TensorWrapper H, struct TensorWrapper K,
	struct TensorArray rotations, struct TensorArray translations,
	struct TensorArray normals)
{
    struct decomposeHomographyMatRetval result;
    std::vector<cv::Mat> rotations_vec, translations_vec, normals_vec;

    if(!rotations.isNull()) rotations_vec = rotations.toMatList();
    if(!translations.isNull()) translations_vec = translations.toMatList();
    if(!normals.isNull()) normals_vec = normals.toMatList();

    result.val = cv::decomposeHomographyMat(H.toMat(), K.toMat(), rotations_vec,
					    translations_vec, normals_vec);

    new(&result.rotations) TensorArray(rotations_vec);
    new(&result.translations) TensorArray(translations_vec);
    new(&result.normals) TensorArray(normals_vec);
    return result;
}

extern "C"
struct TensorArray decomposeProjectionMatrix(
	struct TensorWrapper projMatrix, struct TensorWrapper cameraMatrix,
	struct TensorWrapper rotMatrix, struct TensorWrapper transVect,
	struct TensorWrapper rotMatrixX, struct TensorWrapper rotMatrixY,
	struct TensorWrapper rotMatrixZ, struct TensorWrapper eulerAngles)
{
    std::vector<cv::Mat> vec(7);

    if(!cameraMatrix.isNull()) vec[0] = cameraMatrix.toMat();
    if(!rotMatrix.isNull()) vec[1] = rotMatrix.toMat();
    if(!transVect.isNull()) vec[2] = transVect.toMat();
    if(!rotMatrixX.isNull()) vec[3] = rotMatrixX.toMat();
    if(!rotMatrixY.isNull()) vec[4] = rotMatrixY.toMat();
    if(!rotMatrixZ.isNull()) vec[5] = rotMatrixZ.toMat();
    if(!eulerAngles.isNull()) vec[6] = eulerAngles.toMat();

    cv::decomposeProjectionMatrix(
		projMatrix.toMat(), vec[0], vec[1], vec[2],
		vec[3], vec[4], vec[5], vec[6]);
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
	struct TensorWrapper out, struct TensorWrapper inliers,
	double ransacThreshold, double confidence)
{
    struct TensorArrayPlusInt result;
    std::vector<cv::Mat> vec(2);
    if(!out.isNull()) vec[0] = out.toMat();
    if(!inliers.isNull()) vec[1] = inliers.toMat();
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
	struct TensorWrapper image, struct SizeWrapper patternSize,
	struct TensorWrapper corners, int flags)
{
    cv::Mat corners_mat;
    if(!corners.isNull()) corners_mat = corners.toMat();
    cv::findChessboardCorners(image.toMat(), patternSize, corners_mat, flags);
    return TensorWrapper(corners_mat);
}

extern "C"
struct TensorPlusBool findCirclesGrid(
	struct TensorWrapper image, struct SizeWrapper patternSize,
	struct TensorWrapper centers, int flags, struct SimpleBlobDetectorPtr blobDetector)
{
    cv::Ptr<cv::FeatureDetector> blobDetectorPtr(static_cast<cv::SimpleBlobDetector *>(blobDetector.ptr));
    rescueObjectFromPtr(blobDetectorPtr);

    struct TensorPlusBool result;
    cv::Mat centers_mat;
    if(!centers.isNull()) centers_mat = centers.toMat();

    result.val = cv::findCirclesGrid(
		image.toMat(), patternSize, centers_mat, flags, blobDetectorPtr);
    new(&result.tensor) TensorWrapper(centers_mat);
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
struct TensorArray findFundamentalMat2(
	struct TensorWrapper points1, struct TensorWrapper points2,
	struct TensorWrapper mask, int method, double param1, double param2)
{
    std::vector<cv::Mat> vec(2);
    if(!mask.isNull()) vec[1] = mask.toMat();
    vec[0] = cv::findFundamentalMat(
			points1.toMat(), points1.toMat(),
			vec[1], method, param1, param2);
    return TensorArray(vec);
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
struct TensorArray findHomography2(
	struct TensorWrapper srcPoints, struct TensorWrapper dstPoints,
	struct TensorWrapper mask, int method, double ransacReprojThreshold)
{
    std::vector<cv::Mat> vec(2);
    if(!mask.isNull()) vec[1] = mask.toMat();
    vec[0] = cv::findHomography(
			srcPoints.toMat(), dstPoints.toMat(),
			vec[1], method, ransacReprojThreshold);
    return TensorArray(vec);
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
	struct TensorWrapper A, struct TensorWrapper B,
	struct TensorWrapper dABdA, struct TensorWrapper dABdB)
{
    std::vector<cv::Mat> result(2);
    if(!dABdA.isNull()) result[0] = dABdA.toMat();
    if(!dABdB.isNull()) result[1] = dABdB.toMat();
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
	struct TensorWrapper points2, struct TensorWrapper R,
	struct TensorWrapper t, double focal,
	struct Point2dWrapper pp, struct TensorWrapper mask)
{
    struct TensorArrayPlusInt result;
    std::vector<cv::Mat> vec(2);
    if(!R.isNull()) vec[0] = R;
    if(!t.isNull()) vec[1] = t;
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
struct TensorArrayPlusDouble stereoCalibrate(
	struct TensorWrapper objectPoints, struct TensorWrapper imagePoints1,
	struct TensorWrapper imagePoints2, struct TensorWrapper cameraMatrix1,
	struct TensorWrapper distCoeffs1, struct TensorWrapper cameraMatrix2,
	struct TensorWrapper distCoeffs2, struct SizeWrapper imageSize,
	struct TensorWrapper R, struct TensorWrapper T,
	struct TensorWrapper E, struct TensorWrapper F,
	int flags, struct TermCriteriaWrapper criteria)
{
    struct TensorArrayPlusDouble result;
    std::vector<cv::Mat> vec(8);
    
    if(!cameraMatrix1.isNull()) vec[0] = cameraMatrix1.toMat();
    if(!distCoeffs1.isNull()) vec[1] = distCoeffs1.toMat();
    if(!cameraMatrix2.isNull()) vec[2] = cameraMatrix2.toMat();
    if(!distCoeffs2.isNull()) vec[3] = distCoeffs2.toMat();
    if(!R.isNull()) vec[4] = R.toMat();
    if(!T.isNull()) vec[5] = T.toMat();
    if(!E.isNull()) vec[6] = E.toMat();
    if(!F.isNull()) vec[7] = F.toMat();

    result.val = cv::stereoCalibrate(
			objectPoints.toMat(), imagePoints1.toMat(),
			imagePoints2.toMat(), vec[0], vec[1], vec[2],
			vec[3], imageSize, vec[4], vec[5], vec[6],
			vec[7], flags, criteria);
    new(&result.tensors) TensorArray(vec);
    return result;
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

//******************Fisheye camera model***************

extern "C"
struct calibrateCameraRetval fisheye_calibrate(
	struct TensorArray objectPoints, struct TensorArray imagePoints,
	struct SizeWrapper imageSize, struct TensorWrapper K,
	struct TensorWrapper D, struct TensorArray rvecs,
	struct TensorArray tvecs, int flags, struct TermCriteriaWrapper criteria)
{   
    struct calibrateCameraRetval result;
    std::vector<cv::Mat> intrinsics(2), rvecs_vec, tvecs_vec;

    if(!K.isNull()) intrinsics[0] = K.toMat();
    if(!D.isNull()) intrinsics[1] = D.toMat();
    if(!rvecs.isNull()) rvecs_vec = rvecs.toMatList();
    if(!tvecs.isNull()) tvecs_vec = tvecs.toMatList();

    result.retval = fisheye::calibrate(
				objectPoints.toMatList(), imagePoints.toMatList(),
				imageSize, intrinsics[0], intrinsics[1],
				rvecs_vec, tvecs_vec, flags, criteria);
    new(&result.intrinsics) TensorArray(intrinsics);
    new(&result.rvecs) TensorArray(rvecs_vec);
    new(&result.tvecs) TensorArray(tvecs_vec);
    return result;
}

extern "C"
struct TensorWrapper fisheye_distortPoints(
	struct TensorWrapper undistorted, struct TensorWrapper distorted,
	struct TensorWrapper K, struct TensorWrapper D, double alpha)
{
    cv::Mat distorted_mat;
    if(!distorted.isNull()) distorted_mat = distorted.toMat();
    fisheye::distortPoints(
			undistorted.toMat(), distorted_mat,
			K.toMat(), D.toMat(), alpha);
    return TensorWrapper(distorted_mat);
}

extern "C"
struct TensorWrapper fisheye_estimateNewCameraMatrixForUndistortRectify(
	struct TensorWrapper K, struct TensorWrapper D,
	struct SizeWrapper image_size, struct TensorWrapper R,
	struct TensorWrapper P, double balance,
	struct SizeWrapper new_size, double fov_scale)
{
    cv::Mat P_mat;
    if(!P.isNull()) P_mat = P.toMat();
    fisheye::estimateNewCameraMatrixForUndistortRectify(
			K.toMat(), D.toMat(), image_size, R.toMat(),
			P_mat, balance, new_size, fov_scale);
    return TensorWrapper(P_mat);
}

struct TensorArray fisheye_initUndistortRectifyMap(
	struct TensorWrapper K, struct TensorWrapper D,
	struct TensorWrapper R, struct TensorWrapper P,
	struct SizeWrapper size, int m1type,
	struct TensorWrapper map1, struct TensorWrapper map2)
{
    std::vector<cv::Mat> vec(2);
    if(!map1.isNull()) vec[0] = map1.toMat();
    if(!map2.isNull()) vec[1] = map2.toMat();
    fisheye::initUndistortRectifyMap(
		K.toMat(), D.toMat(), R.toMat(), P.toMat(),
		size, m1type, vec[0], vec[1]);
    return TensorArray(vec);
}

//TODO need to add cv::Affine3< T > Class
extern "C"
struct TensorArray fisheye_projectPoints(
	struct TensorWrapper objectPoints, struct TensorWrapper imagePoints,
	/*struct Affine3dWrapper affine,*/ struct TensorWrapper K,
	struct TensorWrapper D, double alpha, struct TensorWrapper jacobian)
{
    std::vector<cv::Mat> vec(2);
    if(!imagePoints.isNull()) vec[0] = imagePoints.toMat();
    if(!jacobian.isNull()) vec[1] = jacobian.toMat();
//    fisheye::projectPoints(
//		objectPoints.toMat(), vec[0], affine, K.toMat(),
//		D.toMat(), alpha, jacobian.toMat());
    return TensorArray(vec);
}

extern "C"
struct TensorArray fisheye_projectPoints2(
	struct TensorWrapper objectPoints, struct TensorWrapper imagePoints,
	struct TensorWrapper rvec, struct TensorWrapper tvec,
	struct TensorWrapper K, struct TensorWrapper D, double alpha,
	struct TensorWrapper jacobian)
{
    std::vector<cv::Mat> vec(2);
    if(!imagePoints.isNull()) vec[0] = imagePoints.toMat();
    if(!jacobian.isNull()) vec[1] = jacobian.toMat();
    fisheye::projectPoints(
		objectPoints.toMat(), vec[0], rvec.toMat(), tvec.toMat(),
		K.toMat(), D.toMat(), alpha, vec[1]);
    return TensorArray(vec);
}

extern "C"
struct TensorArrayPlusDouble fisheye_stereoCalibrate(
	struct TensorWrapper objectPoints, struct TensorWrapper imagePoints1,
	struct TensorWrapper imagePoints2, struct TensorWrapper K1,
	struct TensorWrapper D1, struct TensorWrapper K2,
	struct TensorWrapper D2, struct SizeWrapper imageSize,
	struct TensorWrapper R, struct TensorWrapper T,
	int flags, struct TermCriteriaWrapper criteria)
{
    struct TensorArrayPlusDouble result;
    std::vector<cv::Mat> vec(6);
    
    if(!K1.isNull()) vec[0] = K1.toMat();
    if(!D1.isNull()) vec[1] = D1.toMat();
    if(!K2.isNull()) vec[2] = K2.toMat();
    if(!D2.isNull()) vec[3] = D2.toMat();
    if(!R.isNull()) vec[4] = R.toMat();
    if(!T.isNull()) vec[5] = T.toMat();

    result.val = fisheye::stereoCalibrate(
			objectPoints.toMat(), imagePoints1.toMat(),
			imagePoints2.toMat(), vec[0], vec[1], vec[2],
			vec[3], imageSize, vec[4], vec[5], flags, criteria);
    new(&result.tensors) TensorArray(vec);
    return result;
}

extern "C"
struct TensorArray fisheye_stereoRectify(
	struct TensorWrapper K1, struct TensorWrapper D1,
	struct TensorWrapper K2, struct TensorWrapper D2,
	struct SizeWrapper imageSize, struct TensorWrapper R,
	struct TensorWrapper tvec, struct TensorWrapper R1,
	struct TensorWrapper R2, struct TensorWrapper P1,
	struct TensorWrapper P2, struct TensorWrapper Q,
	int flags, struct SizeWrapper newImageSize,
	double balance, double fov_scale)
{
    std::vector<cv::Mat> vec(5);
    if(!R1.isNull()) vec[0] = R1.toMat();
    if(!R2.isNull()) vec[1] = R2.toMat();
    if(!P1.isNull()) vec[2] = P1.toMat();
    if(!P2.isNull()) vec[3] = P2.toMat();
    if(!Q.isNull()) vec[4] = Q.toMat();

    fisheye::stereoRectify(
		K1.toMat(), D1.toMat(), K2.toMat(), D2.toMat(),
		imageSize, R.toMat(), tvec.toMat(), vec[0],
		vec[1], vec[2], vec[3], vec[4], flags,
		newImageSize, balance, fov_scale);
    return TensorArray(vec);
}

struct TensorWrapper fisheye_undistortImage(
	struct TensorWrapper distorted, struct TensorWrapper undistorted,
	struct TensorWrapper K, struct TensorWrapper D,
	struct TensorWrapper Knew, struct SizeWrapper new_size)
{
    cv::Mat undistorted_mat;
    if(!undistorted.isNull()) undistorted_mat = undistorted.toMat();
    fisheye::undistortImage(
		distorted.toMat(), undistorted_mat, K.toMat(),
		D.toMat(), TO_MAT_OR_NOARRAY(Knew), new_size);
    return TensorWrapper(undistorted_mat);
}

extern "C"
struct TensorWrapper fisheye_undistortPoints(
	struct TensorWrapper distorted, struct TensorWrapper undistorted,
	struct TensorWrapper K, struct TensorWrapper D,
	struct TensorWrapper R, struct TensorWrapper P)
{
    cv::Mat undistorted_mat;
    if(!undistorted.isNull()) undistorted_mat = undistorted.toMat();
    fisheye::undistortPoints(
		distorted.toMat(), undistorted_mat, K.toMat(), D.toMat(),
		TO_MAT_OR_NOARRAY(R), TO_MAT_OR_NOARRAY(P));
    return TensorWrapper(undistorted_mat);
}

/****************** Classes ******************/

//StereoMatcher

extern "C"
struct TensorWrapper compute(
	struct StereoMatcherPtr ptr, struct TensorWrapper left,
	struct TensorWrapper right, struct TensorWrapper disparity)
{
    cv::Mat disparity_mat;
    if(!disparity.isNull()) disparity_mat = disparity.toMat();
    ptr->compute(left.toMat(), right.toMat(), disparity_mat);
    return TensorWrapper(disparity_mat);
}

extern "C"
int StereoMatcher_getBlockSize(
	struct StereoMatcherPtr ptr)
{
    return ptr->getBlockSize();
}

extern "C"
int StereoMatcher_getDisp12MaxDiff(
	struct StereoMatcherPtr ptr)
{
    return ptr->getDisp12MaxDiff();
}

extern "C"
int StereoMatcher_getMinDisparity(
	struct StereoMatcherPtr ptr)
{
    return ptr->getMinDisparity();
}

extern "C"
int StereoMatcher_getNumDisparities(
	struct StereoMatcherPtr ptr)
{
    return ptr->getNumDisparities();
}

extern "C"
int StereoMatcher_getSpeckleRange(
	struct StereoMatcherPtr ptr)
{
    return ptr->getSpeckleRange();
}

extern "C"
int StereoMatcher_getSpeckleWindowSize(
	struct StereoMatcherPtr ptr)
{
    return ptr->getSpeckleWindowSize();
}

extern "C"
void StereoMatcher_setBlockSize(
	struct StereoMatcherPtr ptr, int blockSize)
{
    ptr->setBlockSize(blockSize);
}

extern "C"
void StereoMatcher_setDisp12MaxDiff(
	struct StereoMatcherPtr ptr, int disp12MaxDiff)
{
    ptr->setDisp12MaxDiff(disp12MaxDiff);
}

extern "C"
void StereoMatcher_setMinDisparity(
	struct StereoMatcherPtr ptr, int minDisparity)
{
    ptr->setMinDisparity(minDisparity);
}

extern "C"
void StereoMatcher_setNumDisparities(
	struct StereoMatcherPtr ptr, int numDisparities)
{
    ptr->setNumDisparities(numDisparities);
}

extern "C"
void StereoMatcher_setSpeckleRange(
	struct StereoMatcherPtr ptr, int speckleRange)
{
    ptr->setSpeckleRange(speckleRange);
}

extern "C"
void StereoMatcher_setSpeckleWindowSize(
	struct StereoMatcherPtr ptr, int speckleWindowSize)
{
    ptr->setSpeckleWindowSize(speckleWindowSize);
}

//StereoBM

extern "C"
struct StereoBMPtr StereoBM_ctor(
	int numDisparities, int blockSize)
{
    return rescueObjectFromPtr(
		cv::StereoBM::create(numDisparities, blockSize));
}

extern "C"
int StereoBM_getPreFilterCap(
	struct StereoBMPtr ptr)
{
    return ptr->getPreFilterCap();
}

extern "C"
int StereoBM_getPreFilterSize(
	struct StereoBMPtr ptr)
{
    return ptr->getPreFilterSize();
}

extern "C"
int StereoBM_getPreFilterType(
	struct StereoBMPtr ptr)
{
    return ptr->getPreFilterType();
}

extern "C"
struct RectWrapper StereoBM_getROI1(
	struct StereoBMPtr ptr)
{
    return ptr->getROI1();
}

extern "C"
struct RectWrapper StereoBM_getROI2(
	struct StereoBMPtr ptr)
{
    return ptr->getROI2();
}

extern "C"
int StereoBM_getSmallerBlockSize(
	struct StereoBMPtr ptr)
{
    return ptr->getSmallerBlockSize();
}

extern "C"
int StereoBM_getTextureThreshold(
	struct StereoBMPtr ptr)
{
    return ptr->getTextureThreshold();
}

extern "C"
int StereoBM_getUniquenessRatio(
	struct StereoBMPtr ptr)
{
    return ptr->getUniquenessRatio();
}

extern "C"
void StereoBM_setPreFilterCap(
	struct StereoBMPtr ptr, int preFilterCap)
{
    ptr->setPreFilterCap(preFilterCap);
}

extern "C"
void StereoBM_setPreFilterSize(
	struct StereoBMPtr ptr, int preFilterSize)
{
    ptr->setPreFilterSize(preFilterSize);
}

extern "C"
void StereoBM_setPreFilterType(
	struct StereoBMPtr ptr, int preFilterType)
{
    ptr->setPreFilterType(preFilterType);
}

extern "C"
void StereoBM_setROI1(
	struct StereoBMPtr ptr, RectWrapper roi1)
{
    ptr->setROI1(roi1);
}

extern "C"
void StereoBM_setROI2(
	struct StereoBMPtr ptr, RectWrapper roi2)
{
    ptr->setROI2(roi2);
}

extern "C"
void StereoBM_setSmallerBlockSize(
	struct StereoBMPtr ptr, int blockSize)
{
    ptr->setSmallerBlockSize(blockSize);
}

extern "C"
void StereoBM_setTextureThreshold(
	struct StereoBMPtr ptr, int textureThreshold)
{
    ptr->setTextureThreshold(textureThreshold);
}

extern "C"
void StereoBM_setUniquenessRatio(
	struct StereoBMPtr ptr, int uniquenessRatio)
{
    ptr->setUniquenessRatio(uniquenessRatio);
}

//StereoSGBM

extern "C"
struct StereoSGBMPtr StereoSGBM_ctor(
	int minDisparity, int numDisparities, int blockSize,
	int P1, int P2, int disp12MaxDiff, int preFilterCap,
	int uniquenessRatio, int speckleWindowSize,
	int speckleRange, int mode)
{
    return rescueObjectFromPtr(
		cv::StereoSGBM::create(
			minDisparity, numDisparities, blockSize, P1, P2,
			disp12MaxDiff, preFilterCap, uniquenessRatio,
			speckleWindowSize, speckleRange, mode));
}

extern "C"
int StereoSGBM_getMode(
	struct StereoSGBMPtr ptr)
{
    return ptr->getMode();
}

extern "C"
int StereoSGBM_getP1(
	struct StereoSGBMPtr ptr)
{
    return ptr->getP1();
}

extern "C"
int StereoSGBM_getP2(
	struct StereoSGBMPtr ptr)
{
    return ptr->getP2();
}

extern "C"
int StereoSGBM_getPreFilterCap(
	struct StereoSGBMPtr ptr)
{
    return ptr->getPreFilterCap();
}

extern "C"
int StereoSGBM_getUniquenessRatio(
	struct StereoSGBMPtr ptr)
{
    return ptr->getUniquenessRatio();
}

extern "C"
void StereoSGBM_setMode(
	struct StereoSGBMPtr ptr, int mode)
{
    ptr->setMode(mode);
}


extern "C"
void StereoSGBM_setP1(
	struct StereoSGBMPtr ptr, int P1)
{
    ptr->setP1(P1);
}

extern "C"
void StereoSGBM_setP2(
	struct StereoSGBMPtr ptr, int P2)
{
    ptr->setP2(P2);
}

extern "C"
void StereoSGBM_setPreFilterCap(
	struct StereoSGBMPtr ptr, int preFilterCap)
{
    ptr->setPreFilterCap(preFilterCap);
}

extern "C"
void StereoSGBM_setUniquenessRatio(
	struct StereoSGBMPtr ptr, int uniquenessRatio)
{
    ptr->setUniquenessRatio(uniquenessRatio);
}

















