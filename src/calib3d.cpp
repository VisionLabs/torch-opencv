#include <calib3d.hpp>

extern "C"
struct calibrateCameraRetval calibrateCamera(
	struct TensorArray objectPoints, struct TensorArray imagePoints,
	struct SizeWrapper imageSize, struct TensorWrapper cameraMatrix,
	struct TensorWrapper distCoeffs, struct TensorArray rvecs,
	struct TensorArray tvecs, int flags, struct TermCriteriaWrapper criteria)
{
    struct calibrateCameraRetval result;
    std::vector<MatT> intrinsics(2);
    std::vector<cv::Mat> rvecs_vec, tvecs_vec;

    intrinsics[0] = cameraMatrix.toMatT();
    intrinsics[1] = distCoeffs.toMatT();
    rvecs_vec = rvecs.toMatList();
    tvecs_vec = tvecs.toMatList();

    result.retval = cv::calibrateCamera(objectPoints.toMatList(), imagePoints.toMatList(),
			       		imageSize, intrinsics[0], intrinsics[1],
			       		rvecs_vec, tvecs_vec, flags, criteria);
    new(&result.intrinsics) TensorArray(intrinsics);

    std::vector<MatT> rvecs_vecT = get_vec_MatT(rvecs_vec), tvecs_vecT = get_vec_MatT(tvecs_vec);
    new(&result.rvecs) TensorArray(rvecs_vecT);
    new(&result.tvecs) TensorArray(tvecs_vecT);
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

    return TensorWrapper(MatT(retval));
}

extern "C"
struct TensorArray composeRT(
	struct TensorWrapper rvec1, struct TensorWrapper tvec1, struct TensorWrapper rvec2,
	struct TensorWrapper tvec2, struct TensorWrapper rvec3, struct TensorWrapper tvec3,
	struct TensorWrapper dr3dr1, struct TensorWrapper dr3dt1, struct TensorWrapper dr3dr2,
	struct TensorWrapper dr3dt2, struct TensorWrapper dt3dr1, struct TensorWrapper dt3dt1,
	struct TensorWrapper dt3dr2, struct TensorWrapper dt3dt2)
{
    std::vector<MatT> vec(10);
    vec[0] = rvec3.toMatT();
    vec[1] = tvec3.toMatT();
    vec[2] = dr3dr1.toMatT();
    vec[3] = dr3dt1.toMatT();
    vec[4] = dr3dr2.toMatT();
    vec[5] = dr3dt2.toMatT();
    vec[6] = dt3dr1.toMatT();
    vec[7] = dt3dt1.toMatT();
    vec[8] = dt3dr2.toMatT();
    vec[9] = dt3dt2.toMatT();

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
    MatT lines_mat = lines.toMatT();
    cv::computeCorrespondEpilines(points.toMat(), whichImage, F.toMat(),lines_mat);
    return TensorWrapper(lines_mat);
}

extern "C"
struct TensorWrapper convertPointsFromHomogeneous(
	struct TensorWrapper src, struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    cv::convertPointsFromHomogeneous(src.toMat(),dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper convertPointsHomogeneous(
	struct TensorWrapper src, struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    cv::convertPointsHomogeneous(src.toMat(), dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorWrapper convertPointsToHomogeneous(
	struct TensorWrapper src, struct TensorWrapper dst)
{
    MatT dst_mat = dst.toMatT();
    cv::convertPointsToHomogeneous(src.toMat(),dst_mat);
    return TensorWrapper(dst_mat);
}

extern "C"
struct TensorArray correctMatches(
	struct TensorWrapper F, struct TensorWrapper points1,
	struct TensorWrapper points2, struct TensorWrapper newPoints1,
	struct TensorWrapper newPoints2)
{
    std::vector<MatT> vec(2);
    vec[0] = newPoints1.toMatT();
    vec[1] = newPoints2.toMatT();
    cv::correctMatches(F.toMat(),
      points1.toMat(), points2.toMat(), vec[0], vec[1]);

    return TensorArray(vec);
}

extern "C"
struct TensorArray decomposeEssentialMat(
	struct TensorWrapper E, struct TensorWrapper R1,
	struct TensorWrapper R2, struct TensorWrapper t)
{
    std::vector<MatT> vec(3);
    vec[0] = R1.toMatT();
    vec[1] = R2.toMatT();
    vec[2] = t.toMatT();
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
    std::vector<MatT> rotations_vec, translations_vec, normals_vec;

    rotations_vec = rotations.toMatTList();
    translations_vec = translations.toMatTList();
    normals_vec = normals.toMatTList();

    result.val = cv::decomposeHomographyMat(
      H.toMat(), K.toMat(), rotations_vec, translations_vec, normals_vec);

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
    std::vector<MatT> vec(7);
    vec[0] = cameraMatrix.toMatT();
    vec[1] = rotMatrix.toMatT();
    vec[2] = transVect.toMatT();
    vec[3] = rotMatrixX.toMatT();
    vec[4] = rotMatrixY.toMatT();
    vec[5] = rotMatrixZ.toMatT();
    vec[6] = eulerAngles.toMatT();

    cv::decomposeProjectionMatrix(projMatrix.toMat(),
      vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6]);
    return TensorArray(vec);
}

extern "C"
void drawChessboardCorners(
	struct TensorWrapper image, struct SizeWrapper patternSize,
	struct TensorWrapper corners, bool patternWasFound)
{
    cv::drawChessboardCorners(image.toMat(),
      patternSize, corners.toMat(), patternWasFound);
}

extern "C"
struct TensorArrayPlusInt estimateAffine3D(
	struct TensorWrapper src, struct TensorWrapper dst,
	struct TensorWrapper out, struct TensorWrapper inliers,
	double ransacThreshold, double confidence)
{
    struct TensorArrayPlusInt result;
    std::vector<MatT> vec(2);
    vec[0] = out.toMatT();
    vec[1] = inliers.toMatT();
    result.val = cv::estimateAffine3D(
      src.toMat(), dst.toMat(), vec[0], vec[1], ransacThreshold, confidence);
    new(&result.tensors) TensorArray(vec);
    return result;
}

extern "C"
void filterSpeckles(
	struct TensorWrapper img, double newVal, int maxSpeckleSize,
	double maxDiff, struct TensorWrapper buf)
{
    if(buf.isNull()) {
        cv::filterSpeckles(img.toMat(), newVal, maxSpeckleSize,
                           maxDiff, cv::noArray());
    } else {
        cv::filterSpeckles(img.toMat(), newVal, maxSpeckleSize,
                           maxDiff, buf.toMat());
    }
}

//TODO InputOutputArray
extern "C"
struct TensorWrapper find4QuadCornerSubpix(
	struct TensorWrapper img, struct TensorWrapper corners,
	struct SizeWrapper region_size)
{
    cv::find4QuadCornerSubpix(img.toMat(), corners.toMat(), region_size);
    return corners;
}

extern "C"
struct TensorPlusBool findChessboardCorners(
	struct TensorWrapper image, struct SizeWrapper patternSize,
	struct TensorWrapper corners, int flags)
{
    struct TensorPlusBool result;
    MatT corners_mat = corners.toMatT();
    result.val = cv::findChessboardCorners(image.toMat(),
                                           patternSize, corners_mat, flags);
    new(&result.tensor) TensorWrapper(corners_mat);
    return result;
}

extern "C"
struct TensorPlusBool findCirclesGrid(
	struct TensorWrapper image, struct SizeWrapper patternSize,
	struct TensorWrapper centers, int flags,
  struct SimpleBlobDetectorPtr blobDetector)
{
    cv::Ptr<cv::FeatureDetector> blobDetectorPtr(
      static_cast<cv::SimpleBlobDetector *>(blobDetector.ptr));
    rescueObjectFromPtr(blobDetectorPtr);

    struct TensorPlusBool result;
    MatT centers_mat = centers.toMatT();

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
    return TensorWrapper(
      MatT(cv::findEssentialMat(points1.toMat(), points2.toMat(),
                					      focal, pp, method, prob, threshold,
                					      TO_MAT_OR_NOARRAY(mask))));
}

extern "C"
struct TensorWrapper findFundamentalMat(
	struct TensorWrapper points1, struct TensorWrapper points2, int method,
	double param1, double param2, struct TensorWrapper mask)
{
    return TensorWrapper(
      MatT(cv::findFundamentalMat(points1.toMat(), points2.toMat(),
                      						method, param1, param2,
                      						TO_MAT_OR_NOARRAY(mask))));
}

extern "C"
struct TensorArray findFundamentalMat2(
	struct TensorWrapper points1, struct TensorWrapper points2,
	struct TensorWrapper mask, int method, double param1, double param2)
{
    std::vector<MatT> vec(2);
    vec[1] = mask.toMatT();
    vec[0] = MatT(cv::findFundamentalMat(points1.toMat(), points2.toMat(),
			vec[1], method, param1, param2));
    return TensorArray(vec);
}

extern "C"
struct TensorWrapper findHomography(
	struct TensorWrapper srcPoints, struct TensorWrapper dstPoints,
	int method, double ransacReprojThreshold, struct TensorWrapper mask,
	const int maxIters, const double confidence)
{
    return TensorWrapper(
      MatT(cv::findHomography(srcPoints.toMat(), dstPoints.toMat(),
                					    method, ransacReprojThreshold,
                					    TO_MAT_OR_NOARRAY(mask),
                					    maxIters, confidence)));
}

extern "C"
struct TensorArray findHomography2(
	struct TensorWrapper srcPoints, struct TensorWrapper dstPoints,
	struct TensorWrapper mask, int method, double ransacReprojThreshold)
{
    std::vector<MatT> vec(2);
    vec[1] = mask.toMatT();
    vec[0] = MatT(cv::findHomography(srcPoints.toMat(), dstPoints.toMat(),
			vec[1], method, ransacReprojThreshold));
    return TensorArray(vec);
}

extern "C"
struct TensorPlusRect getOptimalNewCameraMatrix(
	struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
	struct SizeWrapper imageSize, double alpha, struct SizeWrapper newImgSize,
	bool centerPrincipalPoint)
{
    struct TensorPlusRect result;
    cv::Rect validPixROI;
    new(&result.tensor) TensorWrapper(
				MatT(cv::getOptimalNewCameraMatrix(
						cameraMatrix.toMat(), distCoeffs.toMat(),
            imageSize, alpha, newImgSize,
            &validPixROI, centerPrincipalPoint)));
    result.rect = validPixROI;
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
    return TensorWrapper(MatT(cv::initCameraMatrix2D(
					objectPoints.toMatList(), imagePoints.toMatList(),
					imageSize, aspectRatio)));
}

extern "C"
struct TensorArray matMulDeriv(
	struct TensorWrapper A, struct TensorWrapper B,
	struct TensorWrapper dABdA, struct TensorWrapper dABdB)
{
    std::vector<MatT> result(2);
    result[0] = dABdA.toMatT();
    result[1] = dABdB.toMatT();
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
    std::vector<MatT> result(2);
    result[0] = imagePoints.toMatT();
    result[1] = jacobian.toMatT();
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
    std::vector<MatT> vec(2);
    vec[0] = R.toMatT();
    vec[1] = t.toMatT();
    if(mask.isNull()) {
        result.val = cv::recoverPose(E.toMat(), points1.toMat(), points2.toMat(), vec[0],
                                     vec[1], focal, pp, cv::noArray());
    }
    else {
        result.val = cv::recoverPose(E.toMat(), points1.toMat(), points2.toMat(), vec[0],
                                     vec[1], focal, pp, mask.toMat());
    }
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
	struct TensorWrapper T13, struct TensorWrapper R1,
    struct TensorWrapper R2, struct TensorWrapper R3,
    struct TensorWrapper P1, struct TensorWrapper P2,
    struct TensorWrapper P3, struct TensorWrapper Q,
    double alpha, struct SizeWrapper newImgSize, int flags)
{
    struct TensorArrayPlusRectArrayPlusFloat result;
    std::vector<MatT> vec(7);
    vec[0] = R1.toMatT();
    vec[1] = R1.toMatT();
    vec[2] = R1.toMatT();
    vec[3] = R1.toMatT();
    vec[4] = R1.toMatT();
    vec[5] = R1.toMatT();
    vec[6] = R1.toMatT();

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
    MatT _3dImage_mat = _3dImage.toMatT();
    cv::reprojectImageTo3D(disparity.toMat(), _3dImage_mat,
                           Q.toMat(), handleMissingValues, ddepth);
    return TensorWrapper(_3dImage_mat);
}

extern "C"
struct TensorArray Rodrigues(
	struct TensorWrapper src, struct TensorWrapper dst, struct TensorWrapper jacobian)
{
    std::vector<MatT> result;
    result[0] = dst.toMatT();
    result[1] = jacobian.toMatT();
    cv::Rodrigues(src.toMat(), result[0], result[1]);
    return TensorArray(result);
}

extern "C"
struct TensorArrayPlusVec3d RQDecomp3x3(
	struct TensorWrapper src, struct TensorWrapper mtxR, struct TensorWrapper mtxQ,
	struct TensorWrapper Qx, struct TensorWrapper Qy, struct TensorWrapper Qz)
{
    struct TensorArrayPlusVec3d result;
    std::vector<MatT> vec(5);
    vec[0] = mtxR.toMatT();
    vec[1] = mtxQ.toMatT();
    vec[2] = Qx.toMatT();
    vec[3] = Qy.toMatT();
    vec[4] = Qz.toMatT();
    new(&result.vec3d) Vec3dWrapper(cv::RQDecomp3x3(src.toMat(),
                                    vec[0], vec[1], vec[2], vec[3], vec[4]));
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
    std::vector<MatT> vec(2);
    vec[0] = rvec.toMatT();
    vec[1] = tvec.toMatT();
    result.val = cv::solvePnP(objectPoints.toMat(), imagePoints.toMat(),
                              cameraMatrix.toMat(),distCoeffs.toMat(),
                              vec[0], vec[1], useExtrinsicGuess, flags);
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
    std::vector<MatT> vec(3);
    vec[0] = rvec.toMatT();
    vec[1] = tvec.toMatT();
    vec[2] = inliers.toMatT();
    result.val = cv::solvePnPRansac(objectPoints.toMat(), imagePoints.toMat(),
                 cameraMatrix.toMat(), distCoeffs.toMat(), vec[0], vec[1],
                 useExtrinsicGuess, iterationsCount, reprojectionError,
                 confidence, vec[2], flags);
    new(&result.tensors) TensorArray(vec);
    return result;
}

extern "C"
struct TensorArrayPlusDouble stereoCalibrate(
	struct TensorArray objectPoints, struct TensorArray imagePoints1,
	struct TensorArray imagePoints2, struct TensorWrapper cameraMatrix1,
	struct TensorWrapper distCoeffs1, struct TensorWrapper cameraMatrix2,
	struct TensorWrapper distCoeffs2, struct SizeWrapper imageSize,
	struct TensorWrapper R, struct TensorWrapper T,
	struct TensorWrapper E, struct TensorWrapper F,
	int flags, struct TermCriteriaWrapper criteria)
{
    struct TensorArrayPlusDouble result;
    std::vector<MatT> vec(8);
    vec[0] = cameraMatrix1.toMatT();
    vec[1] = distCoeffs1.toMatT();
    vec[2] = cameraMatrix2.toMatT();
    vec[3] = distCoeffs2.toMatT();
    vec[4] = R.toMatT();
    vec[5] = T.toMatT();
    vec[6] = E.toMatT();
    vec[7] = F.toMatT();

    result.val = cv::stereoCalibrate(
			objectPoints.toMatList(), imagePoints1.toMatList(),
			imagePoints2.toMatList(), vec[0], vec[1], vec[2],
			vec[3], imageSize, vec[4], vec[5], vec[6],
			vec[7], flags, criteria);
    new(&result.tensors) TensorArray(vec);
    return result;
}

extern "C"
struct TensorArrayPlusRectArray stereoRectify(
	struct TensorWrapper cameraMatrix1, struct TensorWrapper distCoeffs1,
	struct TensorWrapper cameraMatrix2, struct TensorWrapper distCoeffs2,
	struct SizeWrapper imageSize, struct TensorWrapper R,
	struct TensorWrapper T, struct TensorWrapper R1,
	struct TensorWrapper R2, struct TensorWrapper P1,
	struct TensorWrapper P2, struct TensorWrapper Q,
	int flags, double alpha,struct SizeWrapper newImageSize)
{
    TensorArrayPlusRectArray retval;
    std::vector<cv::Rect> rec(2);
    std::vector<MatT> vec(5);
    vec[0] = R1.toMatT();
    vec[1] = R2.toMatT();
    vec[2] = P1.toMatT();
    vec[3] = P2.toMatT();
    vec[4] = Q.toMatT();

    cv::stereoRectify(cameraMatrix1.toMat(), distCoeffs1.toMat(),
		                  cameraMatrix2.toMat(), distCoeffs2.toMat(),
                  		imageSize, R.toMat(), T.toMat(),
                      vec[0], vec[1], vec[2], vec[3], vec[4],
                  		flags, alpha, newImageSize, &rec[0], &rec[1]);
    new(&retval.tensors) TensorArray(vec);
    new(&retval.rects) RectArray(rec);
    return retval;
}

extern "C"
struct TensorArrayPlusBool stereoRectifyUncalibrated(
	struct TensorWrapper points1, struct TensorWrapper points2,
	struct TensorWrapper F, struct SizeWrapper imgSize,
	struct TensorWrapper H1, struct TensorWrapper H2, double threshold)
{
    struct TensorArrayPlusBool result;
    std::vector<MatT> vec(2);
    vec[0] = H1.toMatT();
    vec[1] = H2.toMatT();

    cv::stereoRectifyUncalibrated(points1.toMat(), points2.toMat(), F.toMat(),
                              		imgSize, vec[0], vec[1], threshold);
    new(&result.tensors) TensorArray(vec);
    return result;
}

extern "C"
struct TensorWrapper triangulatePoints(
	struct TensorWrapper projMatr1, struct TensorWrapper projMatr2,
	struct TensorWrapper projPoints1, struct TensorWrapper projPoints2,
    struct TensorWrapper points4D)
{
    MatT points4D_mat = points4D.toMatT();
    cv::triangulatePoints(projMatr1.toMat(), projMatr2.toMat(),
                          projPoints1.toMat(), projPoints2.toMat(), points4D_mat);
    return TensorWrapper(points4D_mat);
}

extern "C"
struct TensorWrapper validateDisparity(
	struct TensorWrapper disparity, struct TensorWrapper cost,
        int minDisparity, int numberOfDisparities, int disp12MaxDisp)
{
    cv::validateDisparity(disparity.toMat(), cost.toMat(), minDisparity,
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
    std::vector<MatT> intrinsics(2), rvecs_vec, tvecs_vec;

    intrinsics[0] = K.toMatT();
    intrinsics[1] = D.toMatT();
    rvecs_vec = rvecs.toMatTList();
    tvecs_vec = tvecs.toMatTList();

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
    MatT distorted_mat = distorted.toMatT();
    fisheye::distortPoints(undistorted.toMat(), distorted_mat,
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
    MatT P_mat = P.toMatT();
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
    std::vector<MatT> vec(2);
    vec[0] = map1.toMatT();
    vec[1] = map2.toMatT();

    fisheye::initUndistortRectifyMap(K.toMat(), D.toMat(), R.toMat(), P.toMat(),
		  size, m1type, vec[0], vec[1]);
    return TensorArray(vec);
}

extern "C"
struct TensorArray fisheye_projectPoints2(
	struct TensorWrapper objectPoints, struct TensorWrapper imagePoints,
	struct TensorWrapper rvec, struct TensorWrapper tvec,
	struct TensorWrapper K, struct TensorWrapper D, double alpha,
	struct TensorWrapper jacobian)
{
    std::vector<MatT> vec(2);
    vec[0] = imagePoints.toMatT();
    vec[1] = jacobian.toMatT();

    fisheye::projectPoints(objectPoints.toMat(), vec[0], rvec.toMat(),
      tvec.toMat(), K.toMat(), D.toMat(), alpha, vec[1]);
    return TensorArray(vec);
}

extern "C"
struct TensorArrayPlusDouble fisheye_stereoCalibrate(
	struct TensorArray objectPoints, struct TensorArray imagePoints1,
	struct TensorArray imagePoints2, struct TensorWrapper K1,
	struct TensorWrapper D1, struct TensorWrapper K2,
	struct TensorWrapper D2, struct SizeWrapper imageSize,
	struct TensorWrapper R, struct TensorWrapper T,
	int flags, struct TermCriteriaWrapper criteria)
{
    struct TensorArrayPlusDouble result;
    std::vector<MatT> vec(6);
    vec[0] = K1.toMatT();
    vec[1] = D1.toMatT();
    vec[2] = K2.toMatT();
    vec[3] = D2.toMatT();
    vec[4] = R.toMatT();
    vec[5] = T.toMatT();

    result.val = fisheye::stereoCalibrate(objectPoints.toMatList(),
      imagePoints1.toMatList(),imagePoints2.toMatList(), vec[0],
      vec[1], vec[2],vec[3], imageSize, vec[4], vec[5], flags, criteria);
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
    std::vector<MatT> vec(5);
    vec[0] = R1.toMatT();
    vec[1] = R2.toMatT();
    vec[2] = P1.toMatT();
    vec[3] = P2.toMatT();
    vec[4] = Q.toMatT();

    fisheye::stereoRectify(K1.toMat(), D1.toMat(), K2.toMat(), D2.toMat(),
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
    MatT undistorted_mat = undistorted.toMatT();
    fisheye::undistortImage(distorted.toMat(), undistorted_mat, K.toMat(),
		  D.toMat(), TO_MAT_OR_NOARRAY(Knew), new_size);
    return TensorWrapper(undistorted_mat);
}

extern "C"
struct TensorWrapper fisheye_undistortPoints(
	struct TensorWrapper distorted, struct TensorWrapper undistorted,
	struct TensorWrapper K, struct TensorWrapper D,
	struct TensorWrapper R, struct TensorWrapper P)
{
    MatT undistorted_mat = undistorted.toMatT();
    fisheye::undistortPoints(distorted.toMat(), undistorted_mat,
      K.toMat(), D.toMat(), TO_MAT_OR_NOARRAY(R), TO_MAT_OR_NOARRAY(P));
    return TensorWrapper(undistorted_mat);
}

/****************** Classes ******************/

//StereoMatcher

extern "C"
struct TensorWrapper StereoMatcher_compute(
	struct StereoMatcherPtr ptr, struct TensorWrapper left,
	struct TensorWrapper right, struct TensorWrapper disparity)
{
    MatT disparity_mat = disparity.toMatT();
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
	struct StereoBMPtr ptr, struct RectWrapper roi1)
{
    ptr->setROI1(roi1);
}

extern "C"
void StereoBM_setROI2(
	struct StereoBMPtr ptr, struct RectWrapper roi2)
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
