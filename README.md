imgproc module branch
=====================

**Roughly 24% completed**

*To be implemented:*

**Class-based stuff including algorithms:**
* GeneralizedHough
* GeneralizedHoughBallard
* GeneralizedHoughGuil
* CLAHE
* LineSegmentDetector
* Subdiv2D
* LineIterator

**Separate functions (filters, detectors, ...):**
* warpAffine
* warpPerspective
* remap
* convertMaps
* getRotationMatrix2D
* getPerspectiveTransform
* getAffineTransform
* invertAffineTransform
* getPerspectiveTransform
* getAffineTransform
* getRectSubPix
* logPolar
* linearPolar
* integral
* integral2
* integral3
* accumulate
* accumulateSquare
* accumulateProduct
* accumulateWeighted
* phaseCorrelate
* createHanningWindow
* threshold
* adaptiveThreshold
* pyrDown
* pyrUp
* buildPyramid
* undistort
* initUndistortRectifyMap
* initWideAngleProjMap
* getDefaultNewCameraMatrix
* undistortPoints
* calcHist (oh no! will need to implement SparseMat)
* calcBackProject
* compareHist
* equalizeHist
* EMD
* watershed
* pyrMeanShiftFiltering
* grabCut
* distanceTransformWithLabels
* distanceTransform
* floodFill
* cvtColor
* demosaicing
* moments
* HuMoments
* matchTemplate
* connectedComponents
* connectedComponentsWithStats
* findContours
* findContours
* approxPolyDP
* arcLength
* boundingRect
* contourArea
* minAreaRect
* boxPoints
* minEnclosingCircle
* minEnclosingTriangle
* matchShapes
* convexHull
* convexityDefects
* isContourConvex
* intersectConvexConvex
* fitEllipse
* fitLine
* pointPolygonTest
* rotatedRectangleIntersection
* blendLinear
* applyColorMap
* line
* arrowedLine
* rectangle
* circle
* ellipse
* fillConvexPoly
* fillPoly
* polylines
* drawContours
* clipLine
* ellipse2Poly
* putText
* getTextSize

*Already implemented:*

**Class-based stuff including algorithms:**

:(

**Separate functions (filters, detectors, ...):**
* getGaussianKernel
* getDerivKernels
* getGaborKernel
* getStructuringElement
* medianBlur
* GaussianBlur
* bilateralFilter
* boxFilter
* sqrBoxFilter
* blur
* filter2D
* sepFilter2D
* Sobel
* Scharr
* Laplacian
* Canny
* cornerMinEigenVal
* cornerHarris
* cornerEigenValsAndVecs
* preCornerDetect
* cornerSubPix
* goodFeaturesToTrack
* HoughLines
* HoughLinesP
* HoughCircles
* erode
* dilate
* morphologyEx
* resize
