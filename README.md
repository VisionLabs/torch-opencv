imgproc module branch
=====================

**Roughly 87% completed**

Unstable pieces of code (to be tested):

* undistort (how should it work? please test if it's correct)
* grabCut (same thing)

*To be implemented:*

**Class-based stuff including algorithms:**
* CLAHE
* LineSegmentDetector
* Subdiv2D
* LineIterator

**Separate functions (filters, detectors, ...):**
* calcHist (SparseMat overload)
* calcBackProject (SparseMat overload)
* compareHist (SparseMat overload)

*Already implemented:*

**Class-based stuff including algorithms:**

* GeneralizedHough
* GeneralizedHoughBallard
* GeneralizedHoughGuil

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
* warpAffine
* warpPerspective
* remap
* convertMaps
* getRotationMatrix2D
* getPerspectiveTransform
* getAffineTransform
* invertAffineTransform
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
* calcHist
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
* putText
* demosaicing
* moments
* matchTemplate
* connectedComponents
* connectedComponentsWithStats
* HuMoments
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
* getTextSize

122/141