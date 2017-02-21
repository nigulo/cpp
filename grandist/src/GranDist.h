/*
 * GranDist.h
 *
 *  Created on: 21 Feb 2017
 *      Author: olspern1
 */

#ifndef GRANDIST_H_
#define GRANDIST_H_

#include <opencv2/core/core.hpp>
#include <set>

using namespace cv;
using namespace std;

// Need to use float matrices even for keeping integer values
// because the library supports rotations only for these
typedef float MAT_TYPE_FLOAT;
// Integer type can be used for matrices not involved in rotations
typedef int MAT_TYPE_INT;

#define OUT_GRANULE 1 ///< The point is inside granule
#define IN_GRANULE 2 ///< The point is outside of the

#define DELTA_ANGLE 1.0 ///< The angle increment used in rotations
#define INFTY numeric_limits<float>::max()
#define RED Vec3b(0, 0, 255)

class GranDist {
public:
	GranDist(int layer, Mat granules, int originalHeight, int originalWidth, bool periodic, Rect cropRect);
	void process();
	virtual ~GranDist();
private:
	pair<Mat, Mat> calcDistances(const Mat& mat, const Mat& granuleLabels) const;
	Mat labelGranules() const;
	set<int /*granuleLabel*/> getGranulesOnBoundaries() const;
private:
	int layer;
	Mat granules;
	int originalHeight;
	int originalWidth;
	bool periodic;
	Rect cropRect;
	Mat granuleLabels;
	set<int> granulesOnBoundaries;
};

#endif /* GRANDIST_H_ */
