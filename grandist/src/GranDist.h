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

enum RegionType {
	OUT_OF_DOMAIN, ///< 0
	DOWN_FLOW, 	   ///< 1
	UP_FLOW        ///< 2
};

#define DELTA_ANGLE 1.0 ///< The angle increment used in rotations
#define INFTY numeric_limits<float>::max()
#define RED Vec3b(0, 0, 255)

class GranDist {
public:
	GranDist(int layer, Mat granules, int originalHeight, int originalWidth, bool periodic, Rect cropRect);
	void process();
	virtual ~GranDist();
private:
	tuple<Mat, Mat, Mat, pair<Mat, Mat>> calcDistances(const Mat& mat, const Mat& granuleLabels) const;
	Mat labelRegions() const;
	set<int> getClosedRegions() const;
	set<int /*granuleLabel*/> getGranulesOnBoundaries() const;
	bool inDownFlowBubble(const Mat& regionLabels, int row, int col) const;
	bool onBoundary(const Mat& granuleLabels, int row, int col) const;
private:
	int layer;
	Mat granules;
	int originalHeight;
	int originalWidth;
	bool periodic;
	Rect cropRect;
	Mat regionLabels;
	set<int> regionsOnBoundaries;
	set<int> downFlowBubbles;
	Mat regionLabelsFloat;
};

#endif /* GRANDIST_H_ */
