/*
 * BoundaryDetect.h
 *
 *  Created on: 6 Apr 2017
 *      Author: olspern1
 */

#ifndef BOUNDARYDETECT_H_
#define BOUNDARYDETECT_H_

#include <opencv2/core/core.hpp>
#include <vector>
#include <set>

using namespace cv;
using namespace std;

enum ConnectivityMode {
	CONNECTIVITY_MODE_4,
	CONNECTIVITY_MODE_8
};

class BoundaryDetect {
public:
	BoundaryDetect(const Mat& regions, ConnectivityMode mode = CONNECTIVITY_MODE_4);
	virtual ~BoundaryDetect();
	void detect(int row, int col);

	const vector<Point2i>& getInner() const {
		return inner;
	}

	const vector<Point2i>& getOuter() const {
		return outer;
	}

	const Mat& getInnerBoundaries() const {
		return innerBoundaries;
	}

	const Mat& getOuterBoundaries() const {
		return outerBoundaries;
	}

private:
	Point2i getDelta(int dir);
	void step(Point2i point, int dir);

private:
	const Mat& regions;
	const ConnectivityMode mode;
	const int numNeighbours;
	float value;
	vector<Point2i> inner;
	vector<Point2i> outer;
	Mat innerBoundaries;
	Mat outerBoundaries;
};

#endif /* BOUNDARYDETECT_H_ */
