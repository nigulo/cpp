/*
 * BoundaryDetect.cpp
 *
 *  Created on: 6 Apr 2017
 *      Author: olspern1
 */

#include "BoundaryDetect.h"
#include "common.h"

BoundaryDetect::BoundaryDetect(const Mat& mat, ConnectivityMode mode) :
	mat(mat),
	mode(mode),
	numNeighbours(mode == CONNECTIVITY_MODE_4 ? 4 : 8),
	value(0) {

}

BoundaryDetect::~BoundaryDetect() {

}

Point2i BoundaryDetect::getDelta(int dir) {
	if (mode == CONNECTIVITY_MODE_4) {
		switch (dir) {
			case 0:
				return Point2i(1, 0);
			case 1:
				return Point2i(0, -1);
			case 2:
				return Point2i(-1, 0);
			case 3:
				return Point2i(0, 1);
		}
	} else {
		switch (dir) {
			case 0:
				return Point2i(1, 0);
			case 1:
				return Point2i(1, -1);
			case 2:
				return Point2i(0, -1);
			case 3:
				return Point2i(-1, -1);
			case 4:
				return Point2i(-1, 0);
			case 5:
				return Point2i(-1, 1);
			case 6:
				return Point2i(0, 1);
			case 7:
				return Point2i(1, 1);
		}
	}
	assert(false);
}

void BoundaryDetect::step(Point2i point, int dir) {
	inner.push_back(point);
	//--------------------------------------
	// Meaning of dir in 4-connectivity mode
	//    1
	//    |
	// 2--+--0
	//    |
	//    3
	//--------------------------------------
	// Meaning of dir in 8-connectivity mode
	//  3 2 1
	//   \|/
	// 4--+--0
	//   /|\
	//  5 6 7
	//--------------------------------------
	if (mode == CONNECTIVITY_MODE_4) {
		dir += 3;
	} else {
		if (dir % 2 == 0) {
			dir += 7;
		} else {
			dir += 6;
		}
	}
	dir %= numNeighbours;
	for (int i = 0; i < numNeighbours; i++) {
		auto newPoint = point + getDelta(dir);
		if (newPoint.x >= 0 && newPoint.x < mat.cols && newPoint.y >= 0 && newPoint.y < mat.rows && mat.at<MAT_TYPE_FLOAT>(newPoint.y, newPoint.x) == value) {
			if (inner.size() >= 3 && newPoint == inner[1] && inner[inner.size() - 1] == inner[0]) {
				// loop is connected
				inner.pop_back();
			} else {
				step(newPoint, dir);
			}
			break;
		} else {
			outer.push_back(newPoint);
		}
		dir += 1;
		dir %= numNeighbours;
	}
}

void removeDuplicates(vector<Point2i>& points) {
	for (size_t i = 0; i < points.size() - 1; i++) {
		Point p = points[i];
		for (size_t j = i + 1; j < points.size(); j++) {
			if (p == points[j]) {
				points.erase(points.begin() + j);
				j--;
			}
		}
	}
}

void BoundaryDetect::detect(int row, int col) {
	value = mat.at<MAT_TYPE_FLOAT>(row, col);
	int c = col;
	for (; c > 0; c--) {
		if (mat.at<MAT_TYPE_FLOAT>(row, c - 1) != value) {
			break;
		}
	}
	int dir = mode == CONNECTIVITY_MODE_4 ? 0 : 7;
	step(Point2i(c, row), dir);
	removeDuplicates(inner);
	removeDuplicates(outer);
}
