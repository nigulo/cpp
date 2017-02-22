/*
 * FloodFill.h
 *
 *  Created on: 21 Feb 2017
 *      Author: olspern1
 */

#ifndef FLOODFILL_H_
#define FLOODFILL_H_

#include <opencv2/core/core.hpp>
#include <functional>
#include <set>
#include <map>

using namespace cv;
using namespace std;

class FloodFill {
public:
	FloodFill(const Mat& mat, function<bool(float, float)> compFunc = equal_to<float>());
	virtual ~FloodFill();

	void fill(const int row, const int col);

	const Mat& getLabels() const {
		return labels;
	}

	const set<int>& getClosedRegions() const {
		return closedRegions;
	}

	void setCompFunc(function<bool(float, float)> compFunc) {
		this->compFunc = compFunc;
	}

	int getNumRegions() const {
		return label - 1;
	}

	const map<int, int> getRegionAreas() {
		return regionAreas;
	}
private:
	void fillConnectedRegion(const int row, const int col);
	pair<int, int> fillRow(const int row, const int col);
	void updateClosedRegions(float neighborValue);
private:
	const Mat& mat;
	Mat labels;
	map<int, float> neighbors;
	set<int> closedRegions;
	function<bool(float, float)> compFunc;
	int label; // Label of the last area
	map<int /*label*/, int> regionAreas;
	int area; // Area of the last region filled
};

#endif /* FLOODFILL_H_ */
