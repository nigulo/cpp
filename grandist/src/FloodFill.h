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
#include <tuple>

using namespace cv;
using namespace std;

class FloodFill {
public:

	/**
	 * @param mat matrix of floats to fill with algorithm
	 * @param compFunc function used in neighbor comparison, == by default
	 * @param mask optional matrix of floats representing a mask. If present then single iteration of filling is
	 * done only inside the regions where mask has equal values.
	 */
	FloodFill(const Mat& mat, function<bool(float, float)> compFunc = equal_to<float>(), const Mat* mask = nullptr);
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

	const map<int /*label*/, int /*area*/>& getRegionAreas() const {
		return regionAreas;
	}

	const map<int /*label*/, tuple<int /*minRow*/, int /*maxRow*/, int /*minCol*/, int /*maxCol*/>>& getRegionExtents() const {
		return regionExtents;
	}

private:
	bool checkMask(int row, int col);
	void fillConnectedRegion(const int row, const int col);
	pair<int, int> fillRow(const int row, const int col);
	void updateClosedRegions(float neighborValue);
private:
	const Mat& mat;
	function<bool(float, float)> compFunc;
	const Mat* mask;
	Mat labels;
	map<int, float> neighbors;
	set<int> closedRegions;
	int label; // Label of the last area
	map<int /*label*/, int /*area*/> regionAreas;
	map<int /*label*/, tuple<int /*minRow*/, int /*maxRow*/, int /*minCol*/, int /*maxCol*/>> regionExtents;
	int area; // Area of the last region filled
	int maskValue;
	int minRow;
	int maxRow;
	int minCol;
	int maxCol;
};

#endif /* FLOODFILL_H_ */
