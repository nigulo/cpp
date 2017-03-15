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
#include <memory>
#include <map>

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
	GranDist(int timeMoment, int layer, Mat granules, int originalHeight, int originalWidth, bool periodic, Rect cropRect, bool debug);
	void process();
	virtual ~GranDist();
private:
	void labelRegions();
	tuple<Mat, Mat, Mat, Mat> calcDistances(const Mat& mat, const Mat& granuleLabels) const;
	set<int> getClosedRegions() const;
	set<int /*granuleLabel*/> getGranulesOnBoundaries() const;
	bool inDownFlowPatch(const Mat& regionLabels, int row, int col) const;
	bool onBoundary(const Mat& granuleLabels, int row, int col) const;
	unique_ptr<float> getIgLaneIndex(const Mat& regionLabels, int startRow, int endRow, int col, int domainStart, int domainEnd) const;
	void filterExtrema(vector<tuple<float /*value*/, int /*row*/, int /*col*/>>& extrema, bool byRegionLabel = true) const;

public:
	string getGranuleSizesStr() const {
		return granuleSizesOut.str();
	}

	string getIgLaneMinWidthsStr() const {
		return igLaneMinWidthsOut.str();
	}

	string getIgLaneMaxWidthsStr() const {
		return igLaneMaxWidthsOut.str();
	}

	string getDfPatchSizesStr() const {
		return dfPatchSizesOut.str();
	}

private:
	int timeMoment;
	int layer;
	Mat granules;
	int originalHeight;
	int originalWidth;
	bool periodic;
	Rect cropRect;
	bool saveMaps;
	Mat regionLabels;
	set<int> regionsOnBoundaries;
	set<int> downFlowPatches;
	Mat regionLabelsFloat;
	int numRegions;
	map<int, int> regionAreas;
	// Output streams for results

	stringstream granuleSizesOut;
	stringstream igLaneMinWidthsOut;
	stringstream igLaneMaxWidthsOut;
	stringstream dfPatchSizesOut;
};

#endif /* GRANDIST_H_ */
