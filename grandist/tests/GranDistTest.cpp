#include <turtle/mock.hpp>

#include "../src/GranDist.h"
#include "../src/common.h"

BOOST_AUTO_TEST_CASE(test_regions)
{
	/*
	float data[5][5] = {
			{0, 0, 0, 0, 0},
			{0, 1, 2, 1, 0},
			{0, 1, 1, 1, 0},
			{0, 1, 2, 2, 0},
			{0, 0, 0, 0, 0}
	};
	Mat mat(5, 5, CV_32F, &data);
	int timeMoment = 0;
	int layer = 0;
	bool periodic = true;
	Rect cropRect(1, 1, 3, 3);
	GranDist granDist(timeMoment, layer, mat, periodic, cropRect, false);

	auto& regionLabels = granDist.getRegionLabels();
	BOOST_CHECK_EQUAL(regionLabels.rows, mat.rows);
	BOOST_CHECK_EQUAL(regionLabels.cols, mat.cols);

	float regionLabelsData[5][5] = {
			{0, 0, 0, 0, 0},
			{0, 1, 2, 1, 0},
			{0, 1, 1, 1, 0},
			{0, 1, 2, 2, 0},
			{0, 0, 0, 0, 0}
	};
	Mat mat(5, 5, CV_32F, &data);


	BOOST_CHECK_EQUAL(bd.getInner()[i], expectedInner[i]);
	for (int row = 0; row < innerBoundaries.rows; row++) {
		for (int col = 0; col < innerBoundaries.cols; col++) {
			BOOST_CHECK_EQUAL(innerBoundaries.at<MAT_TYPE_FLOAT>(row, col), innerBoundariesExpected.at<MAT_TYPE_FLOAT>(row, col));
		}
	}
	*/

}
