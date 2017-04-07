#include <turtle/mock.hpp>

#include "../src/GranDist.h"
#include "../src/common.h"
#include <iostream>

using namespace std;

BOOST_AUTO_TEST_CASE(test_tile_matrix)
{
	float data[8][8] = {
			{0, 0, 0, 0, 0, 0, 0, 0},
			{0, 0, 0, 0, 0, 0, 0, 0},
			{0, 0, 0, 0, 0, 0, 0, 0},
			{0, 0, 0, 0, 0, 0, 0, 0},
			{0, 0, 0, 0, 11, 12, 13, 0},
			{0, 0, 0, 0, 21, 22, 23, 0},
			{0, 0, 0, 0, 31, 32, 33, 0},
			{0, 0, 0, 0, 0, 0, 0, 0}
	};
	Mat mat(8, 8, CV_32F, &data);

	float dataExpected[8][8] = {
			{0, 0, 0, 0, 0, 0, 0, 0},
			{0, 11, 12, 13, 11, 12, 13, 0},
			{0, 21, 22, 23, 21, 22, 23, 0},
			{0, 31, 32, 33, 31, 32, 33, 0},
			{0, 11, 12, 13, 11, 12, 13, 0},
			{0, 21, 22, 23, 21, 22, 23, 0},
			{0, 31, 32, 33, 31, 32, 33, 0},
			{0, 0, 0, 0, 0, 0, 0, 0}
	};
	Mat matExpected(8, 8, CV_32F, &dataExpected);

	tileMatrix(mat, 3, 3);
	for (int row = 0; row < mat.rows; row++) {
		for (int col = 0; col < mat.cols; col++) {
			BOOST_CHECK_EQUAL(mat.at<MAT_TYPE_FLOAT>(row, col), matExpected.at<MAT_TYPE_FLOAT>(row, col));
		}
	}
}

BOOST_AUTO_TEST_CASE(test_init)
{

	float data[16][10] = {
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{0, 0, 0, 0, 0, 1, 1, 1, 1, 0},
			{0, 0, 0, 0, 0, 1, 2, 2, 1, 0},
			{0, 0, 0, 0, 0, 1, 2, 2, 1, 0},
			{0, 0, 0, 0, 0, 1, 1, 1, 1, 0},
			{0, 0, 0, 0, 0, 2, 2, 2, 2, 0},
			{0, 0, 0, 0, 0, 1, 2, 1, 2, 0},
			{0, 0, 0, 0, 0, 1, 2, 2, 2, 0},
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	};
	Mat mat(16, 10, CV_32F, &data);
	int timeMoment = 0;
	int layer = 0;
	bool periodic = true;
	Rect cropRect(5, 8, 4, 7);
	GranDist granDist(timeMoment, layer, mat, periodic, cropRect, false);


	int regionLabelsData[16][10] = {
			{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			{1, 2, 2, 2, 2, 2, 2, 2, 2, 1},
			{1, 2, 3, 3, 2, 2, 4, 4, 2, 1},
			{1, 2, 3, 3, 2, 2, 4, 4, 2, 1},
			{1, 2, 2, 2, 2, 2, 2, 2, 2, 1},
			{1, 5, 5, 5, 5, 5, 5, 5, 5, 1},
			{1, 6, 5, 7, 5, 6, 5, 8, 5, 1},
			{1, 6, 5, 5, 5, 6, 5, 5, 5, 1},
			{1, 6, 6, 6, 6, 6, 6, 6, 6, 1},
			{1, 6, 9, 9, 6, 6, 10, 10, 6, 1},
			{1, 6, 9, 9, 6, 6, 10, 10, 6, 1},
			{1, 6, 6, 6, 6, 6, 6, 6, 6, 1},
			{1, 11, 11, 11, 11, 11, 11, 11, 11, 1},
			{1, 12, 11, 13, 11, 14, 11, 15, 11, 1},
			{1, 12, 11, 11, 11, 14, 11, 11, 11, 1},
			{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
		};
	Mat regionLabelsExpected(16, 10, CV_32S, &regionLabelsData);

	auto& regionLabels = granDist.getRegionLabels();

	BOOST_CHECK_EQUAL(regionLabels.size(), regionLabelsExpected.size());
	for (int row = 0; row < regionLabels.rows; row++) {
		for (int col = 0; col < regionLabels.cols; col++) {
			BOOST_CHECK_EQUAL(regionLabels.at<MAT_TYPE_INT>(row, col), regionLabelsExpected.at<MAT_TYPE_INT>(row, col));
		}
	}

	//------------------------------------------------------------
	auto& downFlowPatches = granDist.getDownFlowPatches();
	BOOST_CHECK_EQUAL(downFlowPatches.size(), 4);
	BOOST_CHECK(downFlowPatches.find(7) != downFlowPatches.end());
	BOOST_CHECK(downFlowPatches.find(8) != downFlowPatches.end());
	BOOST_CHECK(downFlowPatches.find(13) != downFlowPatches.end());
	BOOST_CHECK(downFlowPatches.find(15) != downFlowPatches.end());

	//------------------------------------------------------------
	auto& regionsOnBoundaries = granDist.getRegionsOnBoundaries();
	BOOST_CHECK_EQUAL(regionsOnBoundaries.size(), 7);
	BOOST_CHECK(regionsOnBoundaries.find(1) != regionsOnBoundaries.end());
	BOOST_CHECK(regionsOnBoundaries.find(2) != regionsOnBoundaries.end());
	BOOST_CHECK(regionsOnBoundaries.find(5) != regionsOnBoundaries.end());
	BOOST_CHECK(regionsOnBoundaries.find(6) != regionsOnBoundaries.end());
	BOOST_CHECK(regionsOnBoundaries.find(11) != regionsOnBoundaries.end());
	BOOST_CHECK(regionsOnBoundaries.find(12) != regionsOnBoundaries.end());
	BOOST_CHECK(regionsOnBoundaries.find(14) != regionsOnBoundaries.end());
}
