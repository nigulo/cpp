#include <turtle/mock.hpp>

#include "../src/FloodFill.h"
#include "../src/common.h"
#include <iostream>

using namespace std;

BOOST_AUTO_TEST_CASE(test_fill_equal)
{
	float data[7][8] = {
			{0, 0, 0, 0, 0, 0, 0, 3},
			{0, 1, 1, 1, 0, 3, 3, 0},
			{0, 1, 1, 1, 0, 3, 1, 0},
			{0, 1, 0, 1, 0, 0, 0, 0},
			{0, 0, 0, 0, 0, 0, 2, 0},
			{4, 4, 0, 0, 1, 0, 2, 0},
			{0, 4, 0, 0, 0, 0, 0, 0},
	};
	Mat mat(7, 8, CV_32F, &data);
	int timeMoment = 0;
	int layer = 0;
	bool periodic = true;
	Rect cropRect(1, 1, 3, 3);
	FloodFill floodFill(mat);

	floodFill.fill(0, 0);

	int labelsData1[7][8] = {
			{1, 1, 1, 1, 1, 1, 1, 0},
			{1, 0, 0, 0, 1, 0, 0, 1},
			{1, 0, 0, 0, 1, 0, 0, 1},
			{1, 0, 1, 0, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 0, 1},
			{0, 0, 1, 1, 0, 1, 0, 1},
			{0, 0, 1, 1, 1, 1, 1, 1},
	};
	Mat labelsExpected1(7, 8, CV_32S, &labelsData1);

	Mat labels = floodFill.getLabels();
	BOOST_CHECK_EQUAL(labels.rows, labelsExpected1.rows);
	BOOST_CHECK_EQUAL(labels.cols, labelsExpected1.cols);


	for (int row = 0; row < labels.rows; row++) {
		for (int col = 0; col < labels.cols; col++) {
			BOOST_CHECK_EQUAL(labels.at<MAT_TYPE_INT>(row, col), labelsExpected1.at<MAT_TYPE_INT>(row, col));
		}
	}

	//------------------------------------------------------------
	floodFill.fill(3, 3);

	int labelsData2[7][8] = {
			{1, 1, 1, 1, 1, 1, 1, 0},
			{1, 2, 2, 2, 1, 0, 0, 1},
			{1, 2, 2, 2, 1, 0, 0, 1},
			{1, 2, 1, 2, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 0, 1},
			{0, 0, 1, 1, 0, 1, 0, 1},
			{0, 0, 1, 1, 1, 1, 1, 1},
	};
	Mat labelsExpected2(7, 8, CV_32S, &labelsData2);

	labels = floodFill.getLabels();
	BOOST_CHECK_EQUAL(labels.rows, labelsExpected2.rows);
	BOOST_CHECK_EQUAL(labels.cols, labelsExpected2.cols);


	for (int row = 0; row < labels.rows; row++) {
		for (int col = 0; col < labels.cols; col++) {
			BOOST_CHECK_EQUAL(labels.at<MAT_TYPE_INT>(row, col), labelsExpected2.at<MAT_TYPE_INT>(row, col));
		}
	}

	//------------------------------------------------------------
	floodFill.fill(6, 0);

	int labelsData3[7][8] = {
			{1, 1, 1, 1, 1, 1, 1, 0},
			{1, 2, 2, 2, 1, 0, 0, 1},
			{1, 2, 2, 2, 1, 0, 0, 1},
			{1, 2, 1, 2, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 0, 1},
			{0, 0, 1, 1, 0, 1, 0, 1},
			{3, 0, 1, 1, 1, 1, 1, 1},
	};
	Mat labelsExpected3(7, 8, CV_32S, &labelsData3);

	labels = floodFill.getLabels();
	BOOST_CHECK_EQUAL(labels.rows, labelsExpected3.rows);
	BOOST_CHECK_EQUAL(labels.cols, labelsExpected3.cols);


	for (int row = 0; row < labels.rows; row++) {
		for (int col = 0; col < labels.cols; col++) {
			BOOST_CHECK_EQUAL(labels.at<MAT_TYPE_INT>(row, col), labelsExpected3.at<MAT_TYPE_INT>(row, col));
		}
	}

	//------------------------------------------------------------
	floodFill.fill(6, 1);

	int labelsData4[7][8] = {
			{1, 1, 1, 1, 1, 1, 1, 0},
			{1, 2, 2, 2, 1, 0, 0, 1},
			{1, 2, 2, 2, 1, 0, 0, 1},
			{1, 2, 1, 2, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 0, 1},
			{4, 4, 1, 1, 0, 1, 0, 1},
			{3, 4, 1, 1, 1, 1, 1, 1},
	};
	Mat labelsExpected4(7, 8, CV_32S, &labelsData4);

	labels = floodFill.getLabels();
	BOOST_CHECK_EQUAL(labels.rows, labelsExpected4.rows);
	BOOST_CHECK_EQUAL(labels.cols, labelsExpected4.cols);


	for (int row = 0; row < labels.rows; row++) {
		for (int col = 0; col < labels.cols; col++) {
			BOOST_CHECK_EQUAL(labels.at<MAT_TYPE_INT>(row, col), labelsExpected4.at<MAT_TYPE_INT>(row, col));
		}
	}

	//------------------------------------------------------------
	floodFill.fill(1, 5);

	int labelsData5[7][8] = {
			{1, 1, 1, 1, 1, 1, 1, 0},
			{1, 2, 2, 2, 1, 5, 5, 1},
			{1, 2, 2, 2, 1, 5, 0, 1},
			{1, 2, 1, 2, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 0, 1},
			{4, 4, 1, 1, 0, 1, 0, 1},
			{3, 4, 1, 1, 1, 1, 1, 1},
	};
	Mat labelsExpected5(7, 8, CV_32S, &labelsData5);

	labels = floodFill.getLabels();
	BOOST_CHECK_EQUAL(labels.rows, labelsExpected5.rows);
	BOOST_CHECK_EQUAL(labels.cols, labelsExpected5.cols);


	for (int row = 0; row < labels.rows; row++) {
		for (int col = 0; col < labels.cols; col++) {
			BOOST_CHECK_EQUAL(labels.at<MAT_TYPE_INT>(row, col), labelsExpected5.at<MAT_TYPE_INT>(row, col));
		}
	}

	//------------------------------------------------------------
	floodFill.fill(5, 4);

	int labelsData6[7][8] = {
			{1, 1, 1, 1, 1, 1, 1, 0},
			{1, 2, 2, 2, 1, 5, 5, 1},
			{1, 2, 2, 2, 1, 5, 0, 1},
			{1, 2, 1, 2, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 0, 1},
			{4, 4, 1, 1, 6, 1, 0, 1},
			{3, 4, 1, 1, 1, 1, 1, 1},
	};
	Mat labelsExpected6(7, 8, CV_32S, &labelsData6);

	labels = floodFill.getLabels();
	BOOST_CHECK_EQUAL(labels.rows, labelsExpected6.rows);
	BOOST_CHECK_EQUAL(labels.cols, labelsExpected6.cols);


	for (int row = 0; row < labels.rows; row++) {
		for (int col = 0; col < labels.cols; col++) {
			BOOST_CHECK_EQUAL(labels.at<MAT_TYPE_INT>(row, col), labelsExpected6.at<MAT_TYPE_INT>(row, col));
		}
	}

	//------------------------------------------------------------
	floodFill.fill(0, 7);

	int labelsData7[7][8] = {
			{1, 1, 1, 1, 1, 1, 1, 7},
			{1, 2, 2, 2, 1, 5, 5, 1},
			{1, 2, 2, 2, 1, 5, 0, 1},
			{1, 2, 1, 2, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 0, 1},
			{4, 4, 1, 1, 6, 1, 0, 1},
			{3, 4, 1, 1, 1, 1, 1, 1},
	};
	Mat labelsExpected7(7, 8, CV_32S, &labelsData7);

	labels = floodFill.getLabels();
	BOOST_CHECK_EQUAL(labels.rows, labelsExpected7.rows);
	BOOST_CHECK_EQUAL(labels.cols, labelsExpected7.cols);


	for (int row = 0; row < labels.rows; row++) {
		for (int col = 0; col < labels.cols; col++) {
			BOOST_CHECK_EQUAL(labels.at<MAT_TYPE_INT>(row, col), labelsExpected7.at<MAT_TYPE_INT>(row, col));
		}
	}

	//------------------------------------------------------------
	floodFill.fill(4, 6);

	int labelsData8[7][8] = {
			{1, 1, 1, 1, 1, 1, 1, 7},
			{1, 2, 2, 2, 1, 5, 5, 1},
			{1, 2, 2, 2, 1, 5, 0, 1},
			{1, 2, 1, 2, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 8, 1},
			{4, 4, 1, 1, 6, 1, 8, 1},
			{3, 4, 1, 1, 1, 1, 1, 1},
	};
	Mat labelsExpected8(7, 8, CV_32S, &labelsData8);

	labels = floodFill.getLabels();
	BOOST_CHECK_EQUAL(labels.rows, labelsExpected8.rows);
	BOOST_CHECK_EQUAL(labels.cols, labelsExpected8.cols);


	for (int row = 0; row < labels.rows; row++) {
		for (int col = 0; col < labels.cols; col++) {
			BOOST_CHECK_EQUAL(labels.at<MAT_TYPE_INT>(row, col), labelsExpected8.at<MAT_TYPE_INT>(row, col));
		}
	}

	//------------------------------------------------------------
	floodFill.fill(2, 6);

	int labelsData9[7][8] = {
			{1, 1, 1, 1, 1, 1, 1, 7},
			{1, 2, 2, 2, 1, 5, 5, 1},
			{1, 2, 2, 2, 1, 5, 9, 1},
			{1, 2, 1, 2, 1, 1, 1, 1},
			{1, 1, 1, 1, 1, 1, 8, 1},
			{4, 4, 1, 1, 6, 1, 8, 1},
			{3, 4, 1, 1, 1, 1, 1, 1},
	};
	Mat labelsExpected9(7, 8, CV_32S, &labelsData9);

	labels = floodFill.getLabels();
	BOOST_CHECK_EQUAL(labels.rows, labelsExpected9.rows);
	BOOST_CHECK_EQUAL(labels.cols, labelsExpected9.cols);


	for (int row = 0; row < labels.rows; row++) {
		for (int col = 0; col < labels.cols; col++) {
			BOOST_CHECK_EQUAL(labels.at<MAT_TYPE_INT>(row, col), labelsExpected9.at<MAT_TYPE_INT>(row, col));
		}
	}

	//------------------------------------------------------------
	floodFill.fill(4, 6); // Check if second try has no effect

	labels = floodFill.getLabels();
	BOOST_CHECK_EQUAL(labels.rows, labelsExpected9.rows);
	BOOST_CHECK_EQUAL(labels.cols, labelsExpected9.cols);


	for (int row = 0; row < labels.rows; row++) {
		for (int col = 0; col < labels.cols; col++) {
			BOOST_CHECK_EQUAL(labels.at<MAT_TYPE_INT>(row, col), labelsExpected9.at<MAT_TYPE_INT>(row, col));
		}
	}

	//------------------------------------------------------------
	BOOST_CHECK_EQUAL(floodFill.getNumRegions(), 9);

	set<int> closedRegionsExpected;
	closedRegionsExpected.insert(2);
	closedRegionsExpected.insert(3);
	closedRegionsExpected.insert(4);
	closedRegionsExpected.insert(6);
	closedRegionsExpected.insert(7);
	closedRegionsExpected.insert(8);

	auto& closedRegions = floodFill.getClosedRegions();
	BOOST_CHECK_EQUAL(closedRegions.size(), closedRegionsExpected.size());
	{
		auto j = closedRegionsExpected.begin();
		for (auto i = closedRegions.begin(); i != closedRegions.end(); i++, j++) {
			BOOST_CHECK_EQUAL(*i, *j);
		}
	}

	//------------------------------------------------------------
	map<int /*label*/, int /*area*/> regionAreasExpected;
	regionAreasExpected[1] = 36;
	regionAreasExpected[2] = 8;
	regionAreasExpected[3] = 1;
	regionAreasExpected[4] = 3;
	regionAreasExpected[5] = 3;
	regionAreasExpected[6] = 1;
	regionAreasExpected[7] = 1;
	regionAreasExpected[8] = 2;
	regionAreasExpected[9] = 1;

	auto& regionAreas = floodFill.getRegionAreas();
	BOOST_CHECK_EQUAL(regionAreas.size(), regionAreasExpected.size());
	{
		auto j = regionAreasExpected.begin();
		for (auto i = regionAreas.begin(); i != regionAreas.end(); i++, j++) {
			BOOST_CHECK_EQUAL(i->first, j->first);
			BOOST_CHECK_EQUAL(i->second, j->second);
		}
	}


	//------------------------------------------------------------
	map<int /*label*/, tuple<int /*minRow*/, int /*maxRow*/, int /*minCol*/, int /*maxCol*/>> regionExtentsExpected;
	regionExtentsExpected[1] = make_tuple(0, 6, 0, 7);
	regionExtentsExpected[2] = make_tuple(1, 3, 1, 3);
	regionExtentsExpected[3] = make_tuple(6, 6, 0, 0);
	regionExtentsExpected[4] = make_tuple(5, 6, 0, 1);
	regionExtentsExpected[5] = make_tuple(1, 2, 5, 6);
	regionExtentsExpected[6] = make_tuple(5, 5, 4, 4);
	regionExtentsExpected[7] = make_tuple(0, 0, 7, 7);
	regionExtentsExpected[8] = make_tuple(4, 5, 6, 6);
	regionExtentsExpected[9] = make_tuple(2, 2, 6, 6);

	auto& regionExtents = floodFill.getRegionExtents();
	BOOST_CHECK_EQUAL(regionExtents.size(), regionExtentsExpected.size());
	{
		auto j = regionExtentsExpected.begin();
		for (auto i = regionExtents.begin(); i != regionExtents.end(); i++, j++) {
			BOOST_CHECK_EQUAL(i->first, j->first);
			BOOST_CHECK_EQUAL(get<0>(i->second), get<0>(j->second));
			BOOST_CHECK_EQUAL(get<1>(i->second), get<1>(j->second));
			BOOST_CHECK_EQUAL(get<2>(i->second), get<2>(j->second));
			BOOST_CHECK_EQUAL(get<3>(i->second), get<3>(j->second));
		}
	}
}
