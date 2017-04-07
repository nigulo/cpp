#include <turtle/mock.hpp>

#include "../src/BoundaryDetect.h"
#include "../src/common.h"

BOOST_AUTO_TEST_CASE(test_detect_1_point_mode_8)
{
	float data[3][3] = {
			{0, 0, 0},
			{0, 1, 0},
			{0, 0, 0}
	};
	Mat mat(3, 3, CV_32F, &data);
	BoundaryDetect bd(mat, CONNECTIVITY_MODE_8);
	bd.detect(1, 1);

	vector<Point2i> expectedInner;
	expectedInner.push_back(Point2d(1, 1));

	BOOST_CHECK_EQUAL(bd.getInner().size(), expectedInner.size());
	for (size_t i = 0; i < bd.getInner().size(); i++) {
		BOOST_CHECK_EQUAL(bd.getInner()[i], expectedInner[i]);
	}

	vector<Point2i> expectedOuter;
	expectedOuter.push_back(Point2d(0, 2));
	expectedOuter.push_back(Point2d(1, 2));
	expectedOuter.push_back(Point2d(2, 2));
	expectedOuter.push_back(Point2d(2, 1));
	expectedOuter.push_back(Point2d(2, 0));
	expectedOuter.push_back(Point2d(1, 0));
	expectedOuter.push_back(Point2d(0, 0));
	expectedOuter.push_back(Point2d(0, 1));
	BOOST_CHECK_EQUAL(bd.getOuter().size(), expectedOuter.size());
	for (size_t i = 0; i < bd.getOuter().size(); i++) {
		BOOST_CHECK_EQUAL(bd.getOuter()[i], expectedOuter[i]);
	}

	Mat innerBoundariesExpected(3, 3, CV_32F, &data);

	float dataOuterBoundaries[5][5] = {
			{0, 0, 0, 0, 0},
			{0, 1, 1, 1, 0},
			{0, 1, 0, 1, 0},
			{0, 1, 1, 1, 0},
			{0, 0, 0, 0, 0},
	};

	Mat outerBoundariesExpected(5, 5, CV_32F, &dataOuterBoundaries);
	auto& innerBoundaries = bd.getInnerBoundaries();
	BOOST_CHECK_EQUAL(innerBoundaries.rows, innerBoundariesExpected.rows);
	BOOST_CHECK_EQUAL(innerBoundaries.cols, innerBoundariesExpected.cols);
	for (int row = 0; row < innerBoundaries.rows; row++) {
		for (int col = 0; col < innerBoundaries.cols; col++) {
			BOOST_CHECK_EQUAL(innerBoundaries.at<MAT_TYPE_FLOAT>(row, col), innerBoundariesExpected.at<MAT_TYPE_FLOAT>(row, col));
		}
	}

	auto& outerBoundaries = bd.getOuterBoundaries();
	BOOST_CHECK_EQUAL(outerBoundaries.rows, outerBoundariesExpected.rows);
	BOOST_CHECK_EQUAL(outerBoundaries.cols, outerBoundariesExpected.cols);
	for (int row = 0; row < outerBoundaries.rows; row++) {
		for (int col = 0; col < outerBoundaries.cols; col++) {
			BOOST_CHECK_EQUAL(outerBoundaries.at<MAT_TYPE_FLOAT>(row, col), outerBoundariesExpected.at<MAT_TYPE_FLOAT>(row, col));
		}
	}

}

BOOST_AUTO_TEST_CASE(test_detect_1_point_mode_4)
{
	float data[3][3] = {
			{0, 0, 0},
			{0, 1, 0},
			{0, 0, 0}
	};
	Mat mat(3, 3, CV_32F, &data);
	BoundaryDetect bd(mat, CONNECTIVITY_MODE_4);
	bd.detect(1, 1);

	vector<Point2i> expectedInner;
	expectedInner.push_back(Point2d(1, 1));

	BOOST_CHECK_EQUAL(bd.getInner().size(), expectedInner.size());
	for (size_t i = 0; i < bd.getInner().size(); i++) {
		BOOST_CHECK_EQUAL(bd.getInner()[i], expectedInner[i]);
	}

	vector<Point2i> expectedOuter;
	expectedOuter.push_back(Point2d(1, 2));
	expectedOuter.push_back(Point2d(2, 1));
	expectedOuter.push_back(Point2d(1, 0));
	expectedOuter.push_back(Point2d(0, 1));
	BOOST_CHECK_EQUAL(bd.getOuter().size(), expectedOuter.size());
	for (size_t i = 0; i < bd.getOuter().size(); i++) {
		BOOST_CHECK_EQUAL(bd.getOuter()[i], expectedOuter[i]);
	}

	Mat innerBoundariesExpected(3, 3, CV_32F, &data);

	float dataOuterBoundaries[5][5] = {
			{0, 0, 0, 0, 0},
			{0, 0, 1, 0, 0},
			{0, 1, 0, 1, 0},
			{0, 0, 1, 0, 0},
			{0, 0, 0, 0, 0},
	};

	Mat outerBoundariesExpected(5, 5, CV_32F, &dataOuterBoundaries);
	auto& innerBoundaries = bd.getInnerBoundaries();
	BOOST_CHECK_EQUAL(innerBoundaries.rows, innerBoundariesExpected.rows);
	BOOST_CHECK_EQUAL(innerBoundaries.cols, innerBoundariesExpected.cols);
	for (int row = 0; row < innerBoundaries.rows; row++) {
		for (int col = 0; col < innerBoundaries.cols; col++) {
			BOOST_CHECK_EQUAL(innerBoundaries.at<MAT_TYPE_FLOAT>(row, col), innerBoundariesExpected.at<MAT_TYPE_FLOAT>(row, col));
		}
	}

	auto& outerBoundaries = bd.getOuterBoundaries();
	BOOST_CHECK_EQUAL(outerBoundaries.rows, outerBoundariesExpected.rows);
	BOOST_CHECK_EQUAL(outerBoundaries.cols, outerBoundariesExpected.cols);
	for (int row = 0; row < outerBoundaries.rows; row++) {
		for (int col = 0; col < outerBoundaries.cols; col++) {
			BOOST_CHECK_EQUAL(outerBoundaries.at<MAT_TYPE_FLOAT>(row, col), outerBoundariesExpected.at<MAT_TYPE_FLOAT>(row, col));
		}
	}

}

BOOST_AUTO_TEST_CASE(test_detect_1_region_mode_8)
{
	float data[7][8] = {
			{0, 0, 0, 0, 0, 0, 0, 0},
			{0, 1, 1, 0, 1, 0, 0, 0},
			{0, 0, 1, 1, 1, 1, 0, 0},
			{0, 0, 1, 1, 1, 1, 0, 0},
			{0, 1, 1, 1, 1, 1, 0, 0},
			{0, 0, 1, 1, 0, 1, 1, 0},
			{0, 0, 0, 0, 0, 0, 0, 0}
	};
	Mat mat(7, 8, CV_32F, &data);
	BoundaryDetect bd(mat, CONNECTIVITY_MODE_8);
	bd.detect(2, 3);

	vector<Point2i> expectedInner;
	expectedInner.push_back(Point2d(2, 2));
	expectedInner.push_back(Point2d(2, 3));
	expectedInner.push_back(Point2d(1, 4));
	expectedInner.push_back(Point2d(2, 5));
	expectedInner.push_back(Point2d(3, 5));
	expectedInner.push_back(Point2d(4, 4));
	expectedInner.push_back(Point2d(5, 5));
	expectedInner.push_back(Point2d(6, 5));
	expectedInner.push_back(Point2d(5, 4));
	expectedInner.push_back(Point2d(5, 3));
	expectedInner.push_back(Point2d(5, 2));
	expectedInner.push_back(Point2d(4, 1));
	expectedInner.push_back(Point2d(3, 2));
	expectedInner.push_back(Point2d(2, 1));
	expectedInner.push_back(Point2d(1, 1));
	BOOST_CHECK_EQUAL(bd.getInner().size(), expectedInner.size());
	for (size_t i = 0; i < bd.getInner().size(); i++) {
		BOOST_CHECK_EQUAL(bd.getInner()[i], expectedInner[i]);
	}

	vector<Point2i> expectedOuter;
	expectedOuter.push_back(Point2d(1, 3));
	expectedOuter.push_back(Point2d(0, 3));
	expectedOuter.push_back(Point2d(0, 4));
	expectedOuter.push_back(Point2d(0, 5));
	expectedOuter.push_back(Point2d(1, 5));
	expectedOuter.push_back(Point2d(1, 6));
	expectedOuter.push_back(Point2d(2, 6));
	expectedOuter.push_back(Point2d(3, 6));
	expectedOuter.push_back(Point2d(4, 6));
	expectedOuter.push_back(Point2d(4, 5));
	expectedOuter.push_back(Point2d(5, 6));
	expectedOuter.push_back(Point2d(6, 6));
	expectedOuter.push_back(Point2d(7, 6));
	expectedOuter.push_back(Point2d(7, 5));
	expectedOuter.push_back(Point2d(7, 4));
	expectedOuter.push_back(Point2d(6, 4));
	expectedOuter.push_back(Point2d(6, 3));
	expectedOuter.push_back(Point2d(6, 2));
	expectedOuter.push_back(Point2d(6, 1));
	expectedOuter.push_back(Point2d(5, 1));
	expectedOuter.push_back(Point2d(5, 0));
	expectedOuter.push_back(Point2d(4, 0));
	expectedOuter.push_back(Point2d(3, 0));
	expectedOuter.push_back(Point2d(3, 1));
	expectedOuter.push_back(Point2d(2, 0));
	expectedOuter.push_back(Point2d(1, 0));
	expectedOuter.push_back(Point2d(0, 0));
	expectedOuter.push_back(Point2d(0, 1));
	expectedOuter.push_back(Point2d(0, 2));
	expectedOuter.push_back(Point2d(1, 2));
	BOOST_CHECK_EQUAL(bd.getOuter().size(), expectedOuter.size());
	for (size_t i = 0; i < bd.getOuter().size(); i++) {
		BOOST_CHECK_EQUAL(bd.getOuter()[i], expectedOuter[i]);
	}

	float dataInnerBoundaries[7][8] = {
			{0, 0, 0, 0, 0, 0, 0, 0},
			{0, 1, 1, 0, 1, 0, 0, 0},
			{0, 0, 1, 1, 0, 1, 0, 0},
			{0, 0, 1, 0, 0, 1, 0, 0},
			{0, 1, 0, 0, 1, 1, 0, 0},
			{0, 0, 1, 1, 0, 1, 1, 0},
			{0, 0, 0, 0, 0, 0, 0, 0}
	};
	Mat innerBoundariesExpected(7, 8, CV_32F, &dataInnerBoundaries);

	float dataOuterBoundaries[9][10] = {
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{0, 1, 1, 1, 1, 1, 1, 0, 0, 0},
			{0, 1, 0, 0, 1, 0, 1, 1, 0, 0},
			{0, 1, 1, 0, 0, 0, 0, 1, 0, 0},
			{0, 1, 1, 0, 0, 0, 0, 1, 0, 0},
			{0, 1, 0, 0, 0, 0, 0, 1, 1, 0},
			{0, 1, 1, 0, 0, 1, 0, 0, 1, 0},
			{0, 0, 1, 1, 1, 1, 1, 1, 1, 0},
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	};
	Mat outerBoundariesExpected(9, 10, CV_32F, &dataOuterBoundaries);

	auto& innerBoundaries = bd.getInnerBoundaries();
	BOOST_CHECK_EQUAL(innerBoundaries.rows, innerBoundariesExpected.rows);
	BOOST_CHECK_EQUAL(innerBoundaries.cols, innerBoundariesExpected.cols);
	for (int row = 0; row < innerBoundaries.rows; row++) {
		for (int col = 0; col < innerBoundaries.cols; col++) {
			BOOST_CHECK_EQUAL(innerBoundaries.at<MAT_TYPE_FLOAT>(row, col), innerBoundariesExpected.at<MAT_TYPE_FLOAT>(row, col));
		}
	}

	auto& outerBoundaries = bd.getOuterBoundaries();
	BOOST_CHECK_EQUAL(outerBoundaries.rows, outerBoundariesExpected.rows);
	BOOST_CHECK_EQUAL(outerBoundaries.cols, outerBoundariesExpected.cols);
	for (int row = 0; row < outerBoundaries.rows; row++) {
		for (int col = 0; col < outerBoundaries.cols; col++) {
			BOOST_CHECK_EQUAL(outerBoundaries.at<MAT_TYPE_FLOAT>(row, col), outerBoundariesExpected.at<MAT_TYPE_FLOAT>(row, col));
		}
	}

}

BOOST_AUTO_TEST_CASE(test_detect_1_region_mode_4)
{
	float data[7][8] = {
			{0, 0, 0, 0, 0, 0, 0, 0},
			{0, 1, 1, 0, 1, 0, 0, 0},
			{0, 0, 1, 1, 1, 1, 0, 0},
			{0, 0, 1, 1, 1, 1, 0, 0},
			{0, 1, 1, 1, 1, 1, 0, 0},
			{0, 0, 1, 1, 0, 1, 1, 0},
			{0, 0, 0, 0, 0, 0, 0, 0}
	};
	Mat mat(7, 8, CV_32F, &data);
	BoundaryDetect bd(mat, CONNECTIVITY_MODE_4);
	bd.detect(2, 3);

	vector<Point2i> expectedInner;
	expectedInner.push_back(Point2d(2, 2));
	expectedInner.push_back(Point2d(2, 3));
	expectedInner.push_back(Point2d(2, 4));
	expectedInner.push_back(Point2d(1, 4));
	expectedInner.push_back(Point2d(2, 5));
	expectedInner.push_back(Point2d(3, 5));
	expectedInner.push_back(Point2d(3, 4));
	expectedInner.push_back(Point2d(4, 4));
	expectedInner.push_back(Point2d(5, 4));
	expectedInner.push_back(Point2d(5, 5));
	expectedInner.push_back(Point2d(6, 5));
	expectedInner.push_back(Point2d(5, 3));
	expectedInner.push_back(Point2d(5, 2));
	expectedInner.push_back(Point2d(4, 2));
	expectedInner.push_back(Point2d(4, 1));
	expectedInner.push_back(Point2d(3, 2));
	expectedInner.push_back(Point2d(2, 1));
	expectedInner.push_back(Point2d(1, 1));
	BOOST_CHECK_EQUAL(bd.getInner().size(), expectedInner.size());
	for (size_t i = 0; i < bd.getInner().size(); i++) {
		BOOST_CHECK_EQUAL(bd.getInner()[i], expectedInner[i]);
	}

	vector<Point2i> expectedOuter;
	expectedOuter.push_back(Point2d(1, 3));
	expectedOuter.push_back(Point2d(0, 4));
	expectedOuter.push_back(Point2d(1, 5));
	expectedOuter.push_back(Point2d(2, 6));
	expectedOuter.push_back(Point2d(3, 6));
	expectedOuter.push_back(Point2d(4, 5));
	expectedOuter.push_back(Point2d(5, 6));
	expectedOuter.push_back(Point2d(6, 6));
	expectedOuter.push_back(Point2d(7, 5));
	expectedOuter.push_back(Point2d(6, 4));
	expectedOuter.push_back(Point2d(6, 3));
	expectedOuter.push_back(Point2d(6, 2));
	expectedOuter.push_back(Point2d(5, 1));
	expectedOuter.push_back(Point2d(4, 0));
	expectedOuter.push_back(Point2d(3, 1));
	expectedOuter.push_back(Point2d(2, 0));
	expectedOuter.push_back(Point2d(1, 0));
	expectedOuter.push_back(Point2d(0, 1));
	expectedOuter.push_back(Point2d(1, 2));
	BOOST_CHECK_EQUAL(bd.getOuter().size(), expectedOuter.size());
	for (size_t i = 0; i < bd.getOuter().size(); i++) {
		BOOST_CHECK_EQUAL(bd.getOuter()[i], expectedOuter[i]);
	}

	float dataInnerBoundaries[7][8] = {
			{0, 0, 0, 0, 0, 0, 0, 0},
			{0, 1, 1, 0, 1, 0, 0, 0},
			{0, 0, 1, 1, 1, 1, 0, 0},
			{0, 0, 1, 0, 0, 1, 0, 0},
			{0, 1, 1, 1, 1, 1, 0, 0},
			{0, 0, 1, 1, 0, 1, 1, 0},
			{0, 0, 0, 0, 0, 0, 0, 0}
	};
	Mat innerBoundariesExpected(7, 8, CV_32F, &dataInnerBoundaries);

	float dataOuterBoundaries[9][10] = {
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{0, 0, 1, 1, 0, 1, 0, 0, 0, 0},
			{0, 1, 0, 0, 1, 0, 1, 0, 0, 0},
			{0, 0, 1, 0, 0, 0, 0, 1, 0, 0},
			{0, 0, 1, 0, 0, 0, 0, 1, 0, 0},
			{0, 1, 0, 0, 0, 0, 0, 1, 0, 0},
			{0, 0, 1, 0, 0, 1, 0, 0, 1, 0},
			{0, 0, 0, 1, 1, 0, 1, 1, 0, 0},
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	};
	Mat outerBoundariesExpected(9, 10, CV_32F, &dataOuterBoundaries);

	auto& innerBoundaries = bd.getInnerBoundaries();
	BOOST_CHECK_EQUAL(innerBoundaries.rows, innerBoundariesExpected.rows);
	BOOST_CHECK_EQUAL(innerBoundaries.cols, innerBoundariesExpected.cols);
	for (int row = 0; row < innerBoundaries.rows; row++) {
		for (int col = 0; col < innerBoundaries.cols; col++) {
			BOOST_CHECK_EQUAL(innerBoundaries.at<MAT_TYPE_FLOAT>(row, col), innerBoundariesExpected.at<MAT_TYPE_FLOAT>(row, col));
		}
	}

	auto& outerBoundaries = bd.getOuterBoundaries();
	BOOST_CHECK_EQUAL(outerBoundaries.rows, outerBoundariesExpected.rows);
	BOOST_CHECK_EQUAL(outerBoundaries.cols, outerBoundariesExpected.cols);
	for (int row = 0; row < outerBoundaries.rows; row++) {
		for (int col = 0; col < outerBoundaries.cols; col++) {
			BOOST_CHECK_EQUAL(outerBoundaries.at<MAT_TYPE_FLOAT>(row, col), outerBoundariesExpected.at<MAT_TYPE_FLOAT>(row, col));
		}
	}
}




BOOST_AUTO_TEST_CASE(test_detect_2_regions_mode_8)
{
	float data[5][6] = {
			{0, 0, 0, 0, 0, 0},
			{0, 1, 1, 0, 0, 0},
			{0, 0, 0, 0, 0, 0},
			{0, 0, 0, 1, 1, 0},
			{0, 0, 0, 0, 1, 0}
	};
	Mat mat(5, 6, CV_32F, &data);
	BoundaryDetect bd(mat, CONNECTIVITY_MODE_8);
	bd.detect(1, 1);

	vector<Point2i> expectedInner;
	expectedInner.push_back(Point2d(1, 1));
	expectedInner.push_back(Point2d(2, 1));

	BOOST_CHECK_EQUAL(bd.getInner().size(), expectedInner.size());
	for (size_t i = 0; i < bd.getInner().size(); i++) {
		BOOST_CHECK_EQUAL(bd.getInner()[i], expectedInner[i]);
	}

	vector<Point2i> expectedOuter;
	expectedOuter.push_back(Point2d(0, 2));
	expectedOuter.push_back(Point2d(1, 2));
	expectedOuter.push_back(Point2d(2, 2));
	expectedOuter.push_back(Point2d(3, 2));
	expectedOuter.push_back(Point2d(3, 1));
	expectedOuter.push_back(Point2d(3, 0));
	expectedOuter.push_back(Point2d(2, 0));
	expectedOuter.push_back(Point2d(1, 0));
	expectedOuter.push_back(Point2d(0, 0));
	expectedOuter.push_back(Point2d(0, 1));
	BOOST_CHECK_EQUAL(bd.getOuter().size(), expectedOuter.size());
	for (size_t i = 0; i < bd.getOuter().size(); i++) {
		BOOST_CHECK_EQUAL(bd.getOuter()[i], expectedOuter[i]);
	}

	bd.detect(4, 4);

	expectedInner.clear();
	expectedInner.push_back(Point2d(4, 4));
	expectedInner.push_back(Point2d(4, 3));
	expectedInner.push_back(Point2d(3, 3));

	BOOST_CHECK_EQUAL(bd.getInner().size(), expectedInner.size());
	for (size_t i = 0; i < bd.getInner().size(); i++) {
		BOOST_CHECK_EQUAL(bd.getInner()[i], expectedInner[i]);
	}

	expectedOuter.clear();
	expectedOuter.push_back(Point2d(3, 5));
	expectedOuter.push_back(Point2d(4, 5));
	expectedOuter.push_back(Point2d(5, 5));
	expectedOuter.push_back(Point2d(5, 4));
	expectedOuter.push_back(Point2d(5, 3));
	expectedOuter.push_back(Point2d(5, 2));
	expectedOuter.push_back(Point2d(4, 2));
	expectedOuter.push_back(Point2d(3, 2));
	expectedOuter.push_back(Point2d(2, 2));
	expectedOuter.push_back(Point2d(2, 3));
	expectedOuter.push_back(Point2d(2, 4));
	expectedOuter.push_back(Point2d(3, 4));
	BOOST_CHECK_EQUAL(bd.getOuter().size(), expectedOuter.size());
	for (size_t i = 0; i < bd.getOuter().size(); i++) {
		BOOST_CHECK_EQUAL(bd.getOuter()[i], expectedOuter[i]);
	}

	Mat innerBoundariesExpected(5, 6, CV_32F, &data);

	float dataOuterBoundaries[7][8] = {
			{0, 0, 0, 0, 0, 0, 0, 0},
			{0, 1, 1, 1, 1, 0, 0, 0},
			{0, 1, 0, 0, 1, 0, 0, 0},
			{0, 1, 1, 1, 1, 1, 1, 0},
			{0, 0, 0, 1, 0, 0, 1, 0},
			{0, 0, 0, 1, 1, 0, 1, 0},
			{0, 0, 0, 0, 1, 1, 1, 0}
	};

	Mat outerBoundariesExpected(7, 8, CV_32F, &dataOuterBoundaries);
	auto& innerBoundaries = bd.getInnerBoundaries();
	BOOST_CHECK_EQUAL(innerBoundaries.rows, innerBoundariesExpected.rows);
	BOOST_CHECK_EQUAL(innerBoundaries.cols, innerBoundariesExpected.cols);
	for (int row = 0; row < innerBoundaries.rows; row++) {
		for (int col = 0; col < innerBoundaries.cols; col++) {
			BOOST_CHECK_EQUAL(innerBoundaries.at<MAT_TYPE_FLOAT>(row, col), innerBoundariesExpected.at<MAT_TYPE_FLOAT>(row, col));
		}
	}

	auto& outerBoundaries = bd.getOuterBoundaries();
	BOOST_CHECK_EQUAL(outerBoundaries.rows, outerBoundariesExpected.rows);
	BOOST_CHECK_EQUAL(outerBoundaries.cols, outerBoundariesExpected.cols);
	for (int row = 0; row < outerBoundaries.rows; row++) {
		for (int col = 0; col < outerBoundaries.cols; col++) {
			BOOST_CHECK_EQUAL(outerBoundaries.at<MAT_TYPE_FLOAT>(row, col), outerBoundariesExpected.at<MAT_TYPE_FLOAT>(row, col));
		}
	}

}

BOOST_AUTO_TEST_CASE(test_detect_2_regions_mode_4)
{
	float data[5][6] = {
			{0, 0, 0, 0, 0, 0},
			{0, 1, 1, 0, 0, 0},
			{0, 0, 0, 0, 0, 0},
			{0, 0, 0, 1, 1, 0},
			{0, 0, 0, 0, 1, 0}
	};
	Mat mat(5, 6, CV_32F, &data);
	BoundaryDetect bd(mat, CONNECTIVITY_MODE_4);
	bd.detect(1, 1);

	vector<Point2i> expectedInner;
	expectedInner.push_back(Point2d(1, 1));
	expectedInner.push_back(Point2d(2, 1));

	BOOST_CHECK_EQUAL(bd.getInner().size(), expectedInner.size());
	for (size_t i = 0; i < bd.getInner().size(); i++) {
		BOOST_CHECK_EQUAL(bd.getInner()[i], expectedInner[i]);
	}

	vector<Point2i> expectedOuter;
	expectedOuter.push_back(Point2d(1, 2));
	expectedOuter.push_back(Point2d(2, 2));
	expectedOuter.push_back(Point2d(3, 1));
	expectedOuter.push_back(Point2d(2, 0));
	expectedOuter.push_back(Point2d(1, 0));
	expectedOuter.push_back(Point2d(0, 1));
	BOOST_CHECK_EQUAL(bd.getOuter().size(), expectedOuter.size());
	for (size_t i = 0; i < bd.getOuter().size(); i++) {
		BOOST_CHECK_EQUAL(bd.getOuter()[i], expectedOuter[i]);
	}

	bd.detect(4, 4);

	expectedInner.clear();
	expectedInner.push_back(Point2d(4, 4));
	expectedInner.push_back(Point2d(4, 3));
	expectedInner.push_back(Point2d(3, 3));

	BOOST_CHECK_EQUAL(bd.getInner().size(), expectedInner.size());
	for (size_t i = 0; i < bd.getInner().size(); i++) {
		BOOST_CHECK_EQUAL(bd.getInner()[i], expectedInner[i]);
	}

	expectedOuter.clear();
	expectedOuter.push_back(Point2d(4, 5));
	expectedOuter.push_back(Point2d(5, 4));
	expectedOuter.push_back(Point2d(5, 3));
	expectedOuter.push_back(Point2d(4, 2));
	expectedOuter.push_back(Point2d(3, 2));
	expectedOuter.push_back(Point2d(2, 3));
	expectedOuter.push_back(Point2d(3, 4));
	BOOST_CHECK_EQUAL(bd.getOuter().size(), expectedOuter.size());
	for (size_t i = 0; i < bd.getOuter().size(); i++) {
		BOOST_CHECK_EQUAL(bd.getOuter()[i], expectedOuter[i]);
	}

	Mat innerBoundariesExpected(5, 6, CV_32F, &data);

	float dataOuterBoundaries[7][8] = {
			{0, 0, 0, 0, 0, 0, 0, 0},
			{0, 0, 1, 1, 0, 0, 0, 0},
			{0, 1, 0, 0, 1, 0, 0, 0},
			{0, 0, 1, 1, 1, 1, 0, 0},
			{0, 0, 0, 1, 0, 0, 1, 0},
			{0, 0, 0, 0, 1, 0, 1, 0},
			{0, 0, 0, 0, 0, 1, 0, 0}
	};

	Mat outerBoundariesExpected(7, 8, CV_32F, &dataOuterBoundaries);
	auto& innerBoundaries = bd.getInnerBoundaries();
	BOOST_CHECK_EQUAL(innerBoundaries.rows, innerBoundariesExpected.rows);
	BOOST_CHECK_EQUAL(innerBoundaries.cols, innerBoundariesExpected.cols);
	for (int row = 0; row < innerBoundaries.rows; row++) {
		for (int col = 0; col < innerBoundaries.cols; col++) {
			BOOST_CHECK_EQUAL(innerBoundaries.at<MAT_TYPE_FLOAT>(row, col), innerBoundariesExpected.at<MAT_TYPE_FLOAT>(row, col));
		}
	}

	auto& outerBoundaries = bd.getOuterBoundaries();
	BOOST_CHECK_EQUAL(outerBoundaries.rows, outerBoundariesExpected.rows);
	BOOST_CHECK_EQUAL(outerBoundaries.cols, outerBoundariesExpected.cols);
	for (int row = 0; row < outerBoundaries.rows; row++) {
		for (int col = 0; col < outerBoundaries.cols; col++) {
			BOOST_CHECK_EQUAL(outerBoundaries.at<MAT_TYPE_FLOAT>(row, col), outerBoundariesExpected.at<MAT_TYPE_FLOAT>(row, col));
		}
	}

}
