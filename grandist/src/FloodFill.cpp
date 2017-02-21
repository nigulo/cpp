/*
 * FloodFill.cpp
 *
 *  Created on: 21 Feb 2017
 *      Author: olspern1
 */

#include "FloodFill.h"
#include "common.h"

FloodFill::FloodFill(const Mat& mat, function<bool(float, float)> compFunc) :
		mat(mat),
		labels(Mat::zeros(mat.rows, mat.cols, CV_32S)),
		compFunc(compFunc),
		label(1)
{

}

FloodFill::~FloodFill() {
}

void FloodFill::fillConnectedRegion(const int row, const int col) {
	//cout << "markClosedRegion " << label << " " << row << " " << col << endl;
	auto startEndCols = fillRow(row, col);
	int startCol = startEndCols.first;
	int endCol = startEndCols.second;
	for (int col1 = startCol; col1 <= endCol; col1++) {
		auto value = mat.at<MAT_TYPE_FLOAT>(row, col1);
		if (row > 0 && labels.at<MAT_TYPE_INT>(row - 1, col1) == 0) {
			if (compFunc(mat.at<MAT_TYPE_FLOAT>(row - 1, col1), value)) {
				fillConnectedRegion(row - 1, col1);
			} else {
				updateClosedRegions(value);
			}
		}
		if (row < mat.rows - 1 && labels.at<MAT_TYPE_INT>(row + 1, col1) == 0) {
			if (compFunc(mat.at<MAT_TYPE_FLOAT>(row + 1, col1), value)) {
				fillConnectedRegion(row + 1, col1);
			} else {
				updateClosedRegions(value);
			}
		}
	}
	//cout << "markClosedRegion end " << label << " " << row << " " << col << endl;
}

pair<int, int> FloodFill::fillRow(const int row, const int col) {
	labels.at<MAT_TYPE_INT>(row, col) = label;
	auto initialValue = mat.at<MAT_TYPE_FLOAT>(row, col);
	auto value = initialValue;
	int startCol;
	for (startCol = col - 1; startCol >= 0; startCol--) {
		auto neighborValue = mat.at<MAT_TYPE_FLOAT>(row, startCol);
		if (compFunc(neighborValue, value)) {
			labels.at<MAT_TYPE_INT>(row, startCol) = label;
			value = neighborValue;
		} else {
			updateClosedRegions(neighborValue);
			break;
		}
	}
	value = initialValue;
	int endCol;
	for (endCol = col + 1; endCol < mat.cols; endCol++) {
		auto neighborValue = mat.at<MAT_TYPE_FLOAT>(row, endCol);
		if (compFunc(neighborValue, value)) {
			labels.at<MAT_TYPE_INT>(row, endCol) = label;
			value = neighborValue;
		} else {
			updateClosedRegions(neighborValue);
			break;
		}
	}
	//cout << "markRow " << (startCol + 1) << " " << (endCol - 1) << endl;
	return make_pair(startCol + 1, endCol - 1);
}

void FloodFill::updateClosedRegions(float neighborValue) {
	auto i = neighbors.find(label);
	if (i == neighbors.end()) {
		neighbors[label] = neighborValue;
	} else if (i->second != neighborValue) {
		closedRegions.erase(i->first);
	}
}

void FloodFill::fill(const int row, const int col) {
	if (labels.at<MAT_TYPE_INT>(row, col) == 0) {
		closedRegions.insert(label);
		fillConnectedRegion(row, col);
		label++;
	}
}
