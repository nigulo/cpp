/*
 * GranDist.cpp
 *
 *  Created on: 21 Feb 2017
 *      Author: olspern1
 */

#include "GranDist.h"
#include "common.h"
#include "FloodFill.h"
#include "BoundaryDetect.h"
#include <vector>
#ifdef GPU
#include <opencv2/gpu/gpu.hpp>
#endif
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#ifdef OPENCV_2_4
	#include <opencv2/contrib/contrib.hpp>
#endif
#include <functional>
#include <fstream>
#include <iostream>

/**
 * Repeats the matrix twice in horizontal and vertical directions.
 * Needed to correctly estimate the granule sizes on the boundaries
 */
Mat& tileMatrix(Mat& mat, int rows, int cols) {
	int rowOffset = (mat.rows - 2 * rows) / 2;
	int colOffset = (mat.cols - 2 * cols) / 2;

	int midRow = mat.rows / 2;
	int midCol = mat.cols / 2;

	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			auto value = mat.at<MAT_TYPE_FLOAT>(midRow + row, midCol + col);
			mat.at<MAT_TYPE_FLOAT>(rowOffset + row, colOffset + col) = value;
			mat.at<MAT_TYPE_FLOAT>(midRow + row, colOffset + col) = value;
			mat.at<MAT_TYPE_FLOAT>(rowOffset + row, midCol + col) = value;
		}
	}
	return mat;
}

Mat& convertTo8Bit(Mat& mat) {
	double min, max;
	minMaxLoc(mat, &min, &max);
	mat.convertTo(mat, CV_8U, 255.0 / (max - min),-min * 255.0 / (max - min));
	return mat;
}


GranDist::GranDist(int timeMoment, int layer, Mat granules, bool periodic, Rect cropRect, bool saveMaps) :
		timeMoment(timeMoment),
		layer(layer),
		granules(periodic ? tileMatrix(granules, cropRect.height, cropRect.width) : granules),
		originalHeight(cropRect.height),
		originalWidth(cropRect.width),
		periodic(periodic),
		cropRect(cropRect),
		saveMaps(saveMaps) {
	labelRegions();
	regionLabels.convertTo(regionLabelsFloat, CV_32F);
	const auto& closedRegions = getClosedRegions();

	auto regionsOnBoundaries = getRegionsOnBoundaries();

	for (int row = 0; row < regionLabels.rows; row++) {
		for (int col = 0; col < regionLabels.cols; col++) {
			if (granules.at<MAT_TYPE_FLOAT>(row, col) == DOWN_FLOW)  {
				int label = regionLabels.at<MAT_TYPE_INT>(row, col);
				if (closedRegions.find(label) != closedRegions.end() && regionsOnBoundaries.find(label) == regionsOnBoundaries.end()) {
					downFlowPatches.insert(label);
				}
			}
		}
	}

	//////////////////////////////////////////
	//Visualize the downflow bubbles
    //Mat regionLabelsClone = regionLabels.clone();
	//for (int row = 0; row < regionLabels.rows; row++) {
	//	for (int col = 0; col < regionLabels.cols; col++) {
	//		if (downFlowBubbles.find(regionLabelsClone.at<MAT_TYPE_INT>(row, col)) == downFlowBubbles.end()) {
	//			regionLabelsClone.at<MAT_TYPE_INT>(row, col) = 0;
	//		} else {
	//			regionLabelsClone.at<MAT_TYPE_INT>(row, col) = 1;
	//		}
	//	}
	//}
	//#ifdef DEBUG
	//	imwrite(string("df_bubbles") + to_string(timeMoment) + "_" + to_string(layer) + ".png", convertTo8Bit(regionLabelsClone));
	//#endif
	//////////////////////////////////////////

	this->regionsOnBoundaries = periodic ? regionsOnBoundaries : set<int>();
}

GranDist::~GranDist() {
}

/**
 * Labels the connected regions.
 */
void GranDist::labelRegions() {
	int rowOffset = (granules.rows - 2 * originalHeight) / 2;
	int colOffset = (granules.cols - 2 * originalWidth) / 2;

	FloodFill floodFill(granules);
	for (int row = 0; row < granules.rows; row++) {
		for (int col = 0; col < granules.cols; col++) {
			floodFill.fill(row, col);
		}
	}
	//////////////////////////////////////////
	//Visualizing if the labeling makes sense
    //Mat img;
    //Mat granuleLabels2;
    //granuleLabels.convertTo(granuleLabels2, CV_8UC3);
    //applyColorMap(granuleLabels2, img, COLORMAP_HSV);
	//imwrite("granule_labels.png", img);
	//////////////////////////////////////////
	regionLabels = floodFill.getLabels();
	numRegions = floodFill.getNumRegions();
	regionAreas = floodFill.getRegionAreas();

	regionAreas.erase(regionLabels.at<MAT_TYPE_INT>(0, 0)); // Removing the region of extra padding
	if (periodic) {
		auto& regionExtents = floodFill.getRegionExtents();
		for (auto& regionAndArea : regionAreas) {
			auto& extents = regionExtents.at(regionAndArea.first);
			if (get<0>(extents) <= rowOffset && get<1>(extents) >= rowOffset + 2 * originalHeight - 1) {
				//assert(regionAndArea.second % 2 == 0); // Somehow the areas can be odd, but why
				//cout << "Dividing by two: " << regionAndArea.first << " " << regionAndArea.second;
				regionAndArea.second /= 2;
				//cout << " " << regionAndArea.second << endl;
			}
			if (get<2>(extents) <= colOffset && get<3>(extents) >= colOffset + 2 * originalWidth - 1) {
				//assert(regionAndArea.second % 2 == 0); // Somehow the areas can be odd, but why
				//cout << "Dividing by two: " << regionAndArea.first << " " << regionAndArea.second;
				regionAndArea.second /= 2;
				//cout << " " << regionAndArea.second << endl;
			}
		}
	}

	BoundaryDetect boundaryDetect(granules);
	for (int row = 0; row < granules.rows; row++) {
		for (int col = 0; col < granules.cols; col++) {
			auto regionLabel = regionLabels.at<MAT_TYPE_INT>(row, col);
			if (regionPerimeters.find(regionLabel) == regionPerimeters.end()) {
				boundaryDetect.detect(row, col);
				regionPerimeters[regionLabel] = ((float) (boundaryDetect.getInner().size() + boundaryDetect.getOuter().size())) / (2 * M_PI);
			}
		}
	}
	regionPerimeters.erase(regionLabels.at<MAT_TYPE_INT>(0, 0)); // Removing the region of extra padding
	if (saveMaps) {
		imwrite(string("inner_boundaries_") + to_string(timeMoment) + "_" + to_string(layer) + ".png", (boundaryDetect.getInnerBoundaries()) * 255);
		imwrite(string("outer_boundaries_") + to_string(timeMoment) + "_" + to_string(layer) + ".png", (boundaryDetect.getOuterBoundaries()) * 255);
	}

}

/**
 * Rotates the matrix
 * @param[in] src matrix to rotate
 * @param[in] angle angle of rotation in degrees.
 * @return the rotated matrix
 *
 */
Mat rotate(const Mat& src, double angle) {
	Mat dst;
	// Create a destination to paint the source into.
	dst.create(src.size(), src.type());
	#ifdef GPU
		/////////////////////////////////////////////
		// GPU routine may not work as it is untested
		/////////////////////////////////////////////
		gpu::GpuMat src_gpu;
		gpu::GpuMat dst_gpu;
		// Push the images into the GPU
		src_gpu.upload(src);
		dst_gpu.upload(dst);
		// Rotate in the GPU!
		gpu::rotate(src_gpu, dst_gpu, src_gpu.size(), angle, src_gpu.size().width, src_gpu.size().height);

		// Download the rendered GPU data back to CPU
		dst_gpu.download(dst);
	#else
		Point center = Point(src.cols/2, src.rows/2);
		Mat rot_mat(2, 3, CV_32FC1);
		// Get the rotation matrix with the specifications above
		rot_mat = getRotationMatrix2D(center, angle, 1.0);
		// Rotate the warped image
		warpAffine(src, dst, rot_mat, src.size(), INTER_NEAREST);

	#endif
	return dst;
}

/**
 * @return true if the given point is in the domain of simulation grid,
 * false otherwise (there are extra pixels with 0 values added to fit the rotated image fully to matrix)
 */
bool inDomain(const Mat& granules, int row, int col) {
	return granules.at<MAT_TYPE_FLOAT>(row, col) == UP_FLOW || granules.at<MAT_TYPE_FLOAT>(row, col) == DOWN_FLOW;
}

/**
 * @return true if the given point is in the down flow bubble, false in case it is in down flow lane.
 */
bool GranDist::inDownFlowPatch(const Mat& regionLabels, int row, int col) const {
	return downFlowPatches.find((int) regionLabels.at<MAT_TYPE_FLOAT>(row, col)) != downFlowPatches.end();
}

/**
 * @return true if the given granule intersects with boundary of the simulation domain
 */
bool GranDist::onBoundary(const Mat& regionLabels, int row, int col) const {
	return regionsOnBoundaries.find((int) regionLabels.at<MAT_TYPE_FLOAT>(row, col)) != regionsOnBoundaries.end();
}

//bool endsAtDifferentGranules(const Mat& regionLabels, int startRow, int endRow, int col, int domainStart, int domainEnd) {
//	return startRow > domainStart && endRow < domainEnd && regionLabels.at<MAT_TYPE_FLOAT>(startRow - 1, col) != regionLabels.at<MAT_TYPE_FLOAT>(endRow, col);
//}

unique_ptr<float> GranDist::getIgLaneIndex(const Mat& regionLabels, int startRow, int endRow, int col, int domainStart, int domainEnd) const {
	if (startRow <= domainStart || endRow >= domainEnd) {
		return unique_ptr<float>(nullptr);
	}
	auto startLabel = regionLabels.at<MAT_TYPE_FLOAT>(startRow - 1, col);
	auto endLabel = regionLabels.at<MAT_TYPE_FLOAT>(endRow, col);
	if (startLabel == endLabel) {
		return unique_ptr<float>(nullptr);
	}
	float index = startLabel > endLabel ? startLabel * numRegions + endLabel : endLabel * numRegions + startLabel;
	return make_unique<float>(index);
}


/**
 * Calculates inter- and intragranular distances on vertical lines
 * @param[in] granules rotated matrix of down/up flows
 * @param[in] granuleLabels rotated matrix of granule labels
 * @return matrices of granule sizes, intergranular lane widths, down flow patch sizes and down intergranular lane indices
 */
tuple<Mat, Mat, Mat, Mat> GranDist::calcDistances(const Mat& granules, const Mat& regionLabels) const {
	Mat granuleSizes = Mat::zeros(granules.rows, granules.cols, CV_32F);
	Mat igLaneWidths = Mat::ones(granules.rows, granules.cols, CV_32F) * INFTY;
	Mat dfPatchSizes = Mat::zeros(granules.rows, granules.cols, CV_32F);
	Mat igLaneIndices = Mat::zeros(granules.rows, granules.cols, CV_32F);
	for (int col = 0; col < granules.cols; col++) {
		bool inGranule = granules.at<MAT_TYPE_FLOAT>(0, col) == UP_FLOW;
		float dist = 0;
		int domainStart = granules.rows;
		int domainEnd = granules.rows;
		int startRow = 0;
		int row;
		for (row = 1; row < granules.rows; row++) {
			if (!inDomain(granules, row, col)) { // This point not in domain
				if (inDomain(granules, row - 1, col)) { // Previous point in domain
					// Going out of domain from bottom
					domainEnd = row;
					break;
				}
				// Still not entered the domain
				continue;
			}
			if (!inDomain(granules, row - 1, col)) { // Previous point not in domain
				// Entering domain from top
				domainStart = row;
				startRow = row;
				inGranule = granules.at<MAT_TYPE_FLOAT>(row, col) == UP_FLOW;
				continue;
			}
			if ((granules.at<MAT_TYPE_FLOAT>(row, col) == UP_FLOW) == inGranule) {
				dist++;
			} else {
				bool inIgLane = inGranule ? false : inDomain(granules, startRow - 1, col) && !inDownFlowPatch(regionLabels, startRow, col);
				// In case of periodic boundary skip regions intersecting with the boundary, except for down flow lanes
				if (!periodic || inIgLane || !onBoundary(regionLabels, startRow, col)) {
					auto laneIndex = getIgLaneIndex(regionLabels, startRow, row, col, domainStart, domainEnd);
					// In case of down flow lanes don't count these regions that are on the boundaries
					if (!inIgLane || laneIndex) {
						Mat& dists = inGranule ? granuleSizes : (inIgLane ? igLaneWidths : dfPatchSizes);
						for (int row1 = startRow; row1 < row; row1++) {
							assert(dist == row - startRow - 1);
							dists.at<MAT_TYPE_FLOAT>(row1, col) = row - startRow;
							if (inIgLane) {
								igLaneIndices.at<MAT_TYPE_FLOAT>(row1, col) = *laneIndex;
							}
						}
					}
				}
				dist = 0;
				startRow = row;
				inGranule = !inGranule;
			}
		}
		if (!periodic) {
			bool inIgLane = inGranule ? false : inDomain(granules, startRow - 1, col) && !inDownFlowPatch(regionLabels, startRow, col);
			Mat& dists = inGranule ? granuleSizes : (inIgLane ? igLaneWidths : dfPatchSizes);
			auto laneIndex = getIgLaneIndex(regionLabels, startRow, row, col, domainStart, domainEnd);
			if (!inIgLane || laneIndex) {
				for (int row1 = startRow; row1 < row; row1++) {
					assert(dist == row - startRow - 1);
					dists.at<MAT_TYPE_FLOAT>(row1, col) = row - startRow;
					if (inIgLane) {
						igLaneIndices.at<MAT_TYPE_FLOAT>(row1, col) = *laneIndex;
					}
				}
			}
		}
	}
	return make_tuple(granuleSizes, igLaneWidths, dfPatchSizes, igLaneIndices);
}

set<int> GranDist::getClosedRegions() const {
	// Need to give regionLabelsFloat instead of regionLabels to FloodFill class
	// as it currently only supports float matrices
	FloodFill floodFill(regionLabelsFloat);
	for (int row = 0; row < regionLabelsFloat.rows; row++) {
		for (int col = 0; col < regionLabelsFloat.cols; col++) {
			floodFill.fill(row, col);
		}
	}
	//////////////////////////////////////////
	//Visualizing if the labeling makes sense
    //Mat img;
    //Mat granuleLabels2;
    //granuleLabels.convertTo(granuleLabels2, CV_8UC3);
    //applyColorMap(granuleLabels2, img, COLORMAP_HSV);
	//imwrite("granule_labels.png", img);
	//////////////////////////////////////////
	return floodFill.getClosedRegions();
}


set<int /*regionLabel*/> GranDist::getRegionsOnBoundaries() const {
	set<int> regionsOnBoundaries;

	int rowOffset = (regionLabels.rows - 2 * originalHeight) / 2;
	int colOffset = (regionLabels.cols - 2 * originalWidth) / 2;

	int midRow = regionLabels.rows / 2;
	int midCol = regionLabels.cols / 2;

	for (int row = 0; row < regionLabels.rows; row++) {
		if (regionLabels.at<MAT_TYPE_INT>(row, midCol) == regionLabels.at<MAT_TYPE_INT>(row, midCol - 1)) {
			regionsOnBoundaries.insert(regionLabels.at<MAT_TYPE_INT>(row, colOffset));
			regionsOnBoundaries.insert(regionLabels.at<MAT_TYPE_INT>(row, midCol + originalWidth - 1));
			//cout << granuleLabels.at<MAT_TYPE_INT>(row, 0) << endl;
		}
	}
	for (int col = 0; col < regionLabels.cols; col++) {
		if (regionLabels.at<MAT_TYPE_INT>(midRow, col) == regionLabels.at<MAT_TYPE_INT>(midRow - 1, col)) {
			regionsOnBoundaries.insert(regionLabels.at<MAT_TYPE_INT>(rowOffset, col));
			regionsOnBoundaries.insert(regionLabels.at<MAT_TYPE_INT>(midRow + originalHeight - 1, col));
		}
	}
	return regionsOnBoundaries;
}


/**
 * Labels the local extrema of the distance matrix.
 */
Mat labelExtrema(const Mat& dists, bool minimaOrMaxima, const Mat* mask) {
	FloodFill floodFill(dists, equal_to<float>(), mask);
	for (;;) {
		float globalExtremum = minimaOrMaxima ? INFTY : 0;
		int extremumRow = -1;
		int extremumCol = -1;
		for (int row = 0; row < dists.rows; row++) {
			for (int col = 0; col < dists.cols; col++) {
				if (floodFill.getLabels().at<MAT_TYPE_INT>(row, col) == 0) {
					auto dist = dists.at<MAT_TYPE_FLOAT>(row, col);
					if (minimaOrMaxima) {
						if (dist < globalExtremum) {
							globalExtremum = dist;
							extremumRow = row;
							extremumCol = col;
						}
					} else {
						if (dist > globalExtremum) {
							globalExtremum = dist;
							extremumRow = row;
							extremumCol = col;
						}
					}
				}
			}
		}
		if (extremumRow < 0 || extremumCol < 0) {
			break;
		}
		// Label all points that are in the neighborhood of this extremum
		if (minimaOrMaxima) {
			floodFill.setCompFunc(greater_equal<float>());
			floodFill.fill(extremumRow, extremumCol);
		} else {
			floodFill.setCompFunc(less_equal<float>());
			floodFill.fill(extremumRow, extremumCol);
		}
	}
	//////////////////////////////////////////
	// Visualizing if the labeling makes sense
    //Mat img;
    //Mat extremaLabels2;
    //extremaLabels.convertTo(extremaLabels2, CV_8UC3);
    //applyColorMap(extremaLabels2, img, COLORMAP_HSV);
	//imwrite("extrema_labels.png", img);
	//////////////////////////////////////////
	return floodFill.getLabels();
}

/**
 * Calculates global extrema of distances
 * @param[in] dists distance matrix
 * @param[in] mat matrix of down/up flows
 * @param[in] regionLabels matrix of closed region labels (either granules or regions of local extrema)
 * @param[in] compFunc comparison function, either less<float>() or greater<float()>
 * @return extrema of distances
 */
vector<tuple<float /*value*/, int /*row*/, int /*col*/>> findExtrema(const Mat& dists, const Mat& regionLabels, function<bool(float, float)> compFunc) {
	map<int, tuple<float, int, int>> extremaPerRegion;
	for (int row = 0; row < dists.rows; row++) {
		for (int col = 0; col < dists.cols; col++) {
			int label = regionLabels.at<MAT_TYPE_INT>(row, col);
			auto dist = dists.at<MAT_TYPE_FLOAT>(row, col);
			if (extremaPerRegion.find(label) == extremaPerRegion.end()) {
				extremaPerRegion[label] = make_tuple(dist, row, col);
			} else {
				if (compFunc(dist, get<0>(extremaPerRegion[label]))) {
					extremaPerRegion[label] = make_tuple(dist, row, col);
				}
			}
		}
	}
	vector<tuple<float, int, int>> extrema;
	for (auto extremum : extremaPerRegion) {
		extrema.push_back(extremum.second);
	}
	return extrema;
}

Mat convertToColorAndMarkExtrema(const Mat& src, const vector<tuple<float /*value*/, int /*row*/, int /*col*/>>& extrema, float excludeValue) {
	Mat dst;
	cvtColor(src, dst, CV_GRAY2RGB);
	for (auto extremum : extrema) {
		int row = get<1>(extremum);
		int col = get<2>(extremum);
		if (get<0>(extremum) != excludeValue) {
			dst.at<Vec3b>(row, col) = RED;
		}
	}
	return dst;
}

void GranDist::filterExtrema(vector<tuple<float /*value*/, int /*row*/, int /*col*/>>& extrema, bool byRegionLabel) const {
	for (auto i = extrema.begin(); i != extrema.end();) {
		bool found = false;
		if (byRegionLabel) {
			// The region of the extremum falls into original map
			int label = regionLabels.at<MAT_TYPE_INT>(get<1>(*i), get<2>(*i));
			for (int row = cropRect.y; row < granules.rows; row++) {
				for (int col = cropRect.x; col < granules.cols; col++) {
					if (regionLabels.at<MAT_TYPE_INT>(row, col) == label) {
						found = true;
						break;
					}
				}
				if (found) {
					break;
				}
			}
		} else {
			// The extremum is inside the original map
			found = get<1>(*i) >= cropRect.y && get<2>(*i) >= cropRect.x;
		}
		if (!found) {
			i = extrema.erase(i);
		} else {
			i++;
		}

	}
}

void GranDist::process() {
	if (periodic) {
		tileMatrix(granules, originalHeight, originalWidth);
	}
	//Mat croppedGranules = granules(cropRect);
	Mat granuleSizes;
	Mat igLaneMinWidths;
	Mat dfPatchSizes;
	Mat igLaneIndices;
	for (double angle = 0; angle < 180; angle += DELTA_ANGLE) {
		Mat granulesRotated = angle > 0 ? rotate(granules, angle) : granules;
		Mat regionLabelsRotated = angle > 0 ? rotate(regionLabelsFloat, angle) : regionLabelsFloat;
		if (saveMaps && ((int) angle) == 0) {
			imwrite(string("map_") + to_string(timeMoment) + "_" + to_string(layer) + "_" + to_string((int) angle) + ".png", (granulesRotated - 1) * 255);
		}
		auto dists = calcDistances(granulesRotated, regionLabelsRotated);
		auto verticalGranuleSizes = get<0>(dists);
		auto verticalIgLaneWidths = get<1>(dists);
		auto verticalDfPatchSizes = get<2>(dists);
		auto verticalIgLaneIndices = get<3>(dists);
		//#ifdef DEBUG
		//	if (((int) angle) == 0) {
		//		double min1, max1;
		//		minMaxLoc(verticalGranuleSizes, &min1, &max1);
		//		imwrite(string("dists") + to_string(timeMoment) + "_" + to_string(layer) + "_" + to_string((int) angle) + ".png", (verticalGranuleSizes - min1) * 255 / (max1 - min1));
		//	}
		//#endif
		if (angle > 0) {
			verticalGranuleSizes = rotate(verticalGranuleSizes, -angle);
			verticalIgLaneWidths = rotate(verticalIgLaneWidths, -angle);
			verticalDfPatchSizes = rotate(verticalDfPatchSizes, -angle);
			verticalIgLaneIndices = rotate(verticalIgLaneIndices, -angle);
		}
		Mat newGranuleSizes = verticalGranuleSizes;//(cropRect);
		Mat newIgLaneWidths = verticalIgLaneWidths;//(cropRect);
		Mat newDfPatchSizes = verticalDfPatchSizes;//(cropRect);
		Mat newIgLaneIndices = verticalIgLaneIndices;//(cropRect);
		if (angle == 0) {
			// First time
			granuleSizes = newGranuleSizes;
			igLaneMinWidths = newIgLaneWidths;
			dfPatchSizes = newDfPatchSizes;
			igLaneIndices = newIgLaneIndices;
		} else {
			for (int row = 0; row < newGranuleSizes.rows; row++) {
				for (int col = 0; col < newGranuleSizes.cols; col++) {
					if (granules.at<MAT_TYPE_FLOAT>(row, col) == UP_FLOW) {
						auto newSize = newGranuleSizes.at<MAT_TYPE_FLOAT>(row, col);
						if (newSize > granuleSizes.at<MAT_TYPE_FLOAT>(row, col)) {
							// in granule and new size is greater
							granuleSizes.at<MAT_TYPE_FLOAT>(row, col) = newSize;
						}
						igLaneMinWidths.at<MAT_TYPE_FLOAT>(row, col) = INFTY;
						dfPatchSizes.at<MAT_TYPE_FLOAT>(row, col) = 0;
					} else {
						if (inDownFlowPatch(regionLabelsFloat, row, col)) {
							auto newSize = newDfPatchSizes.at<MAT_TYPE_FLOAT>(row, col);
							if (newSize > dfPatchSizes.at<MAT_TYPE_FLOAT>(row, col)) {
								// in down flow bubble and new size is greater
								dfPatchSizes.at<MAT_TYPE_FLOAT>(row, col) = newSize;
							}
							igLaneMinWidths.at<MAT_TYPE_FLOAT>(row, col) = INFTY;
						} else {
							auto newWidth = newIgLaneWidths.at<MAT_TYPE_FLOAT>(row, col);
							if (newWidth < igLaneMinWidths.at<MAT_TYPE_FLOAT>(row, col)) {
								// in down flow lane and new width is shorter
								igLaneMinWidths.at<MAT_TYPE_FLOAT>(row, col) = newWidth;
								igLaneIndices.at<MAT_TYPE_FLOAT>(row, col) = newIgLaneIndices.at<MAT_TYPE_FLOAT>(row, col);
							}
							dfPatchSizes.at<MAT_TYPE_FLOAT>(row, col) = 0;
						}
						granuleSizes.at<MAT_TYPE_FLOAT>(row, col) = 0;
					}
				}
			}
		}
	}

	Mat igLaneMaxWidths = igLaneMinWidths.clone();

	//-------------------------------------------------------------------------
	// Find and output granule size maxima
	//-------------------------------------------------------------------------
	Mat granuleSizesClone = granuleSizes.clone();
	//std::ofstream output1(string("granule_size_maxima_") + to_string(layer) + ".txt", ios_base::app);
	auto granuleSizeMaxima = findExtrema(granuleSizes, regionLabels, greater<float>());
	filterExtrema(granuleSizeMaxima);
	for (auto extremum : granuleSizeMaxima) {
		if (get<0>(extremum) != 0) {
			int row = get<1>(extremum);
			int col = get<2>(extremum);
			int label = regionLabels.at<MAT_TYPE_INT>(row, col);
			//assert(granules.at<MAT_TYPE_FLOAT>(row, col) == UP_FLOW);
			output << label << " GRAN " << regionAreas[label] << " " << regionPerimeters[label] << " " << get<0>(extremum) << " " << row << " " << col << " " << endl;
		}
	}

	// Convert to 8-bit matrices and normalize from 0 to 255
	convertTo8Bit(granuleSizes);
	convertTo8Bit(granuleSizesClone);

	// Convert to color image and mark positions of extrema red
	Mat granuleSizeMaximaRGB = convertToColorAndMarkExtrema(granuleSizesClone, granuleSizeMaxima, 0);

	// Visualize matrices
	//imwrite(string("granule_sizes") + to_string(layer) + ".png", granuleSizes);
	if (saveMaps) {
		imwrite(string("granule_size_maxima") + to_string(timeMoment) + "_" + to_string(layer) + ".png", granuleSizeMaximaRGB);
	}

	//-------------------------------------------------------------------------
	// Find and output intergranular lane width minima
	//-------------------------------------------------------------------------

	Mat igLaneMinWidthsClone = igLaneMinWidths.clone();
	Mat minimaLabels = labelExtrema(igLaneMinWidths, true, &igLaneIndices);
	//std::ofstream output2(string("df_width_minima_") + to_string(layer) + ".txt", ios_base::app);
	auto igLaneWidthMinima = findExtrema(igLaneMinWidths, minimaLabels, less<float>());
	filterExtrema(igLaneWidthMinima, false);

	map<float /*down flow lane index*/, tuple<float, int, int> /*minimum*/> uniqueMinima;
	for (auto extremum : igLaneWidthMinima) {
		auto row = get<1>(extremum);
		auto col = get<2>(extremum);
		auto index = igLaneIndices.at<MAT_TYPE_FLOAT>(row, col);
		if (index == 0) {
			// These are the lane indices that were turned to zero due to rounding errors in matrix rotation.
			// Actually they should also be included in results, but probably there are not many of these
			//assert(false);
		} else {
			auto i = uniqueMinima.find(index);
			if (i == uniqueMinima.end()) {
				uniqueMinima[index] = extremum;
			} else if (get<0>(extremum) < get<0>(i->second)) {
				uniqueMinima[i->first] = extremum;
			}
		}
	}

	igLaneWidthMinima.clear();
	for (auto i : uniqueMinima) {
		auto extremum = i.second;
		igLaneWidthMinima.push_back(extremum);
	}


	// Replace occurrences of INFTY with zeros
	for (int row = 0; row < igLaneMinWidths.rows; row++) {
		for (int col = 0; col < igLaneMinWidths.cols; col++) {
			if (igLaneMinWidths.at<MAT_TYPE_FLOAT>(row, col) == INFTY) {
				igLaneMinWidths.at<MAT_TYPE_FLOAT>(row, col) = 0;
			}
			if (igLaneMinWidthsClone.at<MAT_TYPE_FLOAT>(row, col) == INFTY) {
				igLaneMinWidthsClone.at<MAT_TYPE_FLOAT>(row, col) = 0;
			}
		}
	}

	convertTo8Bit(igLaneMinWidths);
	convertTo8Bit(igLaneMinWidthsClone);
	Mat igLaneWidthMinimaRGB = convertToColorAndMarkExtrema(igLaneMinWidthsClone, igLaneWidthMinima, INFTY);


	//imwrite(string("df_widths") + to_string(layer) + ".png", downFlowLaneWidths);
	if (saveMaps) {
		imwrite(string("ig_lane_width_minima") + to_string(timeMoment) + "_" + to_string(layer) + ".png", igLaneWidthMinimaRGB);
	}

	//-------------------------------------------------------------------------
	// Find and output intergranular lane width maxima
	//-------------------------------------------------------------------------

	// Replace occurrences of INFTY with zeros
	for (int row = 0; row < igLaneMaxWidths.rows; row++) {
		for (int col = 0; col < igLaneMaxWidths.cols; col++) {
			if (igLaneMaxWidths.at<MAT_TYPE_FLOAT>(row, col) == INFTY) {
				igLaneMaxWidths.at<MAT_TYPE_FLOAT>(row, col) = 0;
			}
		}
	}


	Mat igLaneMaxWidthsClone = igLaneMaxWidths.clone();
	Mat maximaLabels = labelExtrema(igLaneMaxWidths, false, &igLaneIndices);
	//std::ofstream output2(string("df_width_minima_") + to_string(layer) + ".txt", ios_base::app);
	auto igLaneWidthMaxima = findExtrema(igLaneMaxWidths, maximaLabels, greater<float>());
	filterExtrema(igLaneWidthMaxima, false);

	map<float /*down flow lane index*/, tuple<float, int, int> /*maximum*/> uniqueMaxima;
	for (auto extremum : igLaneWidthMaxima) {
		auto row = get<1>(extremum);
		auto col = get<2>(extremum);
		auto index = igLaneIndices.at<MAT_TYPE_FLOAT>(row, col);
		if (index == 0) {
			// These are the lane indices that were turned to zero due to rounding errors in matrix rotation.
			// Actually they should also be included in results, but probably there are not many of these
			//assert(false);
		} else {
			auto i = uniqueMaxima.find(index);
			if (i == uniqueMaxima.end()) {
				uniqueMaxima[index] = extremum;
			} else if (get<0>(extremum) > get<0>(i->second)) {
				uniqueMaxima[i->first] = extremum;
			}
		}
	}

	igLaneWidthMaxima.clear();

	for (auto i : uniqueMaxima) {
		float igIndex = i.first;
		auto maximum = i.second;
		float maximumWidth = get<0>(maximum);
		int maximumRow = get<1>(maximum);
		int maximumCol = get<2>(maximum);
		int label = regionLabels.at<MAT_TYPE_INT>(maximumRow, maximumCol);

		float minimumWidth = 0;
		int minimumRow = -1;
		int minimumCol = -1;
		if (uniqueMinima.find(igIndex) != uniqueMinima.end()) {
			auto minimum = uniqueMinima[igIndex];
			minimumWidth = get<0>(minimum);
			minimumRow = get<1>(minimum);
			minimumCol = get<2>(minimum);
		}
		output << (numRegions + (int) i.first) << " IGL " << regionAreas[label] << " " << regionPerimeters[label] << " " << minimumWidth << " " << maximumWidth << " " << minimumRow << " " << minimumCol << " " << maximumRow << " " << maximumCol << " " << label << endl;
		igLaneWidthMaxima.push_back(maximum);
	}
	// Just in case output those minima which don't have corresponding maxima (should be zero though)
	for (auto i : uniqueMinima) {
		float igIndex = i.first;
		auto minimum = i.second;
		float minimumWidth = get<0>(minimum);
		int minimumRow = get<1>(minimum);
		int minimumCol = get<2>(minimum);
		int label = regionLabels.at<MAT_TYPE_INT>(minimumRow, minimumCol);

		if (uniqueMaxima.find(igIndex) == uniqueMaxima.end()) {
			output << (numRegions + (int) i.first) << " IGL " << regionAreas[label] << " " << regionPerimeters[label] << " " << minimumWidth << " " << 0 << " " << minimumRow << " " << minimumCol << " " << -1 << " " << -1 << " " << label << endl;
		}
	}


	convertTo8Bit(igLaneMaxWidths);
	convertTo8Bit(igLaneMaxWidthsClone);
	Mat igLaneWidthMaximaRGB = convertToColorAndMarkExtrema(igLaneMaxWidthsClone, igLaneWidthMaxima, 0);


	if (saveMaps) {
		imwrite(string("ig_lane_width_maxima") + to_string(timeMoment) + "_" + to_string(layer) + ".png", igLaneWidthMaximaRGB);
	}

	//-------------------------------------------------------------------------
	// Find and output down flow patch size maxima
	//-------------------------------------------------------------------------
	Mat dfPatchSizesClone = dfPatchSizes.clone();
	//std::ofstream output3(string("df_bubble_size_maxima_") + to_string(layer) + ".txt", ios_base::app);
	auto dfPatchSizeMaxima = findExtrema(dfPatchSizes, regionLabels, greater<float>());
	filterExtrema(dfPatchSizeMaxima);

	for (auto extremum : dfPatchSizeMaxima) {
		if (get<0>(extremum) != 0) {
			int row = get<1>(extremum);
			int col = get<2>(extremum);
			int label = regionLabels.at<MAT_TYPE_INT>(row, col);
			output << label << " DFP " << regionAreas[label] << " " << regionPerimeters[label] << " " << get<0>(extremum) << " " << row << " " << col << " " << endl;
		}
	}

	// Convert to 8-bit matrices and normalize from 0 to 255
	convertTo8Bit(dfPatchSizes);
	convertTo8Bit(dfPatchSizesClone);

	// Convert to color image and mark positions of extrema red
	Mat dfPatchSizeMaximaRGB = convertToColorAndMarkExtrema(dfPatchSizesClone, dfPatchSizeMaxima, 0);

	// Visualize matrices
	//imwrite(string("df_bubble_sizes") + to_string(layer) + ".png", downFlowBubbleSizes);
	if (saveMaps) {
		imwrite(string("df_patch_size_maxima") + to_string(timeMoment) + "_" + to_string(layer) + ".png", dfPatchSizeMaximaRGB);
	}

}
