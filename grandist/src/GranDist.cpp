/*
 * GranDist.cpp
 *
 *  Created on: 21 Feb 2017
 *      Author: olspern1
 */

#include "GranDist.h"
#include "common.h"
#include "FloodFill.h"
#include <vector>
#include <map>
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

GranDist::GranDist(int layer, Mat granules, int originalHeight, int originalWidth, bool periodic, Rect cropRect) :
		layer(layer),
		granules(periodic ? tileMatrix(granules, originalHeight, originalWidth) : granules),
		originalHeight(originalHeight),
		originalWidth(originalWidth),
		periodic(periodic),
		cropRect(cropRect),
		regionLabels() {

	auto labels = labelRegions();
	this->regionLabels = labels.first;
	auto closedRegions = labels.second;


	auto regionsOnBoundaries = getGranulesOnBoundaries();
	for (int row = 0; row < regionLabels.rows; row++) {
		for (int col = 0; col < regionLabels.cols; col++) {
			if (granules.at<MAT_TYPE_FLOAT>(row, col) == DOWN_FLOW)  {
				int label = regionLabels.at<MAT_TYPE_INT>(row, col);
				if (closedRegions.find(label) != closedRegions.end() && regionsOnBoundaries.find(label) == regionsOnBoundaries.end()) {
					downFlowBubbles.insert(label);
				}
			}
		}
	}
	this->regionsOnBoundaries = periodic ? regionsOnBoundaries : set<int>();
}

GranDist::~GranDist() {
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
bool GranDist::inDownFlowBubble(const Mat& regionLabels, int row, int col) const {
	return downFlowBubbles.find(regionLabels.at<MAT_TYPE_FLOAT>(row, col)) != downFlowBubbles.end();
}

/**
 * @return true if the given granule intersects with boundary of the simulation domain
 */
bool GranDist::onBoundary(const Mat& regionLabels, int row, int col) const {
	return regionsOnBoundaries.find((int) regionLabels.at<MAT_TYPE_FLOAT>(row, col)) != regionsOnBoundaries.end();
}
/**
 * Calculates inter- and intragranular distances on vertical lines
 * @param[in] granules rotated matrix of down/up flows
 * @param[in] granuleLabels rotated matrix of granule labels
 * @return matrices of granule sizes, down flow widths and down flow bubble sizes
 */
tuple<Mat, Mat, Mat> GranDist::calcDistances(const Mat& granules, const Mat& regionLabels) const {
	Mat granuleSizes = Mat::zeros(granules.rows, granules.cols, CV_32F);
	Mat downFlowLaneWidths = Mat::ones(granules.rows, granules.cols, CV_32F) * INFTY;
	Mat downFlowBubbleSizes = Mat::zeros(granules.rows, granules.cols, CV_32F);
	for (int col = 0; col < granules.cols; col++) {
		bool inGranule = granules.at<MAT_TYPE_FLOAT>(0, col) == UP_FLOW;
		float dist = 1;
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
				bool inDownFlowLane = inGranule ? false : inDomain(granules, startRow - 1, col) && !inDownFlowBubble(regionLabels, startRow, col);
				// In case of periodic boundary skip regions intersecting with the boundary, except for down flow lanes
				if (!periodic || inDownFlowLane || !onBoundary(regionLabels, startRow, col)) {
					// In case of downflow lanes don't count these regions that are on the boundaries
					//if (!inDownFlowLane || startRow > domainStart) {
						Mat& dists = inGranule ? granuleSizes : (inDownFlowLane ? downFlowLaneWidths : downFlowBubbleSizes);
						for (int row1 = startRow; row1 < row; row1++) {
							dists.at<MAT_TYPE_FLOAT>(row1, col) = dist;
						}
					//}
				}
				dist = 1;
				startRow = row;
				inGranule = !inGranule;
			}
		}
		if (!periodic) {
			bool inDownFlowLane = inGranule ? false : inDomain(granules, startRow - 1, col) && !inDownFlowBubble(regionLabels, startRow, col);
			Mat& dists = inGranule ? granuleSizes : (inDownFlowLane ? downFlowLaneWidths : downFlowBubbleSizes);
			if (!inDownFlowLane) {
				for (int row1 = startRow; row1 < row; row1++) {
					dists.at<MAT_TYPE_FLOAT>(row1, col) = dist;
				}
			}
		}
	}
	return make_tuple(granuleSizes, downFlowLaneWidths, downFlowBubbleSizes);
}

/**
 * Labels the closed regions. Only needed to identify the closed regions if we want to extract only
 * single extremum per region. Otherwise local extrema could be calculated from distance matrix.
 */
pair<Mat, set<int>> GranDist::labelRegions() const {
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
	return make_pair(floodFill.getLabels(), floodFill.getClosedRegions());
}

set<int /*granuleLabel*/> GranDist::getGranulesOnBoundaries() const {
	set<int> granulesOnBoundaries;

	int rowOffset = (regionLabels.rows - 2 * originalHeight) / 2;
	int colOffset = (regionLabels.cols - 2 * originalWidth) / 2;

	int midRow = regionLabels.rows / 2;
	int midCol = regionLabels.cols / 2;

	for (int row = 0; row < regionLabels.rows; row++) {
		if (regionLabels.at<MAT_TYPE_INT>(row, midCol) == regionLabels.at<MAT_TYPE_INT>(row, midCol - 1)) {
			granulesOnBoundaries.insert(regionLabels.at<MAT_TYPE_INT>(row, colOffset));
			granulesOnBoundaries.insert(regionLabels.at<MAT_TYPE_INT>(row, midCol + originalWidth - 1));
			//cout << granuleLabels.at<MAT_TYPE_INT>(row, 0) << endl;
		}
	}
	for (int col = 0; col < regionLabels.cols; col++) {
		if (regionLabels.at<MAT_TYPE_INT>(midRow, col) == regionLabels.at<MAT_TYPE_INT>(midRow - 1, col)) {
			granulesOnBoundaries.insert(regionLabels.at<MAT_TYPE_INT>(rowOffset, col));
			granulesOnBoundaries.insert(regionLabels.at<MAT_TYPE_INT>(midRow + originalHeight - 1, col));
		}
	}
	return granulesOnBoundaries;
}


/**
 * Labels the local extrema of the distance matrix.
 */
Mat labelExtrema(const Mat& dists, bool minimaOrMaxima) {
	FloodFill floodFill(dists);
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

void convertTo8Bit(Mat& mat) {
	double min, max;
	minMaxLoc(mat, &min, &max);
	mat.convertTo(mat, CV_8U, 255.0 / (max - min),-min * 255.0 / (max - min));
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

void GranDist::process() {
	if (periodic) {
		tileMatrix(granules, originalHeight, originalWidth);
	}
	Mat croppedGranules = granules(cropRect);
	Mat granuleSizes;
	Mat downFlowLaneWidths;
	Mat downFlowBubbleSizes;
	Mat regionLabelsFloat;
	regionLabels.convertTo(regionLabelsFloat, CV_32F);
	for (double angle = 0; angle < 180; angle += DELTA_ANGLE) {
		Mat granulesRotated = angle > 0 ? rotate(granules, angle) : granules;
		Mat regionLabelsRotated = angle > 0 ? rotate(regionLabelsFloat, angle) : regionLabelsFloat;
		#ifdef DEBUG
			if (((int) angle) % 10 == 0) {
				imwrite(string("granules") + to_string(layer) + "_" + to_string((int) angle) + ".png", (granulesRotated - 1) * 255);
			}
		#endif
		auto dists = calcDistances(granulesRotated, regionLabelsRotated);
		auto verticalGranuleSizes = get<0>(dists);
		auto verticalDownFlowLaneWidths = get<1>(dists);
		auto verticalDownFlowBubbleSizes = get<2>(dists);
		#ifdef DEBUG
			if (((int) angle) % 10 == 0) {
				double min1, max1;
				minMaxLoc(verticalGranuleSizes, &min1, &max1);
				imwrite(string("dists") + to_string(layer) + "_" + to_string((int) angle) + ".png", (verticalGranuleSizes - min1) * 255 / (max1 - min1));
			}
		#endif
		if (angle > 0) {
			verticalGranuleSizes = rotate(verticalGranuleSizes, -angle);
			verticalDownFlowLaneWidths = rotate(verticalDownFlowLaneWidths, -angle);
			verticalDownFlowBubbleSizes = rotate(verticalDownFlowBubbleSizes, -angle);
		}
		Mat newGranuleSizes = verticalGranuleSizes(cropRect);
		Mat newDownFlowLaneWidths = verticalDownFlowLaneWidths(cropRect);
		Mat newDownFlowBubbleSizes = verticalDownFlowBubbleSizes(cropRect);
		if (angle == 0) {
			// First time
			granuleSizes = newGranuleSizes;
			downFlowLaneWidths = newDownFlowLaneWidths;
			downFlowBubbleSizes = newDownFlowBubbleSizes;
		} else {
			for (int row = 0; row < newGranuleSizes.rows; row++) {
				for (int col = 0; col < newGranuleSizes.cols; col++) {
					if (croppedGranules.at<MAT_TYPE_FLOAT>(row, col) == UP_FLOW) {
						auto newSize = newGranuleSizes.at<MAT_TYPE_FLOAT>(row, col);
						if (newSize > granuleSizes.at<MAT_TYPE_FLOAT>(row, col)) {
							// in granule and new size is greater
							granuleSizes.at<MAT_TYPE_FLOAT>(row, col) = newSize;
						}
						downFlowLaneWidths.at<MAT_TYPE_FLOAT>(row, col) = INFTY;
					} else {
						if (inDownFlowBubble(regionLabels, row, col)) {
							auto newSize = newDownFlowBubbleSizes.at<MAT_TYPE_FLOAT>(row, col);
							if (newSize > downFlowBubbleSizes.at<MAT_TYPE_FLOAT>(row, col)) {
								// in down flow bubble and new size is greater
								downFlowBubbleSizes.at<MAT_TYPE_FLOAT>(row, col) = newSize;
							}
						} else {
							auto newWidth = newDownFlowLaneWidths.at<MAT_TYPE_FLOAT>(row, col);
							if (newWidth < downFlowLaneWidths.at<MAT_TYPE_FLOAT>(row, col)) {
								// in downflow lane and new width is shorter
								downFlowLaneWidths.at<MAT_TYPE_FLOAT>(row, col) = newWidth;
							}

						}
						granuleSizes.at<MAT_TYPE_FLOAT>(row, col) = 0;
					}
				}
			}
		}
	}
	regionLabels = regionLabels(cropRect);

	//-------------------------------------------------------------------------
	// Granule sizes
	//-------------------------------------------------------------------------
	Mat lInnerGlobal = granuleSizes.clone();
	std::ofstream output1(string("inner_global_dists") + to_string(layer) + ".txt");
	auto innerGlobalExtrema = findExtrema(granuleSizes, regionLabels, greater<float>());
	for (auto extremum : innerGlobalExtrema) {
		if (get<0>(extremum) != 0) {
			output1 << get<0>(extremum) << " " << get<1>(extremum) << " " << get<2>(extremum) << endl;
		}
	}
	output1.close();
	//cout << lOuter << endl;

	//Mat lInnerLocal = granuleSizes.clone();
	//Mat maximaLabels = labelExtrema(granuleSizes, false);
	//std::ofstream output3(string("inner_local_dists") + to_string(layer) + ".txt");
	//auto innerLocalExtrema = findExtrema(granuleSizes, maximaLabels, greater<float>());
	//for (auto extremum : innerLocalExtrema) {
	//	output3 << get<0>(extremum) << " " << get<1>(extremum) << " " << get<2>(extremum) << endl;
	//}
	//output3.close();

	// Convert to 8-bit matrices and normalize from 0 to 255
	convertTo8Bit(granuleSizes);
	convertTo8Bit(lInnerGlobal);
	//convertTo8Bit(lInnerLocal);

	// Convert to color image and mark positions of extrema red
	Mat lInnerGlobalRGB = convertToColorAndMarkExtrema(lInnerGlobal, innerGlobalExtrema, 0);
	//Mat lInnerLocalRGB = convertToColorAndMarkExtrema(lInnerLocal, innerLocalExtrema, 0);

	// Visualize matrices
	imwrite(string("inner_dists") + to_string(layer) + ".png", granuleSizes);
	imwrite(string("inner_global_extrema") + to_string(layer) + ".png", lInnerGlobalRGB);
	//imwrite(string("inner_local_extrema") + to_string(layer) + ".png", lInnerLocalRGB);

	//-------------------------------------------------------------------------
	// Down flow lane widths
	//-------------------------------------------------------------------------
	//Mat lOuterGlobal = downFlowLaneWidths.clone();
	//std::ofstream output2(string("outer_global_dists") + to_string(layer) + ".txt");
	//auto outerGlobalExtrema = findExtrema(downFlowLaneWidths, regionLabels, less<float>());
	//for (auto extremum : outerGlobalExtrema) {
	//	if (get<0>(extremum) != INFTY) {
	//		output2 << get<0>(extremum) << " " << get<1>(extremum) << " " << get<2>(extremum) << endl;
	//	}
	//	//if (lOuterGlobal.at<MAT_TYPE_FLOAT>(get<1>(extremum), get<2>(extremum)) != INFTY) {
	//	//	lOuterGlobal.at<MAT_TYPE_FLOAT>(get<1>(extremum), get<2>(extremum)) = 20 * lOuterGlobal.at<MAT_TYPE_FLOAT>(get<1>(extremum), get<2>(extremum));
	//	//}
	//}
	//output2.close();

	Mat downFlowLaneWidthMinima = downFlowLaneWidths.clone();
	Mat minimaLabels = labelExtrema(downFlowLaneWidths, true);
	std::ofstream output4(string("dl_width_minima") + to_string(layer) + ".txt");
	auto outerLocalExtrema = findExtrema(downFlowLaneWidths, minimaLabels, less<float>());
	for (auto extremum : outerLocalExtrema) {
		output4 << get<0>(extremum) << " " << get<1>(extremum) << " " << get<2>(extremum) << endl;
	}
	output4.close();

	// Replace occurrences of INFTY with zeros
	for (int row = 0; row < downFlowLaneWidths.rows; row++) {
		for (int col = 0; col < downFlowLaneWidths.cols; col++) {
			if (downFlowLaneWidths.at<MAT_TYPE_FLOAT>(row, col) == INFTY) {
				downFlowLaneWidths.at<MAT_TYPE_FLOAT>(row, col) = 0;
			}
			//if (lOuterGlobal.at<MAT_TYPE_FLOAT>(row, col) == INFTY) {
			//	lOuterGlobal.at<MAT_TYPE_FLOAT>(row, col) = 0;
			//}
			if (downFlowLaneWidthMinima.at<MAT_TYPE_FLOAT>(row, col) == INFTY) {
				downFlowLaneWidthMinima.at<MAT_TYPE_FLOAT>(row, col) = 0;
			}
		}
	}

	convertTo8Bit(downFlowLaneWidths);
	//convertTo8Bit(lOuterGlobal);
	convertTo8Bit(downFlowLaneWidthMinima);
	//Mat lOuterGlobalRGB = convertToColorAndMarkExtrema(lOuterGlobal, outerGlobalExtrema, INFTY);
	Mat downFlowLaneWidthMinimaRGB = convertToColorAndMarkExtrema(downFlowLaneWidthMinima, outerLocalExtrema, INFTY);


	imwrite(string("dl_widths") + to_string(layer) + ".png", downFlowLaneWidths);
	//imwrite(string("outer_global_extrema") + to_string(layer) + ".png", lOuterGlobalRGB);
	imwrite(string("dl_width_minima") + to_string(layer) + ".png", downFlowLaneWidthMinimaRGB);

}
