/*
 * GranDist.cpp
 *
 *  Created on: 21 Feb 2017
 *      Author: olspern1
 */

#include <grandist/src/GranDist.h>
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

GranDist::GranDist(int layer, Mat granules, int originalHeight, int originalWidth, bool periodic, Rect cropRect) :
		layer(layer),
		granules(granules),
		originalHeight(originalHeight),
		originalWidth(originalWidth),
		periodic(periodic),
		cropRect(cropRect),
		granuleLabels(labelGranules()),
		granulesOnBoundaries(periodic ? getGranulesOnBoundaries() : set<int>()) {

}

GranDist::~GranDist() {
}

/**
 * Rotates the matrix
 * @param[in] src matrix to rotate
 * @param[in] angle angle of rotation in degrees.
 * Determines which type of interpolation is used.
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

bool inDomain(const Mat& mat, int row, int col) {
	return mat.at<MAT_TYPE_FLOAT>(row, col) == IN_GRANULE || mat.at<MAT_TYPE_FLOAT>(row, col) == OUT_GRANULE;
}

/**
 * Calculates inter- and intragranular distances
 * @param[in] mat matrix of down/up flows
 * @param[in] periodic whether the boundaries are periodic.
 * If true omits the measures for regions crossing the boundaries.
 * @return matrices of maximum inner and minimum outer distances
 */
pair<Mat, Mat> GranDist::calcDistances(const Mat& granules, const Mat& granuleLabels) const {
	Mat innerDists = Mat::zeros(granules.rows, granules.cols, CV_32F);
	Mat outerDists = Mat::ones(granules.rows, granules.cols, CV_32F) * INFTY;
	for (int col = 0; col < granules.cols; col++) {
		bool inGranule = granules.at<MAT_TYPE_FLOAT>(0, col) == IN_GRANULE;
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
				inGranule = granules.at<MAT_TYPE_FLOAT>(row, col) == IN_GRANULE;
				continue;
			}
			if ((granules.at<MAT_TYPE_FLOAT>(row, col) == IN_GRANULE) == inGranule) {
				dist++;
			} else {
				if (!periodic || !inGranule || granulesOnBoundaries.find((int) granuleLabels.at<MAT_TYPE_FLOAT>(startRow, col)) == granulesOnBoundaries.end()) {
					if (inGranule || (startRow > domainStart && granuleLabels.at<MAT_TYPE_FLOAT>(startRow - 1, col) != granuleLabels.at<MAT_TYPE_FLOAT>(row, col))) {
						Mat& dists = inGranule ? innerDists : outerDists;
						for (int row1 = startRow; row1 < row; row1++) {
							dists.at<MAT_TYPE_FLOAT>(row1, col) = dist;
						}
					}
				}
				dist = 1;
				startRow = row;
				inGranule = !inGranule;
			}
		}
		Mat& dists = inGranule ? innerDists : outerDists;
		if (!periodic) {
			if (inGranule || (startRow > domainStart && row < domainEnd && granuleLabels.at<MAT_TYPE_FLOAT>(startRow - 1, col) != granuleLabels.at<MAT_TYPE_FLOAT>(row, col))) {
				for (int row1 = startRow; row1 < row; row1++) {
					dists.at<MAT_TYPE_FLOAT>(row1, col) = dist;
				}
			}
		}
	}
	return make_pair(innerDists, outerDists);
}

/**
 * Helper method for labeling the regions
 */
pair<int, int> labelRow(Mat& labels, const Mat& mat, const int row, const int col, const int label, function<bool(float, float)> compFunc) {
	labels.at<MAT_TYPE_INT>(row, col) = label;
	auto initialValue = mat.at<MAT_TYPE_FLOAT>(row, col);
	auto value = initialValue;
	int startCol;
	for (startCol = col - 1; startCol >= 0; startCol--) {
		int neighborValue = mat.at<MAT_TYPE_FLOAT>(row, startCol);
		if (compFunc(neighborValue, value)) {
			labels.at<MAT_TYPE_INT>(row, startCol) = label;
			value = neighborValue;
		} else {
			break;
		}
	}
	value = initialValue;
	int endCol;
	for (endCol = col + 1; endCol < mat.cols; endCol++) {
		int neighborValue = mat.at<MAT_TYPE_FLOAT>(row, endCol);
		if (compFunc(neighborValue, value)) {
			labels.at<MAT_TYPE_INT>(row, endCol) = label;
			value = neighborValue;
		} else {
			break;
		}
	}
	//cout << "markRow " << (startCol + 1) << " " << (endCol - 1) << endl;
	return make_pair(startCol + 1, endCol - 1);
}

/**
 * Helper method for labeling the regions
 */
void labelConnectedRegion(Mat& labels, const Mat& mat, const int row, const int col, const int label, function<bool(float, float)> compFunc) {
	//cout << "markClosedRegion " << label << " " << row << " " << col << endl;
	auto startEndCols = labelRow(labels, mat, row, col, label, compFunc);
	int startCol = startEndCols.first;
	int endCol = startEndCols.second;
	for (int col1 = startCol; col1 <= endCol; col1++) {
		auto value = mat.at<MAT_TYPE_FLOAT>(row, col1);
		if (row > 0 && labels.at<MAT_TYPE_INT>(row - 1, col1) == 0) {
			if (compFunc(mat.at<MAT_TYPE_FLOAT>(row - 1, col1), value)) {
				labelConnectedRegion(labels, mat, row - 1, col1, label, compFunc);
			}
		}
		if (row < mat.rows - 1 && labels.at<MAT_TYPE_INT>(row + 1, col1) == 0) {
			if (compFunc(mat.at<MAT_TYPE_FLOAT>(row + 1, col1), value)) {
				labelConnectedRegion(labels, mat, row + 1, col1, label, compFunc);
			}
		}
	}
	//cout << "markClosedRegion end " << label << " " << row << " " << col << endl;
}

/**
 * Labels the closed regions. Only needed to identify the closed regions if we want to extract only
 * single extremum per region. Otherwise local extrema could be calculated from distance matrix.
 */
Mat GranDist::labelGranules() const {
	Mat granuleLabels = Mat::zeros(granules.rows, granules.cols, CV_32S);
	int label = 1;
	for (int row = 0; row < granules.rows; row++) {
		for (int col = 0; col < granules.cols; col++) {
			if (granuleLabels.at<MAT_TYPE_INT>(row, col) == 0) {
				labelConnectedRegion(granuleLabels, granules, row, col, label++, equal_to<float>());
			}
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
	return granuleLabels;
}

set<int /*granuleLabel*/> GranDist::getGranulesOnBoundaries() const {
	set<int> granulesOnBoundaries;

	int rowOffset = (granuleLabels.rows - 2 * originalHeight) / 2;
	int colOffset = (granuleLabels.cols - 2 * originalWidth) / 2;

	int midRow = granuleLabels.rows / 2;
	int midCol = granuleLabels.cols / 2;

	for (int row = 0; row < granuleLabels.rows; row++) {
		if (granuleLabels.at<MAT_TYPE_INT>(row, midCol) == granuleLabels.at<MAT_TYPE_INT>(row, midCol - 1)) {
			granulesOnBoundaries.insert(granuleLabels.at<MAT_TYPE_INT>(row, colOffset));
			granulesOnBoundaries.insert(granuleLabels.at<MAT_TYPE_INT>(row, midCol + originalWidth - 1));
			//cout << granuleLabels.at<MAT_TYPE_INT>(row, 0) << endl;
		}
	}
	for (int col = 0; col < granuleLabels.cols; col++) {
		if (granuleLabels.at<MAT_TYPE_INT>(midRow, col) == granuleLabels.at<MAT_TYPE_INT>(midRow - 1, col)) {
			granulesOnBoundaries.insert(granuleLabels.at<MAT_TYPE_INT>(rowOffset, col));
			granulesOnBoundaries.insert(granuleLabels.at<MAT_TYPE_INT>(midRow + originalHeight - 1, col));
		}
	}
	return granulesOnBoundaries;
}


/**
 * Labels the local extrema of the distance matrix.
 */
Mat labelExtrema(const Mat& dists, bool minimaOrMaxima) {
	Mat extremaLabels = Mat::zeros(dists.rows, dists.cols, CV_32S);
	int label = 1;
	for (;;) {
		float globalExtremum = minimaOrMaxima ? INFTY : 0;
		int extremumRow = -1;
		int extremumCol = -1;
		for (int row = 0; row < dists.rows; row++) {
			for (int col = 0; col < dists.cols; col++) {
				if (extremaLabels.at<MAT_TYPE_INT>(row, col) == 0) {
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
			labelConnectedRegion(extremaLabels, dists, extremumRow, extremumCol, label++, greater_equal<float>());
		} else {
			labelConnectedRegion(extremaLabels, dists, extremumRow, extremumCol, label++, less_equal<float>());
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
	return extremaLabels;
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

/**
 * Repeats the matrix twice in horizontal and vertical directions.
 * Needed to correctly estimate the granule sizes on the boundaries
 */
void tileMatrix(Mat& mat, int rows, int cols) {
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
	Mat lInner;
	Mat lOuter;
	Mat granuleLabelsFloat;
	granuleLabels.convertTo(granuleLabelsFloat, CV_32F);
	for (double angle = 0; angle < 180; angle += DELTA_ANGLE) {
		Mat granulesRotated = angle > 0 ? rotate(granules, angle) : granules;
		Mat granuleLabelsRotated = angle > 0 ? rotate(granuleLabelsFloat, angle) : granuleLabelsFloat;
		#ifdef DEBUG
			if (((int) angle) % 10 == 0) {
				imwrite(string("granules") + to_string(layer) + "_" + to_string((int) angle) + ".png", (granulesRotated - 1) * 255);
			}
		#endif
		auto dists = calcDistances(granulesRotated, granuleLabelsRotated);
		auto innerDists = dists.first;
		auto outerDists = dists.second;
		#ifdef DEBUG
			if (((int) angle) % 10 == 0) {
				double min1, max1;
				minMaxLoc(innerDists, &min1, &max1);
				imwrite(string("dists") + to_string(layer) + "_" + to_string((int) angle) + ".png", (innerDists - min1) * 255 / (max1 - min1));
			}
		#endif
		if (angle > 0) {
			innerDists = rotate(innerDists, -angle);
			outerDists = rotate(outerDists, -angle);
		}
		Mat lNewInner = innerDists(cropRect);
		Mat lNewOuter = outerDists(cropRect);
		if (angle == 0) {
			// First time
			lInner = lNewInner;
			lOuter = lNewOuter;
		} else {
			for (int row = 0; row < lNewInner.rows; row++) {
				for (int col = 0; col < lNewInner.cols; col++) {
					if (croppedGranules.at<MAT_TYPE_FLOAT>(row, col) == IN_GRANULE) {
						auto newDist = lNewInner.at<MAT_TYPE_FLOAT>(row, col);
						if (newDist > lInner.at<MAT_TYPE_FLOAT>(row, col)) {
							// in granule and new distance is longer
							lInner.at<MAT_TYPE_FLOAT>(row, col) = newDist;
						}
						lOuter.at<MAT_TYPE_FLOAT>(row, col) = INFTY;
					} else {
						auto newDist = lNewOuter.at<MAT_TYPE_FLOAT>(row, col);
						if (newDist < lOuter.at<MAT_TYPE_FLOAT>(row, col)) {
							// intergranular and new distance is shorter
							lOuter.at<MAT_TYPE_FLOAT>(row, col) = newDist;
						}
						lInner.at<MAT_TYPE_FLOAT>(row, col) = 0;
					}
				}
			}
		}
	}
	granuleLabels = granuleLabels(cropRect);

	//-------------------------------------------------------------------------
	// Inner extrema
	//-------------------------------------------------------------------------
	Mat lInnerGlobal = lInner.clone();
	std::ofstream output1(string("inner_global_dists") + to_string(layer) + ".txt");
	auto innerGlobalExtrema = findExtrema(lInner, granuleLabels, greater<float>());
	for (auto extremum : innerGlobalExtrema) {
		if (get<0>(extremum) != 0) {
			output1 << get<0>(extremum) << " " << get<1>(extremum) << " " << get<2>(extremum) << endl;
		}
	}
	output1.close();
	//cout << lOuter << endl;

	Mat lInnerLocal = lInner.clone();
	Mat maximaLabels = labelExtrema(lInner, false);
	std::ofstream output3(string("inner_local_dists") + to_string(layer) + ".txt");
	auto innerLocalExtrema = findExtrema(lInner, maximaLabels, greater<float>());
	for (auto extremum : innerLocalExtrema) {
		output3 << get<0>(extremum) << " " << get<1>(extremum) << " " << get<2>(extremum) << endl;
	}
	output3.close();

	// Convert to 8-bit matrices and normalize from 0 to 255
	convertTo8Bit(lInner);
	convertTo8Bit(lInnerGlobal);
	convertTo8Bit(lInnerLocal);
	//double min, max;
	//minMaxLoc(lInner, &min, &max);
	//lInner.convertTo(lInner, CV_8U, 255.0 / (max - min),-min * 255.0/(max - min));
	//lInnerGlobal.convertTo(lInnerGlobal, CV_8U, 255.0 / (max - min),-min * 255.0/(max - min));
	//lInnerLocal.convertTo(lInnerLocal, CV_8U, 255.0 / (max - min),-min * 255.0/(max - min));
	//lInner = (lInner - min) * 255 / (max - min);
	//lInnerGlobal = (lInnerGlobal - min) * 255 / (max - min);
	//lInnerLocal = (lInnerLocal - min) * 255 / (max - min);

	// Convert to color image and mark positions of extrema red
	Mat lInnerGlobalRGB = convertToColorAndMarkExtrema(lInnerGlobal, innerGlobalExtrema, 0);
	Mat lInnerLocalRGB = convertToColorAndMarkExtrema(lInnerLocal, innerLocalExtrema, 0);
	//cvtColor(lInnerGlobal, lInnerGlobalRGB, CV_GRAY2RGB);
	//for (auto extremum : innerGlobalExtrema) {
	//	int row = get<1>(extremum);
	//	int col = get<2>(extremum);
	//	if (croppedMat.at<MAT_TYPE_FLOAT>(row, col) == IN_GRANULE) {
	//		lInnerGlobalRGB.at<Vec3b>(row, col) = RED;
	//	}
	//}

	// Visualize matrices
	imwrite(string("inner_dists") + to_string(layer) + ".png", lInner);
	imwrite(string("inner_global_extrema") + to_string(layer) + ".png", lInnerGlobalRGB);
	imwrite(string("inner_local_extrema") + to_string(layer) + ".png", lInnerLocalRGB);

	//-------------------------------------------------------------------------
	// Outer extrema
	//-------------------------------------------------------------------------
	Mat lOuterGlobal = lOuter.clone();
	std::ofstream output2(string("outer_global_dists") + to_string(layer) + ".txt");
	auto outerGlobalExtrema = findExtrema(lOuter, granuleLabels, less<float>());
	for (auto extremum : outerGlobalExtrema) {
		if (get<0>(extremum) != INFTY) {
			output2 << get<0>(extremum) << " " << get<1>(extremum) << " " << get<2>(extremum) << endl;
		}
		//if (lOuterGlobal.at<MAT_TYPE_FLOAT>(get<1>(extremum), get<2>(extremum)) != INFTY) {
		//	lOuterGlobal.at<MAT_TYPE_FLOAT>(get<1>(extremum), get<2>(extremum)) = 20 * lOuterGlobal.at<MAT_TYPE_FLOAT>(get<1>(extremum), get<2>(extremum));
		//}
	}
	output2.close();

	Mat lOuterLocal = lOuter.clone();
	Mat minimaLabels = labelExtrema(lOuter, true);
	std::ofstream output4(string("outer_local_dists") + to_string(layer) + ".txt");
	auto outerLocalExtrema = findExtrema(lOuter, minimaLabels, less<float>());
	for (auto extremum : outerLocalExtrema) {
		output4 << get<0>(extremum) << " " << get<1>(extremum) << " " << get<2>(extremum) << endl;
		//auto row = get<1>(extremum);
		//auto col = get<2>(extremum);
		//if (lOuterLocal.at<MAT_TYPE_FLOAT>(row, col) != INFTY) {
		//	lOuterLocal.at<MAT_TYPE_FLOAT>(row, col) = 10 * lOuterLocal.at<MAT_TYPE_FLOAT>(row, col);
		//}
	}
	output4.close();

	// Replace occurrences of INFTY with zeros
	for (int row = 0; row < lOuter.rows; row++) {
		for (int col = 0; col < lOuter.cols; col++) {
			if (lOuter.at<MAT_TYPE_FLOAT>(row, col) == INFTY) {
				lOuter.at<MAT_TYPE_FLOAT>(row, col) = 0;
			}
			if (lOuterGlobal.at<MAT_TYPE_FLOAT>(row, col) == INFTY) {
				lOuterGlobal.at<MAT_TYPE_FLOAT>(row, col) = 0;
			}
			if (lOuterLocal.at<MAT_TYPE_FLOAT>(row, col) == INFTY) {
				lOuterLocal.at<MAT_TYPE_FLOAT>(row, col) = 0;
			}
		}
	}

	convertTo8Bit(lOuter);
	convertTo8Bit(lOuterGlobal);
	convertTo8Bit(lOuterLocal);
	Mat lOuterGlobalRGB = convertToColorAndMarkExtrema(lOuterGlobal, outerGlobalExtrema, INFTY);
	Mat lOuterLocalRGB = convertToColorAndMarkExtrema(lOuterLocal, outerLocalExtrema, INFTY);


	//minMaxLoc(lOuter, &min, &max);
	//lOuter.convertTo(lOuter, CV_8U, 255.0 / (max - min),-min * 255.0/(max - min));
	//lOuterGlobal.convertTo(lOuterGlobal, CV_8U, 255.0 / (max - min),-min * 255.0/(max - min));
	//lOuterLocal.convertTo(lOuterLocal, CV_8U, 255.0 / (max - min),-min * 255.0/(max - min));
	//lOuter = (lOuter - min) * 255 / (max - min);
	//lOuterGlobal = (lOuterGlobal - min) * 255 / (max - min);
	//lOuterLocal = (lOuterLocal - min) * 255 / (max - min);


	imwrite(string("outer_dists") + to_string(layer) + ".png", lOuter);
	imwrite(string("outer_global_extrema") + to_string(layer) + ".png", lOuterGlobalRGB);
	imwrite(string("outer_local_extrema") + to_string(layer) + ".png", lOuterLocalRGB);

}
