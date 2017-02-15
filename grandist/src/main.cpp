#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>
#ifdef GPU
#include <opencv2/gpu/gpu.hpp>
#endif
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#ifdef OPENCV_2_4
	#include <opencv2/contrib/contrib.hpp>
#endif
#include <opencv2/core/core.hpp>
#include "utils/utils.h"
#include "pcdl/SnapshotLoader.h"

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace cv;
using namespace utils;
using namespace pcdl;

//#define DEBUG

// Need to use float matrices even for keeping integer values
// because the library supports rotations only for these
typedef float MAT_TYPE_FLOAT;
// Integer type can be used for matrices not involved in rotations
typedef int MAT_TYPE_INT;

#define OUT_GRANULE 1 ///< The point is inside granule
#define IN_GRANULE 2 ///< The point is outside of the

#define DELTA_ANGLE 1.0 ///< The angle increment used in rotations
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
pair<Mat, Mat> calcDistances(const Mat& mat, bool periodic) {
	Mat innerDists = Mat::zeros(mat.rows, mat.cols, CV_32F);
	// We are later looking for minima in intergranular regions so make the initial values
	// as large as possible. The regions of granules (which we are not interested in) will then keep the higher values.
	Mat outerDists = Mat::ones(mat.rows, mat.cols, CV_32F) * numeric_limits<float>::max();
	for (int col = 0; col < mat.cols; col++) {
		bool inGranule = mat.at<MAT_TYPE_FLOAT>(0, col) == IN_GRANULE;
		int dist = 0;
		int domainStart = mat.rows;
		int startRow = 0;
		int row;
		for (row = 1; row < mat.rows; row++) {
			if (!inDomain(mat, row, col)) { // This point not in domain
				if (inDomain(mat, row - 1, col)) { // Previous point in domain
					// Going out of domain from bottom
					break;
				}
				// Still not entered the domain
				continue;
			}
			if (!inDomain(mat, row - 1, col)) { // Previous point not in domain
				// Entering domain from top
				domainStart = row;
				startRow = row;
				inGranule = mat.at<MAT_TYPE_FLOAT>(row, col) == IN_GRANULE;
				continue;
			}
			if ((mat.at<MAT_TYPE_FLOAT>(row, col) == IN_GRANULE) == inGranule) {
				dist++;
			} else {
				if (!periodic || startRow > domainStart) {
					Mat& dists = inGranule ? innerDists : outerDists;
					for (int row1 = startRow; row1 < row; row1++) {
						dists.at<MAT_TYPE_FLOAT>(row1, col) = dist;
					}
				}
				dist = 0;
				startRow = row;
				inGranule = !inGranule;
			}
		}
		Mat& dists = inGranule ? innerDists : outerDists;
		if (!periodic) {
			for (int row1 = startRow; row1 < row; row1++) {
				dists.at<MAT_TYPE_FLOAT>(row1, col) = dist;
			}
		}
	}
	return make_pair(innerDists, outerDists);
}

/**
 * Helper method for labeling the regions
 */
pair<int, int> labelRow(Mat& granuleLabels, const Mat& mat, const int row, const int col, const int label, function<bool(float, float)> compFunc) {
	granuleLabels.at<MAT_TYPE_INT>(row, col) = label;
	auto value = mat.at<MAT_TYPE_FLOAT>(row, col);
	int startCol;
	for (startCol = col - 1; startCol >= 0; startCol--) {
		if (compFunc(mat.at<MAT_TYPE_FLOAT>(row, startCol), value)) {
			granuleLabels.at<MAT_TYPE_INT>(row, startCol) = label;
		} else {
			break;
		}
	}
	int endCol;
	for (endCol = col + 1; endCol < mat.cols; endCol++) {
		if (compFunc(mat.at<MAT_TYPE_FLOAT>(row, endCol), value)) {
			granuleLabels.at<MAT_TYPE_INT>(row, endCol) = label;
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
void labelConnectedRegion(Mat& granuleLabels, const Mat& mat, const int row, const int col, const int label, function<bool(float, float)> compFunc) {
	//cout << "markClosedRegion " << label << " " << row << " " << col << endl;
	auto startEndCols = labelRow(granuleLabels, mat, row, col, label, compFunc);
	int startCol = startEndCols.first;
	int endCol = startEndCols.second;
	auto value = mat.at<MAT_TYPE_FLOAT>(row, col);
	for (int col1 = startCol; col1 <= endCol; col1++) {
		if (row > 0 && granuleLabels.at<MAT_TYPE_INT>(row - 1, col1) == 0) {
			if (compFunc(mat.at<MAT_TYPE_FLOAT>(row - 1, col1), value)) {
				labelConnectedRegion(granuleLabels, mat, row - 1, col1, label, compFunc);
			}
		}
		if (row < mat.rows - 1 && granuleLabels.at<MAT_TYPE_INT>(row + 1, col1) == 0) {
			if (compFunc(mat.at<MAT_TYPE_FLOAT>(row + 1, col1), value)) {
				labelConnectedRegion(granuleLabels, mat, row + 1, col1, label, compFunc);
			}
		}
	}
	//cout << "markClosedRegion end " << label << " " << row << " " << col << endl;
}

/**
 * Labels the closed regions. Only needed to identify the closed regions if we want to extract only
 * single extremum per region. Otherwise local extrema could be calculated from distance matrix.
 */
Mat labelGranules(const Mat& mat) {
	Mat granuleLabels = Mat::zeros(mat.rows, mat.cols, CV_32S);
	int label = 1;
	for (int row = 0; row < mat.rows; row++) {
		for (int col = 0; col < mat.cols; col++) {
			if (granuleLabels.at<MAT_TYPE_INT>(row, col) == 0) {
				labelConnectedRegion(granuleLabels, mat, row, col, label++, equal_to<float>());
			}
		}
	}
	//////////////////////////////////////////
	//Visualizing if the labeling makes sense
    Mat img;
    Mat granuleLabels2;
    granuleLabels.convertTo(granuleLabels2, CV_8UC3);
    applyColorMap(granuleLabels2, img, COLORMAP_HSV);
	imwrite("granule_labels.png", img);
	//////////////////////////////////////////
	return granuleLabels;
}

/**
 * Labels the isocontours of the distance matrix.
 */
Mat labelExtrema(const Mat& dists, bool minimaOrMaxima) {
	Mat extremaLabels = Mat::zeros(dists.rows, dists.cols, CV_32S);
	int label = 1;
	for (int row = 0; row < dists.rows; row++) {
		for (int col = 0; col < dists.cols; col++) {
			if (extremaLabels.at<MAT_TYPE_INT>(row, col) == 0) {
				if (minimaOrMaxima) {
					labelConnectedRegion(extremaLabels, dists, row, col, label++, less_equal<float>());
				} else {
					labelConnectedRegion(extremaLabels, dists, row, col, label++, greater_equal<float>());
				}
			}
		}
	}
	//////////////////////////////////////////
	// Visualizing if the labeling makes sense
    Mat img;
    Mat extremaLabels2;
    extremaLabels.convertTo(extremaLabels2, CV_8UC3);
    applyColorMap(extremaLabels2, img, COLORMAP_HSV);
	imwrite("extrema_labels.png", img);
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
vector<float> findExtrema(const Mat& dists, const Mat& regionLabels, function<bool(float, float)> compFunc) {
	map<int, float> extremaPerRegion;
	for (int row = 0; row < dists.rows; row++) {
		for (int col = 0; col < dists.cols; col++) {
			int label = regionLabels.at<MAT_TYPE_INT>(row, col);
			auto dist = dists.at<MAT_TYPE_FLOAT>(row, col);
			if (extremaPerRegion.find(label) == extremaPerRegion.end()) {
				extremaPerRegion[label] = dist;
			} else {
				if (compFunc(extremaPerRegion[label], dists.at<MAT_TYPE_FLOAT>(row, col))) {
					extremaPerRegion[label] = dist;
				}
			}
		}
	}
	vector<float> extrema;
	for (auto dist : extremaPerRegion) {
		extrema.push_back(dist.second);
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

int main(int argc, char *argv[]) {
	if (argc == 2 && string("-h") == argv[1]) {
		cout << "Usage: ./D2 [param file] [params to overwrite]\nparam file defaults to " << "parameters.txt" << endl;
		return EXIT_SUCCESS;
	}
	string paramFileName = argc > 1 ? argv[1] : "parameters.txt";
	string cmdLineParams = argc > 2 ? argv[2] : "";
	boost::replace_all(cmdLineParams, " ", "\n");

	if (!exists(paramFileName)) {
		cout << "Cannot find " << paramFileName << endl;
		return EXIT_FAILURE;
	}

	map<string, string> paramsFromFile = Utils::ReadProperties(paramFileName);
	map<string, string> params = Utils::ReadPropertiesFromString(cmdLineParams);
	for (const auto& entry : paramsFromFile) {
		params.insert({entry.first, entry.second});
	}
	SnapshotLoader loader(params);

	int fromLayer = Utils::FindIntProperty(params, "fromLayer", 1);
	int toLayer = Utils::FindIntProperty(params, "toLayer", 0);
	int step = Utils::FindIntProperty(params, "step", 10);
	assert(fromLayer >= 0);
	assert(toLayer == 0 || toLayer >= fromLayer);
	assert(step > 0);

	bool periodic = Utils::FindIntProperty(params, "periodic", 1);

	int verticalCoord = Utils::FindIntProperty(params, "verticalCoord", 2);
	int fstCoord = verticalCoord == 0 ? 1 : (verticalCoord == 1 ? 2 : 0);;
	int sndCoord = verticalCoord == 0 ? 2 : (verticalCoord == 1 ? 0 : 1);

	assert(verticalCoord >= 0 && verticalCoord <= 2);
	auto& dims = loader.getDimsDownSampled();
	//auto numX = dims[0];
	//auto numY = dims[1];
	//auto numZ = dims[2];

	auto depth = dims[verticalCoord];
	auto width = dims[fstCoord];
	auto height = dims[sndCoord];

	Mat matrices[dims[verticalCoord]];
	int rows = ceil(((double) height) * sqrt(2));
	int cols = ceil(((double) width) * sqrt(2));

	int rowOffset = (rows - height) / 2;
	int colOffset = (cols - width) / 2;

	if (periodic) {
		rows = ceil(((double) 2 * height) * sqrt(2));
		cols = ceil(((double) 2 * width) * sqrt(2));

		rowOffset = rows / 2;
		colOffset = cols / 2;
	}


	for (int i = 0; i < depth; i++) {
		matrices[i] = Mat::zeros(rows, cols, CV_32F);//CV_8S);
	}

	//int tLast = -1;
	loader.load([&dims, verticalCoord, fstCoord, sndCoord, &matrices, rowOffset, colOffset](int t, int x, int y, int z, double field) {
		int coord[] = {x, y, z};
		assert(coord[verticalCoord] < dims[verticalCoord]);
		auto& mat = matrices[coord[verticalCoord]];
		assert(y < mat.rows);
		assert(z < mat.cols);
		if (field > 0) {
			mat.at<MAT_TYPE_FLOAT>(coord[sndCoord] + rowOffset, coord[fstCoord] + colOffset) = IN_GRANULE;
		} else {
			mat.at<MAT_TYPE_FLOAT>(coord[sndCoord] + rowOffset, coord[fstCoord] + colOffset) = OUT_GRANULE;
		}
	});

	Rect cropRect(colOffset, rowOffset, width, height);
	int layer = 0;
	for (auto& mat : matrices) {
		if (layer < fromLayer || (layer - fromLayer) % step != 0) {
			layer++;
			continue;
		}
		if (toLayer != 0 && layer > toLayer) {
			break;
		}
		cout << "Processing layer " << layer << endl;
		if (periodic) {
			tileMatrix(mat, height, width);
		}
		Mat croppedMat = mat(cropRect);
		Mat lInner;
		Mat lOuter;
		for (double angle = 0; angle < 360; angle += DELTA_ANGLE) {
			Mat mat1 = mat.clone();
			Mat matRotated = angle > 0 ? rotate(mat1, angle) : mat1;
			#ifdef DEBUG
				if (((int) angle) % 10 == 0) {
					imwrite(string("granules") + to_string(layer) + "_" + to_string((int) angle) + ".png", (matRotated - 1) * 255);
				}
			#endif
			auto dists = calcDistances(matRotated, false);
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
				for (int i = 0; i < lNewInner.rows; i++) {
					for (int j = 0; j < lNewInner.cols; j++) {
						if (croppedMat.at<MAT_TYPE_FLOAT>(i, j) == IN_GRANULE) {
							auto newDist = lNewInner.at<MAT_TYPE_FLOAT>(i, j);
							if (newDist > lInner.at<MAT_TYPE_FLOAT>(i, j)) {
								// in granule and new distance is longer
								lInner.at<MAT_TYPE_FLOAT>(i, j) = newDist;
							}
						} else {
							auto newDist = lNewOuter.at<MAT_TYPE_FLOAT>(i, j);
							if (newDist < lOuter.at<MAT_TYPE_FLOAT>(i, j)) {
								// intergranular and new distance is shorter
								lOuter.at<MAT_TYPE_FLOAT>(i, j) = newDist;
							}
						}
					}
				}
			}
		}
		Mat granuleLabels = labelGranules(mat);

		ofstream output1(string("inner_global_dists") + to_string(layer) + ".txt");
		for (auto extremum : findExtrema(lInner, granuleLabels, greater<float>())) {
			output1 << extremum << endl;
		}
		output1.close();

		ofstream output2(string("outer_global_dists") + to_string(layer) + ".txt");
		for (auto extremum : findExtrema(lOuter, granuleLabels, less<float>())) {
			output2 << extremum << endl;
		}
		output2.close();

		Mat maximaLabels = labelExtrema(mat, true);
		ofstream output3(string("inner_local_dists") + to_string(layer) + ".txt");
		output3.close();
		for (auto extremum : findExtrema(lInner, maximaLabels, greater<float>())) {
			output3 << extremum << endl;
		}
		output3.close();

		Mat minimaLabels = labelExtrema(mat, true);
		ofstream output4(string("outer_local_dists") + to_string(layer) + ".txt");
		output4.close();
		for (auto extremum : findExtrema(lOuter, minimaLabels, less<float>())) {
			output4 << extremum << endl;
		}
		output4.close();

		// Visualize distance matrices
		double min, max;
		minMaxLoc(lInner, &min, &max);
		imwrite(string("inner_dists") + to_string(layer) + ".png", (lInner - min) * 255 / (max - min));
		// Replace occurrences of numeric_limits<float>::max() with zeros
		for (int row = 0; row < lOuter.rows; row++) {
			for (int col = 0; col < lOuter.cols; col++) {
				if (lOuter.at<MAT_TYPE_FLOAT>(row, col) == numeric_limits<float>::max()) {
					lOuter.at<MAT_TYPE_FLOAT>(row, col) = 0;
				}
				//cout << lOuter.at<MAT_TYPE_FLOAT>(row, col) << endl;
			}
		}
		minMaxLoc(lOuter, &min, &max);
		imwrite(string("outer_dists") + to_string(layer) + ".png", (lOuter - min) * 255 / (max - min));

		layer++;
	}

	return 0;
}
