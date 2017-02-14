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

/**
 * Rotates the matrix
 * @param[in] src matrix to rotate
 * @param[in] angle angle of rotation in degrees.
 * @param[in] dists either distance matrix or matrix of granules.
 * Determines which type of interpolation is used.
 * @return the rotated matrix
 *
 */
Mat rotate(const Mat& src, double angle, bool dists) {
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
		warpAffine(src, dst, rot_mat, src.size(), dists ? INTER_LINEAR : INTER_NEAREST);

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
	Mat innerDists = Mat::zeros(mat.rows, mat.cols, CV_32F);//, CV_32S);
	Mat outerDists = Mat::ones(mat.rows, mat.cols, CV_32F);//, CV_32S);
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
 * Helper method for labeling the closed regions
 */
pair<int, int> markRow(Mat& granuleLabels, const Mat& mat, const int row, const int col, const int label) {
	granuleLabels.at<MAT_TYPE_INT>(row, col) = label;
	auto value = mat.at<MAT_TYPE_FLOAT>(row, col);
	int startCol;
	for (startCol = col - 1; startCol >= 0; startCol--) {
		if (mat.at<MAT_TYPE_FLOAT>(row, startCol) == value) {
			granuleLabels.at<MAT_TYPE_INT>(row, startCol) = label;
		} else {
			break;
		}
	}
	int endCol;
	for (endCol = col + 1; endCol < mat.cols; endCol++) {
		if (mat.at<MAT_TYPE_FLOAT>(row, endCol) == value) {
			granuleLabels.at<MAT_TYPE_INT>(row, endCol) = label;
		} else {
			break;
		}
	}
	//cout << "markRow " << (startCol + 1) << " " << (endCol - 1) << endl;
	return make_pair(startCol + 1, endCol - 1);
}

/**
 * Helper method for labeling the closed regions
 */
void markClosedRegion(Mat& granuleLabels, const Mat& mat, const int row, const int col, const int label) {
	//cout << "markClosedRegion " << label << " " << row << " " << col << endl;
	auto startEndCols = markRow(granuleLabels, mat, row, col, label);
	int startCol = startEndCols.first;
	int endCol = startEndCols.second;
	auto value = mat.at<MAT_TYPE_FLOAT>(row, col);
	for (int col1 = startCol; col1 <= endCol; col1++) {
		if (row > 0 && granuleLabels.at<MAT_TYPE_INT>(row - 1, col1) == 0) {
			if (mat.at<MAT_TYPE_FLOAT>(row - 1, col1) == value) {
				markClosedRegion(granuleLabels, mat, row - 1, col1, label);
			}
		}
		if (row < mat.rows - 1 && granuleLabels.at<MAT_TYPE_INT>(row + 1, col1) == 0) {
			if (mat.at<MAT_TYPE_FLOAT>(row + 1, col1) == value) {
				markClosedRegion(granuleLabels, mat, row + 1, col1, label);
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
				markClosedRegion(granuleLabels, mat, row, col, label++);
			}
		}
	}
	return granuleLabels;
	//////////////////////////////////////////
	// Visualizing if the labeling makes sense
    //Mat img;
    //granuleLabels.convertTo(granuleLabels, CV_8UC3);
    //applyColorMap(granuleLabels, img, COLORMAP_HSV);
	//imwrite("granule_labels.png", img);
	//////////////////////////////////////////
}

/**
 * Calculates extrema of distances
 * @param[in] dists distance matrix
 * @param[in] mat matrix of down/up flows
 * @param[in] granuleLabelsmatrix of closed region labels
 * @param[in] inGranule inter- or intragranular regions
 */
vector<float> calcExtrema(const Mat& dists, const Mat& mat, const Mat& granuleLabels, bool inGranule) {
	map<int, float> granuleSizes;
	for (int row = 0; row < mat.rows; row++) {
		for (int col = 0; col < mat.cols; col++) {
			if ((mat.at<MAT_TYPE_FLOAT>(row, col) == IN_GRANULE) == inGranule) {
				int label = granuleLabels.at<MAT_TYPE_INT>(row, col);
				if (granuleSizes.find(label) == granuleSizes.end()) {
					granuleSizes[label] = dists.at<MAT_TYPE_FLOAT>(row, col);
				} else {
					if (inGranule) {
						if (granuleSizes[label] < dists.at<MAT_TYPE_FLOAT>(row, col)) {
							granuleSizes[label] = dists.at<MAT_TYPE_FLOAT>(row, col);
						}
					} else {
						if (granuleSizes[label] > dists.at<MAT_TYPE_FLOAT>(row, col)) {
							granuleSizes[label] = dists.at<MAT_TYPE_FLOAT>(row, col);
						}
					}
				}
			}
		}
	}
	vector<float> extrema;
	for (auto dist : granuleSizes) {
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
		if (layer != 280) {
			layer++;
			continue;
		}
		if (periodic) {
			tileMatrix(mat, height, width);
		}
		Mat croppedMat = mat(cropRect);
		ofstream output(string("dists") + to_string(layer) + ".txt");
		Mat lInner;
		Mat lOuter;
		for (double angle = 0; angle < 360; angle += 1) {
			Mat mat1 = mat.clone();
			Mat matRotated = angle > 0 ? rotate(mat1, angle, false) : mat1;
			#ifdef DEBUG
				if (layer == 280 && ((int) angle) % 10 == 0) {
					imwrite(string("granules") + to_string(layer) + "_" + to_string((int) angle) + ".png", (matRotated - 1) * 255);
				}
			#endif
			auto dists = calcDistances(matRotated, false);
			auto innerDists = dists.first;
			auto outerDists = dists.second;
			#ifdef DEBUG
				if (layer == 280 && ((int) angle) % 10 == 0) {
					double min1, max1;
					minMaxLoc(innerDists, &min1, &max1);
					imwrite(string("dists") + to_string(layer) + "_" + to_string((int) angle) + ".png", (innerDists - min1) * 255 / (max1 - min1));
				}
			#endif
			if (angle > 0) {
				innerDists = rotate(innerDists, -angle, true);
				outerDists = rotate(outerDists, -angle, true);
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
							int newDist = lNewInner.at<MAT_TYPE_FLOAT>(i, j);
							if (newDist > lInner.at<MAT_TYPE_FLOAT>(i, j)) {
								// in granule and new distance is longer
								lInner.at<MAT_TYPE_FLOAT>(i, j) = newDist;
							}
						} else {
							int newDist = lNewOuter.at<MAT_TYPE_FLOAT>(i, j);
							if (newDist < lOuter.at<MAT_TYPE_FLOAT>(i, j)) {
								// intergranular and new distance is shorter
								lOuter.at<MAT_TYPE_FLOAT>(i, j) = newDist;
							}
						}
					}
				}
			}
		}
		double min, max;
		minMaxLoc(lInner, &min, &max);
		imwrite(string("inner_dists") + to_string(layer) + ".png", (lInner - min) * 255 / (max - min));
		minMaxLoc(lOuter, &min, &max);
		imwrite(string("outer_dists") + to_string(layer) + ".png", (lOuter - min) * 255 / (max - min));
		Mat granuleLabels = labelGranules(mat);
		for (auto maximum : calcExtrema(lInner, croppedMat, granuleLabels, true)) {
			output << maximum << endl;
		}
		//output << endl;
		//for (auto minimum : calcExtrema(lOuter, croppedMat, granuleLabels, false)) {
		//	output << minimum << endl;
		//}
		output.close();

		layer++;
	}

	return 0;
}
