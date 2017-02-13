#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>
#ifdef GPU
#include <opencv2/gpu/gpu.hpp>
#endif
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils/utils.h"
#include "pcdl/SnapshotLoader.h"

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace cv;
using namespace utils;
using namespace pcdl;

#define DEBUG
typedef float MAT_TYPE;

#define OUT_GRANULE 1
#define IN_GRANULE 2

Mat rotate(const Mat& src, double angle, bool dists) {
	Mat dst;
	// Create a destination to paint the source into.
	dst.create(src.size(), src.type());
	#ifdef GPU
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
		/// Get the rotation matrix with the specifications above
		rot_mat = getRotationMatrix2D(center, angle, 1.0);
		/// Rotate the warped image
		warpAffine(src, dst, rot_mat, src.size(), dists ? INTER_LINEAR : INTER_NEAREST);

	#endif
	return dst;
}

bool inDomain(const Mat& mat, int row, int col) {
	return mat.at<MAT_TYPE>(row, col) == IN_GRANULE || mat.at<MAT_TYPE>(row, col) == OUT_GRANULE;
}

pair<Mat, Mat> calcDistances(const Mat& mat, bool periodic) {
	//cout << "calcDistances1" << endl;
	Mat innerDists = Mat::zeros(mat.rows, mat.cols, CV_32F);//, CV_32S);
	Mat outerDists = Mat::ones(mat.rows, mat.cols, CV_32F);//, CV_32S);
	//cout << "calcDistances2" << endl;
	//if (log) {
	//	cout << "Rows: " << mat.rows << endl;
	//}
	for (int col = 0; col < mat.cols; col++) {
		//cout << "calcDistances3 " << col << endl;
		bool inGranule = mat.at<MAT_TYPE>(0, col) == IN_GRANULE;
		int dist = 0;
		int domainStart = mat.rows;
		int startRow = 0;
		int row;
		//cout << "calcDistances4 " << col << endl;
		for (row = 1; row < mat.rows; row++) {
			if (!inDomain(mat, row, col)) { // This point not in domain
				if (inDomain(mat, row - 1, col)) { // Previous point in domain
					//if (log) {
					//	cout << "Going out of domain: col=" << col << ", row = " << row << ", " << mat.at<MAT_TYPE>(row, col) << endl;
					//}
					// Going out of domain from bottom
					break;
				}
				// Still not entered the domain
				continue;
			}
			if (!inDomain(mat, row - 1, col)) { // Previous point not in domain
				//if (log) {
				//	cout << "Entering domain: col=" << col << ", row = " << row << endl;
				//}
				// Entering domain from top
				domainStart = row;
				startRow = row;
				inGranule = mat.at<MAT_TYPE>(row, col) == IN_GRANULE;
				continue;
			}
			//cout << "calcDistances5 " << col << " " << row << endl;
			if ((mat.at<MAT_TYPE>(row, col) == IN_GRANULE) == inGranule) {
				//cout << "calcDistances6 " << col << " " << row << endl;
				dist++;
			} else {
				//cout << "calcDistances7 " << col << " " << row << endl;
				Mat& dists = inGranule ? innerDists : outerDists;
				for (int row1 = startRow; row1 < row; row1++) {
					dists.at<MAT_TYPE>(row1, col) = dist;
				}
				//cout << "calcDistances8 " << col << " " << row << endl;
				dist = 0;
				startRow = row;
				inGranule = !inGranule;
			}
		}
		//cout << "calcDistances9 " << col << endl;
		Mat& dists = inGranule ? innerDists : outerDists;
		if (periodic) {
			int domainEnd = row;
			int endRow;
			for (endRow = domainStart; endRow < startRow && (mat.at<MAT_TYPE>(endRow, col) == IN_GRANULE) == inGranule; endRow++) {
				dist++;
			}
			//cout << "calcDistances9.5 " << endRow << endl;
			for (int row1 = domainStart; row1 < endRow; row1++) {
				dists.at<MAT_TYPE>(row1, col) = dist;
			}
			//cout << "calcDistances9.7 " << col << endl;
			for (int row1 = startRow; row1 < domainEnd; row1++) {
				dists.at<MAT_TYPE>(row1, col) = dist;
			}
		} else {
			for (int row1 = startRow; row1 < row; row1++) {
				dists.at<MAT_TYPE>(row1, col) = dist;
			}
		}
		//cout << "calcDistances10 " << col << endl;
	}
	//cout << "calcDistances11" << endl;
	return make_pair(innerDists, outerDists);
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
			mat.at<MAT_TYPE>(coord[sndCoord] + rowOffset, coord[fstCoord] + colOffset) = IN_GRANULE;
		} else {
			mat.at<MAT_TYPE>(coord[sndCoord] + rowOffset, coord[fstCoord] + colOffset) = OUT_GRANULE;
		}
	});

	Rect cropRect(colOffset, rowOffset, width, height);
	int layer = 0;
	for (auto& mat : matrices) {
		if (layer != 280) {
			layer++;
			continue;
		}
		ofstream output(string("dists") + to_string(layer) + ".txt");
		Mat lInner;
		Mat lOuter;
		for (double angle = 0; angle < 360; angle += 1) {
			Mat mat1 = mat.clone();
			Mat matRotated = angle > 0 ? rotate(mat1, angle, false) : mat1;
			//#ifdef DEBUG
			//	if (layer == 280 && ((int) angle) % 10 == 0) {
			//		imwrite(string("granules") + to_string(layer) + "_" + to_string((int) angle) + ".png", (matRotated - 1) * 255);
			//	}
			//#endif
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
						if (mat.at<MAT_TYPE>(i + rowOffset, j + colOffset) == IN_GRANULE) {
							int newDist = lNewInner.at<MAT_TYPE>(i, j);
							if (newDist > lInner.at<MAT_TYPE>(i, j)) {
								// in granule and new distance is longer
								lInner.at<MAT_TYPE>(i, j) = newDist;
							}
						} else {
							int newDist = lNewOuter.at<MAT_TYPE>(i, j);
							if (newDist < lOuter.at<MAT_TYPE>(i, j)) {
								// intergranular and new distance is shorter
								lOuter.at<MAT_TYPE>(i, j) = newDist;
							}
						}
					}
				}
			}
			#ifdef DEBUG
			//	output << l << endl;
			#endif
		}
		//output << lInner << endl;
		//output.close();
		double min, max;
		minMaxLoc(lInner, &min, &max);
		imwrite(string("inner_dists") + to_string(layer) + ".png", (lInner - min) * 255 / (max - min));
		minMaxLoc(lOuter, &min, &max);
		imwrite(string("outer_dists") + to_string(layer) + ".png", (lOuter - min) * 255 / (max - min));
		layer++;
	}

	return 0;
}
