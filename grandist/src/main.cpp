//============================================================================
// Name        : garndist.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <opencv2/gpu/gpu.hpp>
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

Mat rotate(const Mat& src, double angle) {
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
		warpAffine(src, dst, rot_mat, src.size());

	#endif
	return dst;
}

Mat calcDistances(const Mat& mat) {
	Mat dists = Mat::zeros(mat.rows, mat.cols, mat.type());
	for (int col = 0; col < mat.cols; col++) {
		bool inGranule = mat.at<char>(0, col);
		int dist = 0;
		int startRow = 0;
		int row;
		for (row = 1; row < mat.rows; row++) {
			if (mat.at<char>(row, col) == inGranule) {
				dist++;
			} else {
				for (int row1 = startRow; row1 < row; row1++) {
					dists.at<char>(row1, col) = dist;
				}
				dist = 0;
				startRow = row;
				inGranule = !inGranule;
			}
		}
		for (int row1 = startRow; row1 < row; row1++) {
			dists.at<char>(row1, col) = dist;
		}
	}
	return dists;
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
	auto& dims = loader.getDims();
	auto numR = dims[0];
	auto numTheta = dims[1];
	auto numPhi = dims[2];

	Mat matrices[numR];
	int rows = ceil(((double) numTheta) * sqrt(2));
	int cols = ceil(((double) numPhi) * sqrt(2));

	int rowOffset = (rows - numTheta) / 2;
	int colOffset = (cols - numPhi) / 2;

	for (int i = 0; i < numR; i++) {
		matrices[i] = Mat::zeros(rows, cols, CV_8S);
	}

	loader.load([&matrices, rowOffset, colOffset](int t, int r, int theta, int phi, double field) {
		auto& mat = matrices[r];
		if (field > 0) {
			mat.at<char>(theta + rowOffset, phi + colOffset) = 1;
		} else {
			// already initialized to zero;
		}
	});

	for (auto& mat : matrices) {
		for (double angle = 0; angle < 360; angle += 1) {
			Mat matRotated;
			if (angle > 0) {
				matRotated = rotate(mat, angle);
			} else {
				matRotated = mat;
			}
		}
	}

	return 0;
}
