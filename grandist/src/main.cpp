#include <iostream>
#include <fstream>
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

#define DEBUG

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
	Mat dists = Mat::zeros(mat.rows, mat.cols, CV_32S);
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
	auto& dims = loader.getDimsDownSampled();
	auto numX = dims[0];
	auto numY = dims[1];
	auto numZ = dims[2];

	Mat matrices[numX];
	int rows = ceil(((double) numY) * sqrt(2));
	int cols = ceil(((double) numZ) * sqrt(2));

	int rowOffset = (rows - numY) / 2;
	int colOffset = (cols - numZ) / 2;

	for (int i = 0; i < numX; i++) {
		matrices[i] = Mat::zeros(rows, cols, CV_8S);
	}

	int tLast = -1;
	loader.load([&tLast, &matrices, rowOffset, colOffset](int t, int r, int theta, int phi, double field) {
		auto& mat = matrices[r];
		if (field > 0) {
			mat.at<char>(theta + rowOffset, phi + colOffset) = 1;
		} else {
			// already initialized to zero;
		}
	});

	Rect cropRect(rowOffset, colOffset, rowOffset + rows, colOffset + cols);
	int x = 0;
	for (auto& mat : matrices) {
		ofstream output(string("dists") + to_string(x) + ".txt");
		Mat l;
		for (double angle = 0; angle < 360; angle += 1) {
			Mat matRotated;
			if (angle > 0) {
				matRotated = rotate(mat, angle);
			} else {
				matRotated = mat;
			}
			#ifdef DEBUG
				imwrite(string("granules") + to_string(x) + "_" + to_string(angle) + ".png", matRotated);
			#endif
			Mat dists = calcDistances(matRotated);
			if (angle > 0) {
				dists = rotate(dists, -angle);
			}
			Mat lNew = dists(cropRect);
			if (angle == 0) {
				// First time
				l = lNew;
			} else {
				for (int i = 0; i < lNew.rows; i++) {
					for (int j = 0; j < lNew.cols; j++) {
						int newDist = lNew.at<int>(i, j);
						if (mat.at<char>(i + rowOffset, j + colOffset)) {
							if (newDist > l.at<int>(i, j)) {
								// in granule and new distance is longer
								l.at<int>(i, j) = newDist;
							}
						} else {
							if (newDist < l.at<int>(i, j)) {
								// intergranular and new distance is shorter
								l.at<int>(i, j) = newDist;
							}
						}
					}
				}
			}
			#ifdef DEBUG
				output << l << endl;
			#endif
		}
		output << l << endl;
		output.close();
		x++;
	}

	return 0;
}
