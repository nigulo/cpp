#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>
#include "utils/utils.h"
#include "pcdl/SnapshotLoader.h"
#include "GranDist.h"
#include "common.h"
#ifndef _NOMPI
#include "mpi.h"
#endif

using namespace boost;
using namespace boost::filesystem;
using namespace utils;
using namespace pcdl;

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

	#ifdef _NOMPI
		int numProc = 1;
		int procId = 0;
	#else
		MPI::Init(argc, argv);
		int numProc = MPI::COMM_WORLD.Get_size();
		int procId = MPI::COMM_WORLD.Get_rank();
	#endif


	map<string, string> paramsFromFile = Utils::ReadProperties(paramFileName);
	map<string, string> params = Utils::ReadPropertiesFromString(cmdLineParams);
	for (const auto& entry : paramsFromFile) {
		params.insert({entry.first, entry.second});
	}

	int fromLayer = Utils::FindIntProperty(params, "fromLayer", 1);
	int toLayer = Utils::FindIntProperty(params, "toLayer", 0);
	int startTime = Utils::FindIntProperty(params, "startTime", 100);
	int endTime = Utils::FindIntProperty(params, "endTime", 0);
	int step = Utils::FindIntProperty(params, "step", 10);
	assert(fromLayer >= 0);
	assert(toLayer == 0 || toLayer >= fromLayer);
	assert(step > 0);

	bool periodic = Utils::FindIntProperty(params, "periodic", 1);

	int verticalCoord = Utils::FindIntProperty(params, "verticalCoord", 2);
	int fstCoord = verticalCoord == 0 ? 1 : (verticalCoord == 1 ? 2 : 0);;
	int sndCoord = verticalCoord == 0 ? 2 : (verticalCoord == 1 ? 0 : 1);

	assert(verticalCoord >= 0 && verticalCoord <= 2);

	string filePath = Utils::FindProperty(params, "filePath", "");
	filePath += "proc0";

	directory_iterator end_itr; // default construction yields past-the-end
	path dir(filePath);
	vector<int> timeMoments;
	int timeMomentIndex = 0;
	for (directory_iterator itr(dir); itr != end_itr; ++itr) {
		if (!is_regular_file(itr->status())) {
			continue;
		}
		const auto& fileName = itr->path().stem();
		auto index = fileName.find("VAR");
		if (index != 0) {
			continue;
		}
		int timeMoment = stoi(fileName.substr(index + 3));
		if (timeMoment < startTime || (endTime > 0 && timeMoment > endTime)) {
			continue;
		}
		if (timeMomentIndex++ % numProc == procId) {
			timeMoments.push_back(timeMoment);
		}
	}

	for (int timeMoment : timeMoments) {
		params["timeMoment"] = to_string(timeMoment);

		SnapshotLoader loader(params);
		auto& dims = loader.getDimsDownSampled();
		//auto numX = dims[0];
		//auto numY = dims[1];
		//auto numZ = dims[2];

		auto depth = dims[verticalCoord];
		auto width = dims[fstCoord];
		auto height = dims[sndCoord];

		int numLayers = dims[verticalCoord];
		Mat matrices[numLayers];
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
			// Using hidden fact that RegionType.OUT_OF_DOMAIN is 0
			matrices[i] = Mat::zeros(rows, cols, CV_32F);
		}

		loader.load([&dims, verticalCoord, fstCoord, sndCoord, &matrices, rowOffset, colOffset](int t, int x, int y, int z, double field) {
			int coord[] = {x, y, z};
			assert(coord[verticalCoord] < dims[verticalCoord]);
			auto& mat = matrices[coord[verticalCoord]];
			assert(y < mat.rows);
			assert(z < mat.cols);
			if (field > 0) {
				mat.at<MAT_TYPE_FLOAT>(coord[sndCoord] + rowOffset, coord[fstCoord] + colOffset) = UP_FLOW;
			} else {
				mat.at<MAT_TYPE_FLOAT>(coord[sndCoord] + rowOffset, coord[fstCoord] + colOffset) = DOWN_FLOW;
			}
		});

		Rect cropRect(colOffset, rowOffset, width, height);
		#pragma omp parallel for
		for (int layer = 0; layer < numLayers; layer++) {
			auto& mat = matrices[layer];
			if (layer < fromLayer || (layer - fromLayer) % step != 0) {
				continue;
			}
			if (toLayer == 0 || layer <= toLayer) {
				cout << "Processing layer " << layer << endl;
				GranDist granDist(timeMoment, layer, mat, height, width, periodic, cropRect);
				granDist.process();
			}
		}
	}

	return 0;
}
