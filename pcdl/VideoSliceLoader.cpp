/*
 * VideoSliceLoader.cpp
 *
 *  Created on: Jan 26, 2017
 *      Author: nigul
 */

#include "VideoSliceLoader.h"
#include "utils.h"
#include <cassert>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

using namespace utils;
using namespace boost;
using namespace boost::filesystem;
using namespace pcdl;

VideoSliceLoader::VideoSliceLoader(const map<string, string>& params, std::function<void(const string&)> logFunc) : DataLoader(params, logFunc) {
}

VideoSliceLoader::~VideoSliceLoader() {
}

void VideoSliceLoader::load(std::function<void(int /*y*/, int /*z*/, double /*val*/)> f1, std::function<void(int /*time*/)> f2) {
	assert(dims.size() == 2);
	varIndices.push_back(0);
	int startTime = Utils::FindIntProperty(params, "startTime", 0);
	int endTime = Utils::FindIntProperty(params, "endTime", -1);

	string dataFile = filePath;
	logFunc("Reading: " + dataFile + "\n");
	assert(exists(dataFile));
	BinaryDataLoader dl(dataFile, bufferSize, dims, regions, 1 /*totalNumVars*/, varIndices, TYPE_VIDEO, prec);
	int timeOffset = 0;
	while (dl.Next()) {
		//cout << "procId:" << procId << endl;
		for (int t = 0; t < dl.GetPageSize(); t++) {
			if (t + timeOffset < startTime) {
				continue;
			}
			if (endTime >= 0 && t + timeOffset > endTime) {
				break;
			}
			const auto timeIndex = t + timeOffset;
			double time = dl.GetX(t);
			logFunc("Reading time moment " + to_string(timeIndex) + " (" + to_string(time) + ")\n");
			auto values = dl.GetY(t);
			for (int i = 0; i < dl.GetDim(); i++) {
				auto i1 = i;
				vector<int> coords(dl.GetDims().size());
				//cout << "coords= ";
				for (int j = 0; j < (int) dl.GetDims().size(); j++) {
					int coord = i1 % dl.GetDims()[j];
					//cout << coord << ", ";
					coords[j] = coord;
					//if (j == thetaIndex && (procPositions[j] % 2) != 0) {
					//	//cout << procPositions[j] << endl;
					//	coords[j] = dl.GetDims()[j] - coords[j] - 1;
					//}
					i1 -= coord;
					i1 /= dl.GetDims()[j];
				}
				int y = coords[yIndex];
				int z = coords[zIndex];
				if ((y % yDownSample == 0) && (z % zDownSample == 0)) {
					double value;
					if (prec == SinglePrecision) {
						value = ((float*) values)[i];
					} else {
						value = ((double*) values)[i];
					}
					f1(y / yDownSample, z / zDownSample, value);
				}
			}
			f2(timeIndex);
		}
		timeOffset += dl.GetPageSize();
		if (endTime >= 0 && timeOffset > endTime) {
			break;
		}
	}
}
