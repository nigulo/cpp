/*
 * SnapshotLoader.cpp
 *
 *  Created on: Jan 26, 2017
 *      Author: nigul
 */

#include "SnapshotLoader.h"
#include "BinaryDataLoader.h"
#include "utils.h"
#include <cassert>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

using namespace utils;
using namespace boost;
using namespace boost::filesystem;
using namespace pcdl;

SnapshotLoader::SnapshotLoader(const map<string, string>& params) : DataLoader(params) {
}

SnapshotLoader::~SnapshotLoader() {
}

void SnapshotLoader::load(std::function<void(int /*time*/, int /*x*/, int /*y*/, int /*z*/, double /*val*/)> f) {
	assert(dims.size() == 3);
	const int xIndex = 0;
	int numGhost = Utils::FindIntProperty(params, "numGhost", 3);
	string strNumProcs = Utils::FindProperty(params, "numProcs", "1");
	vector<string> numProcsStr;
	vector<int> numProcs;
	boost::split(numProcsStr, strNumProcs, boost::is_any_of(",;"), boost::token_compress_on);
	for (vector<string>::iterator it = numProcsStr.begin() ; it != numProcsStr.end(); ++it) {
		if ((*it).length() != 0) {
			numProcs.push_back(stoi(*it));
			//numProcs.insert(numProcs.begin(), stoi(*it));
		}
	}

	vector<int> dimsPerProc;

	for (size_t i = 0; i < dims.size(); i++) {
		assert(dims[i] % numProcs[i] == 0);
		int procDim = dims[i] / numProcs[i] + 2 * numGhost;
		//cout << "dimsPerProc[" << i << "]=" << procDim << endl;
		dimsPerProc.push_back(procDim);
	}

	assert(numProcs.size() == dims.size());
	auto numProc = accumulate(numProcs.begin(), numProcs.end(), 1, multiplies<int>());
	//cout << "numProc:" << numProc << endl;

	int layer = Utils::FindIntProperty(params, "layer", -1);
	int timeMoment = Utils::FindIntProperty(params, "timeMoment", -1);
	assert(timeMoment >= 0);

	//double wedgeAngle = Utils::FindDoubleProperty(params, "wedgeAngle", 90);
	int totalNumVars = Utils::FindIntProperty(params, "numVars", 10);

	int varIndex = Utils::FindIntProperty(params, "varIndex", 0);
	varIndices.push_back(varIndex);

	for (int procId = 0; procId < numProc; procId++) {
		vector<int> procMinCoords;
		vector<int> procPositions;
		int procSize = 1;
		for (size_t i = 0; i < dims.size(); i++) {
		//for (int i = dims.size() - 1; i >= 0; i--) {
			int procPos = (procId / procSize) % numProcs[i];
			procPositions.push_back(procPos);
			procSize *= numProcs[i];
			int procMinCoord = procPos * (dimsPerProc[i] - 2 * numGhost);
			//procMinCoords.insert(procMinCoords.begin(), procMinCoord);
			procMinCoords.push_back(procMinCoord);
			//cout << "PROC" << procId << " minCoord for " << i << ": " << procMinCoord << "\n";
		}

		if (filePath[filePath.length() - 1] != '/') {
			filePath += "/";
		}
		string dataFile = filePath + "proc" + to_string(procId) + "/VAR" + to_string(timeMoment);
		cout << "Reading: " << dataFile << endl;
		assert(exists(dataFile));
		BinaryDataLoader dl(dataFile, 1000000, dimsPerProc, regions, totalNumVars, varIndices, TYPE_SNAPSHOT);
		//cout << "procId:" << procId << endl;
		dl.Next();
		assert(dl.GetPageSize() == 1);
		//real time = dl.GetX(0);
		auto value = dl.GetY(0);
		for (int i = 0; i < dl.GetDim(); i++) {
			auto i1 = i;
			vector<int> coords(3);
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
			bool ghost = false;
			for (int j = 0; j < (int) dimsPerProc.size(); j++) {
				if (coords[j] < numGhost || (coords[j] + numGhost) >= dimsPerProc[j]) {
					ghost = true;
					break;
				}
			}
			//cout << ghost << endl;
			if (!ghost) {
				auto x = coords[xIndex] - numGhost + procMinCoords[xIndex];
				if (layer < 0 || x == layer) {
					int y = coords[yIndex] - numGhost + procMinCoords[yIndex];
					int z = coords[zIndex] - numGhost + procMinCoords[zIndex];
					if ((y % yDownSample == 0) && (z % zDownSample == 0)) {
						f(timeMoment, x, y / yDownSample, z / zDownSample, value[i]);
					}
				}
			}
		}
	}
}
