/*
 * DataLoader.cpp
 *
 *  Created on: Jan 26, 2017
 *      Author: nigul
 */

#include "DataLoader.h"
#include "utils.h"
#include <cassert>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

using namespace utils;
using namespace boost;
using namespace boost::filesystem;
using namespace pcdl;

DataLoader::DataLoader(const map<string, string>& params) : params(params) {
	string strDims = Utils::FindProperty(params, "dims", "1");
	vector<string> dimsStr;
	boost::split(dimsStr, strDims, boost::is_any_of(",;"), boost::token_compress_on);
	for (vector<string>::iterator it = dimsStr.begin() ; it != dimsStr.end(); ++it) {
		if ((*it).length() != 0) {
			dims.push_back(stoi(*it));
			//dims.insert(dims.begin(), stoi(*it));
		}
	}

	assert(dims.size() >= 2);

	thetaIndex = 1;
	phiIndex = 2;
	if (dims.size() == 2) {
		thetaIndex = 0;
		phiIndex = 1;
	}
	thetaDownSample = Utils::FindIntProperty(params, "thetaDownSample", 1);
	phiDownSample = Utils::FindIntProperty(params, "phiDownSample", 1);

	int numTheta = dims[thetaIndex];
	int numPhi = dims[phiIndex];

	assert(thetaDownSample > 0 && (numTheta % thetaDownSample == 0));
	assert(phiDownSample > 0 && (numPhi % phiDownSample == 0));

	numTheta /= thetaDownSample;
	numPhi /= phiDownSample;

	regions.push_back(vector<pair<int, int>>());

	polarGap = Utils::FindDoubleProperty(params, "polarGap", 15);
	polarGap /= 360; // Convert to NFSFT units

	filePath = Utils::FindProperty(params, string("filePath"), "");
	assert(filePath.size() > 0);

}


DataLoader::~DataLoader() {
}

