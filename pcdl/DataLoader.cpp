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
#include <iostream>

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

	assert(dims.size() == 2 || dims.size() == 3);

	xIndex = 0;
	yIndex = 1;
	zIndex = 2;
	if (dims.size() == 2) {
		xIndex = -1;
		yIndex = 0;
		zIndex = 1;
	}

	xDownSample = Utils::FindIntProperty(params, "xDownSample", 1);
	yDownSample = Utils::FindIntProperty(params, "yDownSample", 1);
	zDownSample = Utils::FindIntProperty(params, "zDownSample", 1);

	// Just for backward compatibility
	int thetaDownSample = Utils::FindIntProperty(params, "thetaDownSample", 0);
	int phiDownSample = Utils::FindIntProperty(params, "phiDownSample", 0);
	if (thetaDownSample > 0) {
		yDownSample = thetaDownSample;
	}
	if (phiDownSample > 0) {
		zDownSample = phiDownSample;
	}

	if (xIndex > 0) {
		assert(xDownSample > 0 && (dims[xIndex] % xDownSample == 0));
	}
	assert(yDownSample > 0 && (dims[yIndex] % yDownSample == 0));
	assert(zDownSample > 0 && (dims[zIndex] % zDownSample == 0));

	dimsDownSampled.resize(zIndex + 1);
	dimsDownSampled[0] = dims[0]; // x (if present, otherwise y)
	dimsDownSampled[yIndex] = dims[yIndex] / yDownSample;
	dimsDownSampled[zIndex] = dims[zIndex] / zDownSample;

	regions.push_back(vector<pair<int, int>>());

	filePath = Utils::FindProperty(params, "filePath", "");
	assert(filePath.size() > 0);

	string strPrec = Utils::FindProperty(params, "precision", "single");
	to_lower(strPrec);
	assert(strPrec == "single" || strPrec == "double");
	prec = SinglePrecision;
	if (strPrec == "double") {
		prec = DoublePrecision;
	}
	bufferSize = Utils::FindIntProperty(params, "bufferSize", prec == SinglePrecision ? 8000 : 4000);

}


DataLoader::~DataLoader() {
}

