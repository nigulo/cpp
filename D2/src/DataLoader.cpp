/*
 * DataLoader.cpp
 *
 *  Created on: Aug 3, 2015
 *      Author: nigul
 */

#include "DataLoader.h"

DataLoader::DataLoader(const string& fileName, unsigned bufferSize, ios::openmode mode, unsigned dim, unsigned totalNumVars, const vector<unsigned>& varIndices) :
	fileName(fileName),
	bufferSize(bufferSize),
	page(-1),
	mode(mode),
	input(fileName, mode),
	dim(dim),
	totalNumVars(totalNumVars),
	varIndices(varIndices) {

}

DataLoader::DataLoader(const DataLoader& dataLoader) :
	fileName(dataLoader.fileName),
	bufferSize(dataLoader.bufferSize),
	page(-1),
	mode(dataLoader.mode),
	input(fileName, mode),
	dim(dataLoader.dim),
	totalNumVars(dataLoader.totalNumVars),
	varIndices(dataLoader.varIndices) {
}

DataLoader::~DataLoader() {
	if (input.is_open()) {
		input.close();
	}
}

void DataLoader::Reset() {
	x.clear();
	for (auto& yVals : y) {
		delete[] yVals;
	}
	y.clear();
	if (input.is_open()) {
		input.close();
	}
	input.open(fileName, mode);
	page = -1;
}

