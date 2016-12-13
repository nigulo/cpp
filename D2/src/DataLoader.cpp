/*
 * DataLoader.cpp
 *
 *  Created on: Aug 3, 2015
 *      Author: nigul
 */

#include "DataLoader.h"

DataLoader::DataLoader(const string& fileName, int bufferSize, ios::openmode mode,
			const vector<int>& dims,
			const vector<vector<pair<int, int>>>& regions,
			int totalNumVars, const vector<int>& varIndices) :
		fileName(fileName),
		bufferSize(bufferSize),
		mode(mode),
		dims(dims),
		regions(regions),
		totalNumVars(totalNumVars),
		varIndices(varIndices),
		input(fileName, mode),
		page(-1),
		data(nullptr),
		pageSize(0) {
	assert(bufferSize > 0);
	assert(input.is_open());
	dim = 1;
	for (auto dimx : dims) {
		dim *= dimx;
	}
	for (int varIndex : varIndices) {
		assert(varIndex < GetTotalNumVars());
	}
	inRegion = new bool[dim];
	for (int i = 0; i < dim; i++) {
		inRegion[i] = InRegion(i);
	}
}

DataLoader::DataLoader(const DataLoader& dataLoader) :
	fileName(dataLoader.fileName),
	bufferSize(dataLoader.bufferSize),
	mode(dataLoader.mode),
	dims(dataLoader.dims),
	regions(dataLoader.regions),
	totalNumVars(dataLoader.totalNumVars),
	varIndices(dataLoader.varIndices),
	input(dataLoader.fileName, dataLoader.mode),
	page(-1),
	data(nullptr),
	pageSize(0),
	dim(dataLoader.dim) {
	assert(input.is_open());
	inRegion = new bool[dim];
	for (int i = 0; i < dim; i++) {
		inRegion[i] = dataLoader.inRegion[i];
	}
}

DataLoader::~DataLoader() {
	if (input.is_open()) {
		input.close();
	}
	if (inRegion) {
		delete[] inRegion;
	}
	if (data) {
		delete[] data;
	}

}

