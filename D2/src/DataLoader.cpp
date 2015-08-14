/*
 * DataLoader.cpp
 *
 *  Created on: Aug 3, 2015
 *      Author: nigul
 */

#include "DataLoader.h"

DataLoader::DataLoader(const string& fileName, unsigned bufferSize, ios::openmode mode,
		const vector<unsigned>& dims,
		const vector<unsigned>& mins,
		const vector<unsigned>& maxs,
		unsigned totalNumVars, const vector<unsigned>& varIndices) :
	fileName(fileName),
	bufferSize(bufferSize),
	mode(mode),
	dims(dims),
	mins(mins),
	maxs(maxs),
	totalNumVars(totalNumVars),
	varIndices(varIndices),
	input(fileName, mode),
	page(-1),
	data(nullptr),
	pageSize(0) {
	assert(input.is_open());
	for (unsigned varIndex : varIndices) {
		assert(varIndex < GetYSize());
	}
	dim = 1;
	for (auto dimx : dims) {
		dim *= dimx;
	}
}

DataLoader::DataLoader(const DataLoader& dataLoader) :
	fileName(dataLoader.fileName),
	bufferSize(dataLoader.bufferSize),
	mode(dataLoader.mode),
	dims(dataLoader.dims),
	mins(dataLoader.mins),
	maxs(dataLoader.maxs),
	totalNumVars(dataLoader.totalNumVars),
	varIndices(dataLoader.varIndices),
	input(dataLoader.fileName, dataLoader.mode),
	page(-1),
	data(nullptr),
	pageSize(0),
	dim(dataLoader.dim) {
	assert(input.is_open());
}

DataLoader::~DataLoader() {
	if (input.is_open()) {
		input.close();
	}
}

