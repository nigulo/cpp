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
	mode(mode),
	dim(dim),
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
}

DataLoader::DataLoader(const DataLoader& dataLoader) :
	fileName(dataLoader.fileName),
	bufferSize(dataLoader.bufferSize),
	mode(dataLoader.mode),
	dim(dataLoader.dim),
	totalNumVars(dataLoader.totalNumVars),
	varIndices(dataLoader.varIndices),
	input(fileName, mode),
	page(-1),
	data(nullptr),
	pageSize(0) {
	assert(input.is_open());
}

DataLoader::~DataLoader() {
	if (input.is_open()) {
		input.close();
	}
}

