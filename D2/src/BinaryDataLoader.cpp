#include "BinaryDataLoader.h"
#include <iostream>
using namespace std;

BinaryDataLoader::BinaryDataLoader(const string& fileName, unsigned bufferSize, unsigned dim, unsigned totalNumVars, const vector<unsigned>& varIndices) :
		DataLoader(fileName, bufferSize, ios::in | ios::binary, dim, totalNumVars, varIndices) {
}

BinaryDataLoader::BinaryDataLoader(const BinaryDataLoader& dataLoader) : DataLoader(dataLoader) {
	if (dataLoader.page > 1) {
		input.seekg((dataLoader.page - 1) * bufferSize * (dim * totalNumVars + 1) * sizeof (real), ios::cur);
		page = dataLoader.page - 1;
	}
	if (dataLoader.page >= 0) {
		Next();
	}
}

BinaryDataLoader::~BinaryDataLoader() {
}


bool BinaryDataLoader::Next() {
	if (!input.is_open()) {
		return false;
	}
	page++;
	delete[] data;
	unsigned dataSize = bufferSize * (dim * totalNumVars + 1);
	data = new real[dataSize];
	input.read((char*) data, (sizeof (real)) * bufferSize * (dim * totalNumVars + 1));
	unsigned numBytesRead = input.gcount();
	if (numBytesRead < (sizeof (real)) * dataSize) {
		assert(numBytesRead % ((sizeof (real)) * (dim * totalNumVars + 1)) == 0);
		pageSize = numBytesRead / ((sizeof (real)) * (dim * totalNumVars + 1));
		input.close();
	} else {
		pageSize = bufferSize;
	}
	return pageSize > 0;
}

unique_ptr<DataLoader> BinaryDataLoader::Clone() const {
	return unique_ptr<DataLoader>(new BinaryDataLoader(*this));
}
