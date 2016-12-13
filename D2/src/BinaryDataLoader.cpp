#include "BinaryDataLoader.h"
#include <iostream>
using namespace std;

BinaryDataLoader::BinaryDataLoader(const string& fileName, int bufferSize,
		const vector<int>& dims,
		const vector<vector<pair<int, int>>>& regions,
		int totalNumVars, const vector<int>& varIndices) :
				DataLoader(fileName, bufferSize, ios::in | ios::binary, dims, regions, totalNumVars, varIndices) {
}

// Creates new DataLoader with the current page set to the next page of input DataLoader
BinaryDataLoader::BinaryDataLoader(const BinaryDataLoader& dataLoader) : DataLoader(dataLoader) {
	if (dataLoader.page >= 0) {
		if (RECORDHEADER) {
			input.seekg((dataLoader.page + 1) * bufferSize * ((dim * totalNumVars + 1) * sizeof (real) + 16), ios::cur);
		} else {
			input.seekg((dataLoader.page + 1) * bufferSize * (dim * totalNumVars + 1) * sizeof (real), ios::cur);
		}
	}
	page = dataLoader.page;
	Next();
}

BinaryDataLoader::~BinaryDataLoader() {
}


bool BinaryDataLoader::Next() {
	if (data) {
		delete[] data;
	}
	if (!input.is_open()) {
		pageSize = 0;
		return false;
	}
	page++;
	int varSize = dim * GetNumVars() + 1;
	int dataPageSize = bufferSize * varSize;
	data = new real[dataPageSize];
	if (RECORDHEADER) {
		assert(sizeof (int) == 4);
		int i = 0;
		int dataOffset = 0;
		while (i < bufferSize) {
			int recordSize;
			input.read((char*) &recordSize, 4);
			int numBytesRead = input.gcount();
			if (input.eof()) {
				input.close();
				break;
			}
			assert(numBytesRead == 4);
			assert(recordSize == 4);
			input.read((char*) (data + dataOffset++), recordSize);
			numBytesRead = input.gcount();
			assert(numBytesRead == 4);
			input.read((char*) &recordSize, 4);
			numBytesRead = input.gcount();
			assert(numBytesRead == 4);
			input.read((char*) &recordSize, 4);
			numBytesRead = input.gcount();
			assert(numBytesRead == 4);
			assert(recordSize == (sizeof (real) * (dim * totalNumVars)));
			//input.read((char*) (data + dataOffset + 1), recordSize);
			//numBytesRead = input.gcount();
			//assert(numBytesRead == recordSize);
			int lastVarIndex = 0;
			for (int varIndex : GetVarIndices()) {
				assert(varIndex >= lastVarIndex);
				if (varIndex - lastVarIndex != 0) {
					input.seekg((varIndex - lastVarIndex) * dim * sizeof (real), ios::cur);
				}
				input.read((char*) (data + dataOffset), dim * sizeof (real));
				numBytesRead = input.gcount();
				assert(numBytesRead == dim * sizeof (real));
				lastVarIndex = varIndex + 1;
				dataOffset += dim;
			}
			assert(recordSize >= lastVarIndex * dim * sizeof (real));
			if (recordSize - lastVarIndex * dim * sizeof (real) != 0) {
				input.seekg(recordSize - lastVarIndex * dim * sizeof (real), ios::cur);
			}
			input.read((char*) &recordSize, 4);
			numBytesRead = input.gcount();
			assert(numBytesRead == 4);
			//dataOffset += varSize;
			i++;
		}
		pageSize = i;
	} else {
		input.read((char*) data, (sizeof (real)) * dataPageSize);
		int numBytesRead = input.gcount();
		if (numBytesRead < (sizeof (real)) * dataPageSize) {
			assert(numBytesRead % ((sizeof (real)) * varSize) == 0);
			pageSize = numBytesRead / ((sizeof (real)) * varSize);
			input.close();
		} else {
			pageSize = bufferSize;
		}
	}
	return pageSize > 0;
}

DataLoader* BinaryDataLoader::Clone() const {
	BinaryDataLoader* dl = new BinaryDataLoader(*this);
	if (dl->GetPageSize() == 0) {
		delete dl;
		return nullptr;
	}
	return dl;
}
