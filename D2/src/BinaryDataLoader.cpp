#include "BinaryDataLoader.h"
#include <iostream>
using namespace std;

BinaryDataLoader::BinaryDataLoader(const string& fileName, unsigned bufferSize, unsigned dim, unsigned totalNumVars, const vector<unsigned>& varIndices) :
		DataLoader(fileName, bufferSize, ios::in | ios::binary, dim, totalNumVars, varIndices) {
}

BinaryDataLoader::BinaryDataLoader(const BinaryDataLoader& dataLoader) : DataLoader(dataLoader) {
	if (dataLoader.page > 1) {
		if (RECORDHEADER) {
			input.seekg((dataLoader.page - 1) * bufferSize * ((dim * totalNumVars + 1) * sizeof (real) + 16), ios::cur);
		} else {
			input.seekg((dataLoader.page - 1) * bufferSize * (dim * totalNumVars + 1) * sizeof (real), ios::cur);
		}
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
	unsigned varSize = dim * totalNumVars + 1;
	unsigned dataPageSize = bufferSize * varSize;
	data = new real[dataPageSize];
	if (RECORDHEADER) {
		assert(sizeof (unsigned) == 4);
		unsigned i = 0;
		unsigned dataOffset = 0;
		while (i < bufferSize) {
			if (input.eof()) {
				input.close();
				break;
			}
			unsigned recordSize;
			input.read((char*) &recordSize, 4);
			unsigned numBytesRead = input.gcount();
			assert(numBytesRead == 4);
			assert(recordSize == 4);
			input.read((char*) (data + dataOffset), recordSize);
			numBytesRead = input.gcount();
			assert(numBytesRead == 4);
			input.read((char*) &recordSize, 4);
			numBytesRead = input.gcount();
			assert(numBytesRead == 4);
			input.read((char*) &recordSize, 4);
			numBytesRead = input.gcount();
			assert(numBytesRead == 4);
			assert(recordSize == (sizeof (real) * (varSize - 1)));
			input.read((char*) (data + dataOffset + 1), recordSize);
			numBytesRead = input.gcount();
			assert(numBytesRead == recordSize);
			input.read((char*) &recordSize, 4);
			numBytesRead = input.gcount();
			assert(numBytesRead == 4);
			dataOffset += varSize;
			i++;
		}
		pageSize = i;
	} else {
		input.read((char*) data, (sizeof (real)) * dataPageSize);
		unsigned numBytesRead = input.gcount();
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

unique_ptr<DataLoader> BinaryDataLoader::Clone() const {
	BinaryDataLoader* clone = new BinaryDataLoader(*this);
	return unique_ptr<DataLoader>(clone);
}
