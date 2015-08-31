#include "BinaryDataLoader.h"
#include <iostream>
using namespace std;

BinaryDataLoader::BinaryDataLoader(const string& fileName, unsigned bufferSize,
		const vector<unsigned>& dims,
		const vector<vector<pair<unsigned, unsigned>>>& regions,
		unsigned totalNumVars, const vector<unsigned>& varIndices) :
				DataLoader(fileName, bufferSize, ios::in | ios::binary, dims, regions, totalNumVars, varIndices) {
}

// Creates new DataLoader with the current page set to the next page of input DataLoader
BinaryDataLoader::BinaryDataLoader(const BinaryDataLoader& dataLoader) : DataLoader(dataLoader) {
	if (dataLoader.page > 0) {
		if (RECORDHEADER) {
			input.seekg(dataLoader.page * bufferSize * ((dim * totalNumVars + 1) * sizeof (real) + 16), ios::cur);
		} else {
			input.seekg(dataLoader.page * bufferSize * (dim * totalNumVars + 1) * sizeof (real), ios::cur);
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
			unsigned recordSize;
			input.read((char*) &recordSize, 4);
			unsigned numBytesRead = input.gcount();
#ifdef TEST
			if (page > 3 || input.eof()) {
				input.close();
				break;
			}
#else
			if (input.eof()) {
				input.close();
				break;
			}
#endif
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

DataLoader* BinaryDataLoader::Clone() const {
	BinaryDataLoader* dl = new BinaryDataLoader(*this);
	if (dl->GetPageSize() == 0) {
		delete dl;
		return nullptr;
	}
	return dl;
}
