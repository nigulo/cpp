#include "BinaryDataLoader.h"
#include <iostream>
using namespace std;

BinaryDataLoader::BinaryDataLoader(const string& fileName, unsigned bufferSize, unsigned dim, unsigned totalNumVars, const vector<unsigned>& varIndices) :
		DataLoader(fileName, bufferSize, ios::in | ios::binary, dim, totalNumVars, varIndices) {
}

BinaryDataLoader::BinaryDataLoader(const BinaryDataLoader& dataLoader) : DataLoader(dataLoader) {
	if (dataLoader.page > 1) {
		input.seekg((dataLoader.page - 1) * bufferSize * (totalNumVars * dim) * sizeof (double), ios::cur);
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
	data = new double[bufferSize * (dim * totalNumVars + 1)];
	input.read((char*) data, (sizeof (double)) * bufferSize * (dim * totalNumVars + 1));
	unsigned numBytesRead = input.gcount();
	if (numBytesRead < sizeof data) {
		assert(numBytesRead % ((sizeof (double)) * (dim * totalNumVars + 1)) == 0);
		pageSize = numBytesRead / ((sizeof (double)) * (dim * totalNumVars + 1));
		input.close();
	} else {
		pageSize = bufferSize;
	}
	return pageSize > 0;
	/*
	unsigned i = 0;
	while (bufferSize == 0 || i < bufferSize) {
		if (!input.eof()) {
			double xVal;
			input >> xVal;
			x.push_back(xVal);
			double* yVals = new double[dim * totalNumVars];
			for (unsigned j = 0; j < dim; j++) {
				//double* yVals = new double[varIndices.size()];
				unsigned lastVarIndex = 0;
				for (unsigned k = 0; k < varIndices.size(); k++) {
					unsigned seek = varIndices[k] - lastVarIndex;
					lastVarIndex = varIndices[k] + 1;
					if (seek > 0) {
						input.seekg(seek * sizeof (double), ios::cur);
					}
					input.read((char*) yVals, sizeof yVals);
					//input >> yVals[j * dim + k];
				}
				unsigned seek = totalNumVars - lastVarIndex;
				if (seek > 0) {
					input.seekg(seek * sizeof (double), ios::cur);
				}
			}
			y.push_back(yVals);
			//cout << xVal;
			//for (unsigned iy = 0 ; iy < GetYSize(); iy++) {
			//	cout << " " << yVals[iy];
			//}
			//cout << endl;
			i++;
		} else {
			input.close();
			break;
		}
	}
	return !x.empty();
	*/
}

unique_ptr<DataLoader> BinaryDataLoader::Clone() const {
	return unique_ptr<DataLoader>(new BinaryDataLoader(*this));
}
