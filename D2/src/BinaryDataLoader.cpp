#include "BinaryDataLoader.h"

BinaryDataLoader::BinaryDataLoader(const string& fileName, unsigned bufferSize, unsigned dim, unsigned totalNumVars, const vector<unsigned>& varIndices) :
		DataLoader(fileName, bufferSize, ios::in | ios::binary, dim, totalNumVars, varIndices) {
}

BinaryDataLoader::BinaryDataLoader(const BinaryDataLoader& dataLoader) : DataLoader(dataLoader) {
	for (int i = 0; i < dataLoader.page; i++) {
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
	x.clear();
	for (auto& yVals : y) {
		delete[] yVals;
	}
	y.clear();
	unsigned i = 0;
	while (bufferSize == 0 || i < bufferSize) {
		if (!input.eof()) {
			double xVal;
			input >> xVal;
			x.push_back(xVal);
			double* yVals = new double[dim * varIndices.size()];
			for (unsigned j = 0; j < dim; j++) {
				//double* yVals = new double[varIndices.size()];
				unsigned lastVarIndex = 0;
				for (unsigned k = 0; k < varIndices.size(); k++) {
					unsigned seek = varIndices[k] - lastVarIndex;
					lastVarIndex = varIndices[k] + 1;
					if (seek > 0) {
						input.seekg(seek * sizeof (double), ios::cur);
					}
					input >> yVals[j * dim + k];
				}
				unsigned seek = totalNumVars - lastVarIndex;
				if (seek > 0) {
					input.seekg(seek * sizeof (double), ios::cur);
				}
			}
			y.push_back(yVals);
			i++;
		} else {
			input.close();
			break;
		}
	}
	return !x.empty();
}

unique_ptr<DataLoader> BinaryDataLoader::Clone() const {
	return unique_ptr<DataLoader>(new BinaryDataLoader(*this));
}
