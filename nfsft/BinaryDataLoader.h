/*
 * BinaryDataLoader.h
 *
 *  Created on: Aug 7, 2015
 *      Author: nigul
 */

#ifndef SRC_BINARYDATALOADER_H_
#define SRC_BINARYDATALOADER_H_

#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <utility>
#include <random>
#include <iostream>
#include <tuple>
#include <iostream>

using namespace std;

#define RECORDHEADER true
#define TYPE_SNAPSHOT 0
#define TYPE_VIDEO 1

enum Precision {
	SinglePrecision,
	DoublePrecision
};

class BinaryDataLoader {
public:
	BinaryDataLoader(const string& fileName, int bufferSize,
			const vector<int>& dims,
			const vector<vector<pair<int, int>>>& regions,
			int totalNumVars, const vector<int>& varIndices, int type, Precision prec);
	BinaryDataLoader(const BinaryDataLoader& dataLoader);
	virtual ~BinaryDataLoader();

	bool Next();

	BinaryDataLoader* Clone() const;

	const string& GetFileName() const {
		return fileName;
	}

	double GetX(int i) const {
		assert(i < pageSize);
		if (sizeOfReal == sizeof (float)) {
			return ((float*) data)[((size_t) i) * (dim * GetNumVars() + 1) + dim * GetNumVars()];
		} else {
			return ((double*) data)[((size_t) i) * (dim * GetNumVars() + 1) + dim * GetNumVars()];
		}
	}


	const char* GetY(int i) const {
		assert(i < pageSize);
		if (sizeOfReal == sizeof (float)) {
			return (const char*) &((float*) data)[((size_t) i) * (dim * GetNumVars() + 1)];
		} else {
			return (const char*) &((double*) data)[((size_t) i) * (dim * GetNumVars() + 1)];
		}
	}

	int GetPage() const {
		return page;
	}

	const vector<int>& GetDims() const {
		return dims;
	}

	int GetDim() const {
		return dim;
	}

	int GetPageSize() const {
		return pageSize;
	}

	int GetTotalNumVars() const {
		return totalNumVars;
	}

	int GetNumVars() const {
		if (varIndices.size() > 0) {
			return varIndices.size();
		} else {
			return totalNumVars;
		}
	}

	const vector<int>& GetVarIndices() const {
		return varIndices;
	}

	bool IsInRegion(int i) const {
		return inRegion[i];
	}

private:
	bool InRegion(int i) const {
		if (regions.empty()) {
			return false;
		}
		for (vector<pair<int, int>> region : regions) {
			bool inRegion = true;
			for (int j = 0; j < (int) dims.size(); j++) {
				int d = i % dims[j];
				if ((int) region.size() > j && (d < get<0>(region[j]) || d > get<1>(region[j]))) {
					inRegion = false;
					break;
				}
				i -= d;
				i /= dims[j];
			}
			if (inRegion) {
				return true;
			}
		}
		return false;
	}

private:
	const int type;
	string fileName;
	int bufferSize;
	ios::openmode mode;
	vector<int> dims;
	vector<vector<pair<int /*min*/, int /*max*/>>> regions;
	int totalNumVars;
	vector<int> varIndices;
	ifstream input;
	int page;
	char* data;
	int pageSize;
	int dim;
	bool* inRegion;
	unsigned sizeOfReal;

};

#endif /* SRC_BINARYDATALOADER_H_ */
