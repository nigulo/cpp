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

using namespace std;

typedef float real;

#define RECORDHEADER true
#define TYPE_SNAPSHOT 0
#define TYPE_VIDEO 1

class BinaryDataLoader {
public:
	BinaryDataLoader(const string& fileName, int bufferSize,
			const vector<int>& dims,
			const vector<vector<pair<int, int>>>& regions,
			int totalNumVars, const vector<int>& varIndices, int type);
	BinaryDataLoader(const BinaryDataLoader& dataLoader);
	virtual ~BinaryDataLoader();

	bool Next();
	BinaryDataLoader* Clone() const;

	void Reset();

	const string& GetFileName() const {
		return fileName;
	}

	real GetX(int i) const {
		assert(i < pageSize);
		return data[i * (dim * GetNumVars() + 1) + dim * GetNumVars()];
	}

	const real* GetY(int i) const {
		assert(i < pageSize);
		return &data[i * (dim * GetNumVars() + 1)];
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
	real* data;
	int pageSize;
	int dim;
	bool* inRegion;

};

#endif /* SRC_BINARYDATALOADER_H_ */
