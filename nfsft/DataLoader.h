/*
 * DataLoader.h
 *
 *  Created on: Aug 3, 2015
 *      Author: nigul
 */

#ifndef SRC_DATALOADER_H_
#define SRC_DATALOADER_H_

#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <utility>
#include <random>
#include <iostream>

using namespace std;

typedef float real;

class DataLoader {
public:
	DataLoader(const string& fileName, int bufferSize, ios::openmode mode,
			const vector<int>& dims,
			const vector<vector<pair<int, int>>>& regions,
			int totalNumVars,
			const vector<int>& varIndices);
	DataLoader(const DataLoader& dataLoader);
	virtual ~DataLoader();

	virtual bool Next() = 0;
	virtual DataLoader* Clone() const = 0;

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

	real* GetYToModify(int i) {
		if (i >= pageSize) {
			cout << "pageSize, i" << pageSize << ", " << i << endl;
		}
		assert(i < pageSize);
		return &data[i * (dim * GetNumVars() + 1) + 1];
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

protected:
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


#endif /* SRC_DATALOADER_H_ */
