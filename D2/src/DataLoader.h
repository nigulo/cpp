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
#include <tuple>

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
		return data[i * (dim * GetNumVars() + 1)];
	}

	/*
	real GetRandomX(default_random_engine& e1) const {
		uniform_int_distribution<int> uniform_dist(0, pageSize - 1);
		return data[uniform_dist(e1) * (dim * GetNumVars() + 1)];
	}
	*/

	const real* GetY(int i) const {
		if (i >= pageSize) {
			cout << "pageSize, i" << pageSize << ", " << i << endl;
		}
		assert(i < pageSize);
		return &data[i * (dim * GetNumVars() + 1) + 1];
	}

	real* GetYToModify(int i) {
		if (i >= pageSize) {
			cout << "pageSize, i" << pageSize << ", " << i << endl;
		}
		assert(i < pageSize);
		return &data[i * (dim * GetNumVars() + 1) + 1];
	}

	/*
	// Not tested
	void SetY(int i, const real* y) const {
		if (i >= pageSize) {
			cout << "pageSize, i" << pageSize << ", " << i << endl;
		}
		assert(i < pageSize);
		for (int j = 0; j < GetNumVars() * GetDim(); j++) {
			data[i * (dim * GetNumVars() + 1) + 1 + j] = y[j];
		}

	}
	*/

	/*
	int GetAbsoluteIndex(const vector<int>& indices) const {
		assert(indices.size() == dims.size());
		assert(mins.empty() || mins[0] <= indices[0]);
		assert(maxs.empty() || maxs[0] >= indices[0]);
		int i = indices[0];
		int d = 1;
		for (int j = 1; j < indices.size(); j++) {
			assert(mins.size() <= j || mins[j] <= indices[j]);
			assert(maxs.size() <= j || maxs[j] >= indices[j]);
			d *= dims[j - 1];
			i += d * indices[j];
		}
		return i;
	}
	*/

	int GetPage() const {
		return page;
	}

	const vector<int>& GetDims() const {
		return dims;
	}

	int GetDim() const {
		return dim;
	}

	/*
	const vector<int>& GetMins() const {
		return mins;
	}

	const vector<int>& GetMaxs() const {
		return maxs;
	}
	*/

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

	tuple<real /*startTime*/, real /*endTime*/, int /*length*/> CalcRangeAndLength() const;

	/*
	void ShufflePage(default_random_engine& e1) {
		uniform_int_distribution<int> uniform_dist(0, pageSize - 1);
		for (int i = 0; i < pageSize; i++) {
			auto oldPos = i * (dim * GetNumVars() + 1);
			auto val = data[oldPos];
			auto newIndex = uniform_dist(e1);
			auto newPos = newIndex * (dim * GetNumVars() + 1);
			data[oldPos] = data[newPos];
			data[newPos] = val;
		}
	}
	*/

private:
	bool InRegion(int i) const {
		if (regions.empty()) {
			return false;
		}
		for (vector<pair<int, int>> region : regions) {
			bool inRegion = true;
			for (int j = 0; j < dims.size(); j++) {
				int d = i % dims[j];
				if (region.size() > j && (d < get<0>(region[j]) || d > get<1>(region[j]))) {
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
	const string fileName;
	const int bufferSize;
	const ios::openmode mode;
	const vector<int> dims;
	const vector<vector<pair<int /*min*/, int /*max*/>>> regions;
	const int totalNumVars;
	const vector<int> varIndices;
	ifstream input;
	int page;
	real* data;
	int pageSize;
	int dim;
	bool* inRegion;

};


#endif /* SRC_DATALOADER_H_ */
