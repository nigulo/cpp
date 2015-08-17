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

using namespace std;

typedef float real;

class DataLoader {
public:
	DataLoader(const string& fileName, unsigned bufferSize, ios::openmode mode,
			const vector<unsigned>& dims,
			const vector<unsigned>& mins,
			const vector<unsigned>& maxs,
			unsigned totalNumVars,
			const vector<unsigned>& varIndices);
	DataLoader(const DataLoader& dataLoader);
	virtual ~DataLoader();

	virtual bool Next() = 0;
	virtual DataLoader* Clone() const = 0;

	void Reset();

	const string& GetFileName() const {
		return fileName;
	}

	real GetX(unsigned i) const {
		assert(i < pageSize);
		return data[i * (dim * totalNumVars + 1)];
	}

	const real* GetY(unsigned i) const {
		assert(i < pageSize);
		return &data[i * (dim * totalNumVars + 1) + 1];
	}

	unsigned GetAbsoluteIndex(const vector<unsigned>& indices) const {
		assert(indices.size() == dims.size());
		assert(mins.empty() || mins[0] <= indices[0]);
		assert(maxs.empty() || maxs[0] >= indices[0]);
		unsigned i = indices[0];
		unsigned d = 1;
		for (unsigned j = 1; j < indices.size(); j++) {
			assert(mins.size() <= j || mins[j] <= indices[j]);
			assert(maxs.size() <= j || maxs[j] >= indices[j]);
			d *= dims[j - 1];
			i += d * indices[j];
		}
		return i;
	}

	bool Skip(unsigned i, const vector<bool> useMin, const vector<bool> useMax) const {
		for (unsigned j = 0; j < dims.size(); j++) {
			unsigned d = i % dims[j];
			if ((useMin[j] && mins.size() > j && d < mins[j]) || (useMax[j] && maxs.size() > j && d > maxs[j])) {
				return true;
			}
			i -= d;
			i /= dims[j];
		}
		return false;
	}

	int GetPage() const {
		return page;
	}

	const vector<unsigned>& GetDims() const {
		return dims;
	}

	unsigned GetDim() const {
		return dim;
	}

	const vector<unsigned>& GetMins() const {
		return mins;
	}

	const vector<unsigned>& GetMaxs() const {
		return maxs;
	}

	unsigned GetPageSize() const {
		return pageSize;
	}

	unsigned GetNumVars() const {
		return totalNumVars;
	}

	const vector<unsigned>& GetVarIndices() const {
		return varIndices;
	}

protected:
	const string fileName;
	const unsigned bufferSize;
	const ios::openmode mode;
	const vector<unsigned> dims;
	const vector<unsigned> mins;
	const vector<unsigned> maxs;
	const unsigned totalNumVars;
	const vector<unsigned> varIndices;
	ifstream input;
	int page;
	real* data;
	unsigned pageSize;
	unsigned dim;

};


#endif /* SRC_DATALOADER_H_ */
