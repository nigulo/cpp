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
#include <memory>
#include <cassert>

using namespace std;

class DataLoader {
public:
	DataLoader(const string& fileName, unsigned bufferSize = 0, ios::openmode mode = ios::in, unsigned dim = 1, unsigned totalNumVars = 1, const vector<unsigned>& varIndices = {0});
	DataLoader(const DataLoader& dataLoader);
	virtual ~DataLoader();

	virtual bool Next() = 0;
	virtual unique_ptr<DataLoader> Clone() const = 0;

	void Reset();

	const string& GetFileName() const {
		return fileName;
	}

	const double GetX(unsigned i) const {
		assert(i < pageSize);
		return data[i * (dim * totalNumVars + 1)];
	}

	const double* GetY(unsigned i) const {
		assert(i < pageSize);
		return &data[i * (dim * totalNumVars + 1) + 1];
	}

	int GetPage() const {
		return page;
	}

	unsigned GetDim() const {
		return dim;
	}

	unsigned GetYSize() const {
		return dim * totalNumVars;
	}

	unsigned GetPageSize() const {
		return pageSize;
	}

	const vector<unsigned>& GetVarIndices() const {
		return varIndices;
	}

protected:
	const string fileName;
	const unsigned bufferSize;
	//vector<double> x;
	//vector<double*> y;
	const ios::openmode mode;
	const unsigned dim;
	const unsigned totalNumVars;
	const vector<unsigned> varIndices;
	ifstream input;
	int page;
	double* data;
	unsigned pageSize;

};


#endif /* SRC_DATALOADER_H_ */
