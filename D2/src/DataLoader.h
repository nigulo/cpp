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

using namespace std;

class DataLoader {
public:
	DataLoader(const string& fileName, unsigned startCol = 1, unsigned dim = 1, unsigned bufferSize = 0);
	DataLoader(const DataLoader& dataLoader);
	virtual ~DataLoader();

	bool Next();

	void Reset();

	const string& GetFileName() const {
		return fileName;
	}

	unsigned GetStartCol() const {
		return startCol;
	}

	unsigned GetDim() const {
		return dim;
	}

	const vector<double>& GetX() const {
		return x;
	}

	const vector<double*>& GetY() const {
		return y;
	}

	int GetPage() const {
		return page;
	}

private:
	const string fileName;
	ifstream input;
	const unsigned startCol;
	const unsigned dim;
	const unsigned bufferSize;
	vector<double> x;
	vector<double*> y;

	int page;

};


#endif /* SRC_DATALOADER_H_ */
