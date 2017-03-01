/*
 * DataLoader.h
 *
 *  Created on: Jan 26, 2017
 *      Author: nigul
 */

#ifndef DATALOADER_H_
#define DATALOADER_H_

#include <map>
#include <vector>
#include <functional>
#include "BinaryDataLoader.h"

using namespace std;

void defaultLogFunc(const string& str);

namespace pcdl {
class DataLoader {
public:
	DataLoader(const map<string, string>& params, std::function<void(const string&)> logFunc = defaultLogFunc);
	virtual ~DataLoader();

	const vector<int>& getDimsDownSampled() const {
		return dimsDownSampled;
	}

protected:
	map<string, string> params;
	std::function<void(const string&)> logFunc;
	vector<int> dims;
	vector<int> dimsDownSampled;
	int xIndex;
	int yIndex;
	int zIndex;
	vector<int> varIndices;
	int xDownSample;
	int yDownSample;
	int zDownSample;

	vector<vector<pair<int, int>>> regions;
	string filePath;
	Precision prec;
	int bufferSize;
};
}
#endif /* DATALOADER_H_ */
