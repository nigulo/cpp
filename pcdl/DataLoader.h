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

using namespace std;

namespace pcdl {
class DataLoader {
public:
	DataLoader(const map<string, string>& params);
	virtual ~DataLoader();

	const vector<int>& getDimsDownSampled() const {
		return dimsDownSampled;
	}

protected:
	map<string, string> params;
	vector<int> dims;
	vector<int> dimsDownSampled;
	int yIndex;
	int zIndex;
	vector<int> varIndices;
	int yDownSample;
	int zDownSample;

	vector<vector<pair<int, int>>> regions;
	string filePath;
};
}
#endif /* DATALOADER_H_ */
