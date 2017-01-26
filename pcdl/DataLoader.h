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

class DataLoader {
public:
	DataLoader(const map<string, string>& params);
	virtual ~DataLoader();

	virtual void load(std::function<void(int /*time*/, int /*x*/, int /*y*/, int /*val*/)> f) = 0;
protected:
	map<string, string> params;
	vector<int> dims;
	int thetaIndex;
	int phiIndex;
	vector<int> varIndices;
	int thetaDownSample;
	int phiDownSample;

	vector<vector<pair<int, int>>> regions;
	double polarGap;
	string filePath;
};

#endif /* DATALOADER_H_ */
