/*
 * BinaryDataLoader.h
 *
 *  Created on: Aug 7, 2015
 *      Author: nigul
 */

#ifndef SRC_BINARYDATALOADER_H_
#define SRC_BINARYDATALOADER_H_

#include "DataLoader.h"

#define RECORDHEADER true

class BinaryDataLoader: public DataLoader {
public:
	BinaryDataLoader(const string& fileName, int bufferSize,
			const vector<int>& dims,
			const vector<vector<pair<int, int>>>& regions,
			int totalNumVars, const vector<int>& varIndices);
	BinaryDataLoader(const BinaryDataLoader& dataLoader);
	virtual ~BinaryDataLoader();

	virtual bool Next();
	virtual DataLoader* Clone() const;

};

#endif /* SRC_BINARYDATALOADER_H_ */
