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
	BinaryDataLoader(const string& fileName, unsigned bufferSize = 0, unsigned dim = 1, unsigned totalNumVars = 1, const vector<unsigned>& varIndices = {0});
	BinaryDataLoader(const BinaryDataLoader& dataLoader);
	virtual ~BinaryDataLoader();

	virtual bool Next();
	virtual unique_ptr<DataLoader> Clone() const;

};

#endif /* SRC_BINARYDATALOADER_H_ */
