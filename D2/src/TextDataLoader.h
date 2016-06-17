/*
 * TextDataLoader.h
 *
 *  Created on: Aug 7, 2015
 *      Author: nigul
 */

#ifndef TEXTDATALOADER_H_
#define TEXTDATALOADER_H_

#include "DataLoader.h"

class TextDataLoader: public DataLoader {
public:
	TextDataLoader(const string& fileName, int bufferSize,
			const vector<int>& dims,
			const vector<vector<pair<int, int>>>& regions,
			int totalNumVars, const vector<int>& varIndices);
	TextDataLoader(const TextDataLoader& dataLoader);
	virtual ~TextDataLoader();

	virtual bool Next();
	virtual DataLoader* Clone() const;

};


#endif /* TEXTDATALOADER_H_ */
