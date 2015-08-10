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
	TextDataLoader(const string& fileName, unsigned bufferSize = 0, unsigned dim = 1, unsigned totalNumVars = 1, const vector<unsigned>& varIndices = {0});
	TextDataLoader(const TextDataLoader& dataLoader);
	virtual ~TextDataLoader();

	virtual bool Next();
	virtual unique_ptr<DataLoader> Clone() const;

};


#endif /* TEXTDATALOADER_H_ */
