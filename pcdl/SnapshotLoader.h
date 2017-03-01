/*
 * SnapshotLoader.h
 *
 *  Created on: Jan 26, 2017
 *      Author: nigul
 */

#ifndef SNAPSHOTLOADER_H_
#define SNAPSHOTLOADER_H_

#include "DataLoader.h"

namespace pcdl {

class SnapshotLoader : public DataLoader {
public:
	SnapshotLoader(const map<string, string>& params, std::function<void(const string&)> logFunc = defaultLogFunc);
	virtual ~SnapshotLoader();
public:
	void load(std::function<void(int /*time*/, int /*x*/, int /*y*/, int /*z*/, double /*val*/)> f);
};

}

#endif /* SNAPSHOTLOADER_H_ */
