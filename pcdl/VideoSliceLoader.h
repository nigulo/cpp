/*
 * VideoSliceLoader.h
 *
 *  Created on: Jan 26, 2017
 *      Author: nigul
 */

#ifndef VIDEOSLICELOADER_H_
#define VIDEOSLICELOADER_H_

#include "DataLoader.h"

namespace pcdl {

class VideoSliceLoader : public DataLoader {
public:
	VideoSliceLoader(const map<string, string>& params, std::function<void(const string&)> logFunc = defaultLogFunc);
	virtual ~VideoSliceLoader();
public:
	void load(std::function<void(int /*y*/, int /*z*/, double /*val*/)> f1, std::function<void(int /*time*/)> f2);
};
}
#endif /* VIDEOSLICELOADER_H_ */
