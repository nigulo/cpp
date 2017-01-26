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
	VideoSliceLoader(const map<string, string>& params);
	virtual ~VideoSliceLoader();
public:
	void load(std::function<void(int /*time*/, int /*x*/, int /*y*/, int /*val*/)> f);
};
}
#endif /* VIDEOSLICELOADER_H_ */
