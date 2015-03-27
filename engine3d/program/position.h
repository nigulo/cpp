/*
 * position.h
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#ifndef POSITION_H_
#define POSITION_H_

#include "object.h"

namespace engine3d {

class Position: public Object {
public:
	Position(const Program& rProgram, const std::string& rName = "position");
	virtual ~Position();
};

} /* namespace engine3d */

#endif /* POSITION_H_ */
