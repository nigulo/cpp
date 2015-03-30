/*
 * globject.cpp
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#include "attribute.h"

using namespace engine3d;

Attribute::Attribute(const Program& rProgram, const std::string& rName) {
	mId = glGetUniformLocation(rProgram.GetId(), rName.data());
}

Attribute::~Attribute() {
}
