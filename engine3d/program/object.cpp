/*
 * globject.cpp
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#include "object.h"

using namespace engine3d;

Object::Object(const Program& rProgram, const std::string& rName) {
	mId = glGetUniformLocation(rProgram.GetId(), rName.data());
}

Object::~Object() {
}
