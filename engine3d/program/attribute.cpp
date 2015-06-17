/*
 * globject.cpp
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#include "attribute.h"
#include "program.h"
#include "base/object.h"

using namespace engine3d;

Attribute::Attribute(Program& rProgram, const string& rName) {
	mId = glGetAttribLocation(rProgram.GetId(), rName.data());
	base::Object::Dbg(to_string(mId) + " = glGetAttribLocation(" + to_string(rProgram.GetId()) + ", " + rName + ")");
}

Attribute::~Attribute() {
}
