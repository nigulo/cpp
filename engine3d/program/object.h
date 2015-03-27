/*
 * globject.h
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#ifndef OBJECT_H_
#define OBJECT_H_

#include "program.h"
#include <GL/glew.h>
#include <string>

namespace engine3d {

class Object {
public:
	Object(const Program& rProgram, const std::string& rName);
	virtual ~Object();

	GLuint GetId() const {
		return mId;
	}
private:
	GLuint mId;
};

} /* namespace engine3d */

#endif /* OBJECT_H_ */
