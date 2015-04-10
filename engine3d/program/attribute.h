/*
 * globject.h
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#ifndef PROGRAM_ATTRIBUTE_H_
#define PROGRAM_ATTRIBUTE_H_

#include <GL/glew.h>
#include <string>

namespace engine3d {

class Program;
class Attribute {
	friend class Program;
private:
	Attribute(Program& rProgram, const std::string& rName);
	virtual ~Attribute();
public:
	GLint GetId() const {
		return mId;
	}
private:
	GLint mId;
};

} /* namespace engine3d */

#endif /* PROGRAM_ATTRIBUTE_H_ */
