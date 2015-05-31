#ifndef PROGRAM_UNIFORM_H_
#define PROGRAM_UNIFORM_H_

#include <GL/glew.h>
#include <string>

namespace engine3d {

class Program;
class Uniform {
	friend class Program;
private:
	Uniform(Program& rProgram, const std::string& rName);
	virtual ~Uniform();
public:
	GLint GetId() const {
		return mId;
	}
private:
	GLint mId;
};

} /* namespace engine3d */

#endif /* PROGRAM_UNIFORM_H_ */
