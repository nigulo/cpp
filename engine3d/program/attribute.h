#ifndef PROGRAM_ATTRIBUTE_H_
#define PROGRAM_ATTRIBUTE_H_

#include <GL/glew.h>
#include <string>

using namespace std;
namespace engine3d {

class Program;
class Attribute {
	friend class Program;
private:
	Attribute(Program& rProgram, const string& rName);
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
