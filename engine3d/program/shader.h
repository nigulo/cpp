/*
 * shader.h
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#ifndef SHADER_H_
#define SHADER_H_

#include <GL/glew.h>
#include <string>

namespace engine3d {

class Program;
class Shader {
	friend class Program;
protected:
	Shader(GLenum type, const std::string& rShaderScript);
	virtual ~Shader();
public:
	GLuint GetId() const {
		return mId;
	}
private:
	GLuint mId;
};

} /* namespace engine3d */

#endif /* SHADER_H_ */
