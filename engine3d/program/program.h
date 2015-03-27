/*
 * program.h
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#ifndef PROGRAM_H_
#define PROGRAM_H_

#include "vertexshader.h"
#include "fragmentshader.h"
#include <string>

namespace engine3d {

class Program {
public:
	Program(const std::string& rVertexShaderScript, const std::string& rFragmentShaderScript);

	virtual ~Program();

	GLuint GetId() const {
		return mId;
	}

private:
	VertexShader mVertexShader;
	FragmentShader mFragmentShader;
	GLuint mId;
};

} /* namespace engine3d */

#endif /* PROGRAM_H_ */
