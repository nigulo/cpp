/*
 * fragmentshader.h
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#ifndef FRAGMENTSHADER_H_
#define FRAGMENTSHADER_H_

#include "shader.h"
#include <string>

namespace engine3d {

class Program;
class FragmentShader: public Shader {
	friend class Program;
protected:
	FragmentShader(const std::string& rShaderScript);
	virtual ~FragmentShader();
};

} /* namespace engine3d */

#endif /* FRAGMENTSHADER_H_ */
