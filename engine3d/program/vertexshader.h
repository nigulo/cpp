/*
 * vertexshader.h
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#ifndef VERTEXSHADER_H_
#define VERTEXSHADER_H_

#include "shader.h"
#include <string>

namespace engine3d {

class Program;
class VertexShader : public Shader {
	friend class Program;
protected:
	VertexShader(const std::string& rShaderScript);
	virtual ~VertexShader();
};

} /* namespace engine3d */

#endif /* VERTEXSHADER_H_ */
