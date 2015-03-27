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

class VertexShader : public Shader {
public:
	VertexShader(const std::string& rShaderScript);
	virtual ~VertexShader();
};

} /* namespace engine3d */

#endif /* VERTEXSHADER_H_ */
