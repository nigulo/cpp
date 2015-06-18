/*
 * vertexshader.cpp
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#include "vertexshader.h"
#include <GL/glew.h>
using namespace engine3d;

VertexShader::VertexShader(const std::string& rShaderScript) : Shader(GL_VERTEX_SHADER, rShaderScript) {

}

engine3d::VertexShader::~VertexShader() {
}
