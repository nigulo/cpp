/*
 * fragmentshader.cpp
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#include "fragmentshader.h"
#include <GL/gl.h>

using namespace engine3d;

FragmentShader::FragmentShader(const std::string& rShaderScript) : Shader(GL_FRAGMENT_SHADER, rShaderScript) {
	// TODO Auto-generated constructor stub

}

FragmentShader::~FragmentShader() {
	// TODO Auto-generated destructor stub
}
