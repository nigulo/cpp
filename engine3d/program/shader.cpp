/*
 * shader.cpp
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#include "shader.h"
#include "utils.h"
#include <GL/glew.h>
#include <GL/gl.h>

using namespace engine3d;
using namespace std;

Shader::Shader(GLenum type, const string& rShaderScript) {
    const GLchar* source = rShaderScript.data();

    if (rShaderScript.length() == 0) {
		throw ("Empty shader script");
    }

    mId = glCreateShader(type);
    GLint length;
    glShaderSource(mId, 1, (const GLchar**)&source, &length);
    glCompileShader(mId);

    GLint shader_ok;
    glGetShaderiv(mId, GL_COMPILE_STATUS, &shader_ok);
    if (!shader_ok) {
        string info_log = Utils::InfoLog(mId, glGetShaderiv, glGetShaderInfoLog);
        glDeleteShader(mId);
		throw (info_log);
    }
}

Shader::~Shader() {
	// TODO Auto-generated destructor stub
}
