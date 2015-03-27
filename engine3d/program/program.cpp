/*
 * program.cpp
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#include "program.h"
#include <GL/glew.h>
#include <GL/gl.h>
#include "glutils.h"

using namespace engine3d;
using namespace std;

Program::Program(const std::string& rVertexShaderScript,
			const std::string& rFragmentShaderScript) :
		mVertexShader(rVertexShaderScript),
		mFragmentShader(rFragmentShaderScript) {

    mId = glCreateProgram();

    glAttachShader(mId, mVertexShader.GetId());
    glAttachShader(mId, mFragmentShader.GetId());
    glLinkProgram(mId);

    GLint program_ok;
    glGetProgramiv(mId, GL_LINK_STATUS, &program_ok);
    if (!program_ok) {
        const string info_log = GLUtils::InfoLog(mId, glGetProgramiv, glGetProgramInfoLog);
        glDeleteProgram(mId);
        throw (info_log);
    }

}

Program::~Program() {
}
