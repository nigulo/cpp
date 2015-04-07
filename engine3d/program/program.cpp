/*
 * program.cpp
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#include "program.h"
#include <GL/glew.h>
#include <GL/gl.h>

#include "utils.h"

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
        const string info_log = Utils::InfoLog(mId, glGetProgramiv, glGetProgramInfoLog);
        glDeleteProgram(mId);
        throw (info_log);
    }

}

Program::~Program() {
    glDetachShader(mId, mVertexShader.GetId());
    glDetachShader(mId, mFragmentShader.GetId());
    glDeleteProgram(mId);
    glDeleteShader(mVertexShader.GetId());
    glDeleteShader(mFragmentShader.GetId());
	for (auto i = mAttributes.begin(); i != mAttributes.end(); i++) {
		delete i;
	}
	mAttributes.clear();
}

const Attribute& Program::CreateAttribute(const string& rName) {
	Attribute* p_attribute = new Attribute(*this, rName);
	mAttributes.push_back(p_attribute);
	return *p_attribute;
}


