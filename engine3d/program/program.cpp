/*
 * program.cpp
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#include "program.h"
#include "utils.h"
#include <GL/glew.h>
#include "base/object.h"

using namespace engine3d;
using namespace std;

Program::Program(const string& rVertexShaderScript,
			const string& rFragmentShaderScript) :
		mVertexShader(rVertexShaderScript),
		mFragmentShader(rFragmentShaderScript) {

    mId = glCreateProgram();
	base::Object::Dbg(to_string(mId) + " = glCreateProgram()");

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
		Attribute* p_attribute = i->second;
		delete p_attribute;
	}
	mAttributes.clear();
	for (auto i = mUniforms.begin(); i != mUniforms.end(); i++) {
		Uniform* p_uniform = i->second;
		delete p_uniform;
	}
	mUniforms.clear();
}

const Attribute& Program::GetAttribute(const string& rName) {
	auto i = mAttributes.find(rName);
	if (i != mAttributes.end()) {
		return *(i->second);
	}
	return CreateAttribute(rName);
}

const Attribute& Program::CreateAttribute(const string& rName) {
	Attribute* p_attribute = new Attribute(*this, rName);
	mAttributes.insert({rName, p_attribute});
	return *p_attribute;
}


const Uniform& Program::GetUniform(const string& rName) {
	auto i = mUniforms.find(rName);
	if (i != mUniforms.end()) {
		return *(i->second);
	}
	return CreateUniform(rName);
}

const Uniform& Program::CreateUniform(const string& rName) {
	Uniform* p_uniform = new Uniform(*this, rName);
	mUniforms.insert({rName, p_uniform});
	return *p_uniform;
}
