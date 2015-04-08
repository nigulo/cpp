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
#include "attribute.h"
#include "texture.h"
#include <string>
#include <map>

using namespace std;
namespace engine3d {

class Program {
public:
	Program(const string& rVertexShaderScript, const string& rFragmentShaderScript);

	virtual ~Program();

	GLuint GetId() const {
		return mId;
	}

	const Attribute& GetAttribute(const string& rName);

protected:
	const Attribute& CreateAttribute(const string& rName);

private:
	VertexShader mVertexShader;
	FragmentShader mFragmentShader;
	GLuint mId;
	map<string, Attribute*> mAttributes;
};

} /* namespace engine3d */

#endif /* PROGRAM_H_ */
