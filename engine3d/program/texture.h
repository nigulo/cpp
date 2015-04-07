/*
 * Texture.h
 *
 *  Created on: Mar 30, 2015
 *      Author: nigul
 */

#ifndef PROGRAM_TEXTURE_H_
#define PROGRAM_TEXTURE_H_

#include <GL/glew.h>
#include <string>
#include "attribute.h"
#include "program.h"

using namespace std;

class Texture {
public:
	Texture(const Program& rProgram, const string& rName, const string& rTextureFile);
	virtual ~Texture();
	void Use() const;
private:
	const Attribute& mrAttribute;
};

#endif /* PROGRAM_TEXTURE_H_ */
