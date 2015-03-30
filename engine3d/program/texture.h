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

using namespace std;

class Texture {
public:
	Texture(const string& rTextureFile);
	virtual ~Texture();
	void Use() const;
private:
	GLuint mId;
};

#endif /* PROGRAM_TEXTURE_H_ */
