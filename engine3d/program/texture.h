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
#include "uniform.h"
#include "program.h"

using namespace std;
namespace engine3d {

	class Texture {
	public:
		Texture(Program& rProgram, const string& rName, const string& rTextureFile);
		virtual ~Texture();
		void Use() const;
	private:
		const Uniform& mrUniform;
		GLuint mId;
	};

}
#endif /* PROGRAM_TEXTURE_H_ */
