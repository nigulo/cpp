/*
 * Texture.cpp
 *
 *  Created on: Mar 30, 2015
 *      Author: nigul
 */

#include "texture.h"
#include "base/object.h"
#include "lodepng/lodepng.h"

using namespace engine3d;

Texture::Texture(Program& rProgram, const string& rName, const string& rTextureFile) :
	mrUniform(rProgram.GetUniform(rName))
{
	// Only png support currently
    vector<unsigned char> image;
    unsigned width, height;
	unsigned error = lodepng::decode(image, width, height, rTextureFile.c_str());
	if (error != 0) {
		throw (string("Loading texture ") + rTextureFile + " failed: " + to_string(error) + ", " + lodepng_error_text(error));
	}

	vector<unsigned char>* p_image = new std::vector<unsigned char>(width * height * 4);
	vector<unsigned char> image2(width * height * 4);
	for(size_t y = 0; y < height; y++) {
		for(size_t x = 0; x < width; x++) {
			for(size_t c = 0; c < 4; c++) {
				(*p_image)[4 * width * y + 4 * x + c] = image[4 * width * y + 4 * x + c];
			}
		}
	}

    glGenTextures(1, &mId);
	base::Object::Dbg(to_string(mId) + string(" = glGenTextures(1)"));
    glBindTexture(GL_TEXTURE_2D, mId);
	base::Object::Dbg(string("glBindTexture(GL_TEXTURE_2D, ") + to_string(mId) + ")");
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);
    glTexImage2D(
        GL_TEXTURE_2D, 0,           /* target, level */
        GL_RGBA,                    /* internal format */
        width, height, 0,           /* width, height, border */
		GL_RGBA, GL_UNSIGNED_BYTE,  /* external format, type */
		p_image->data()             /* pixels */
    );
    //string log("Texture data ");
    //for (unsigned i = 0; i < p_image->size(); i++) {
    //	log += " " + to_string(p_image->data()[i]);
    //}
    //base::Object::Dbg(log);
	if (p_image) {
		p_image->clear();
		delete p_image;
	}
}

Texture::~Texture() {
	// TODO Auto-generated destructor stub
}

void Texture::Use() const {
    glActiveTexture(GL_TEXTURE0); // These 2 are not per texture, need to be moved out from this class
    glUniform1i(mrUniform.GetId(), 0);
    base::Object::Dbg(string("glUniform1i(") + to_string(mrUniform.GetId()) + ", 0)");
    glBindTexture(GL_TEXTURE_2D, mId);
    base::Object::Dbg(string("glBindTexture(GL_TEXTURE_2D, ") + to_string(mId) + ")");
}
