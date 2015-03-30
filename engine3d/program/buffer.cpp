/*
 * buffer.cpp
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#include "buffer.h"

using namespace engine3d;

Buffer::Buffer(GLenum type) :
		mType(type) {
    glGenBuffers(1, &mId);

}

Buffer::~Buffer() {
	// TODO Auto-generated destructor stub
}

void Buffer::SetData(GLsizeiptr size, const GLvoid* pData, GLenum usage) {
    glBindBuffer(mType, mId);
    glBufferData(mType, size, pData, usage);
}

void Buffer::Render() const {
    glBindBuffer(mType, mId);

}
