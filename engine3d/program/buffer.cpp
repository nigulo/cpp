/*
 * buffer.cpp
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#include "buffer.h"
#include "base/object.h"

using namespace engine3d;

Buffer::Buffer(GLenum type) :
		mType(type) {
    glGenBuffers(1, &mId);
	base::Object::Dbg(to_string(mId) + string(" = glGenBuffers(1)"));

}

Buffer::~Buffer() {
}

void Buffer::SetData(GLsizeiptr size, const GLvoid* pData, GLenum usage) {
	base::Object::Dbg(string("glBindBuffer(") + to_string(mType) + ", " + to_string(mId) + ")");
    glBindBuffer(mType, mId);
    base::Object::Dbg(string("glBufferData(") + to_string(mType) + ", " + to_string(size) + ", _, " + to_string(usage) + ")");
    glBufferData(mType, size, pData, usage);
}

void Buffer::Render() const {
	base::Object::Dbg(string("glBindBuffer(") + to_string(mType) + ", " + to_string(mId) + ")");
    glBindBuffer(mType, mId);
}
