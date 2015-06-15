/*
 * elementbuffer.cpp
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#include "elementbuffer.h"
#include "base/object.h"

using namespace engine3d;

ElementBuffer::ElementBuffer(GLenum mode) :
		Buffer(GL_ELEMENT_ARRAY_BUFFER),
		mMode(mode) {
}

ElementBuffer::~ElementBuffer() {
}

void ElementBuffer::Render() const {
	base::Object::Dbg(string("ElementBuffer::Render ") + std::to_string(mElementCount));
	Buffer::Render();
    glDrawElements(
        mMode,
		mElementCount,
        GL_UNSIGNED_SHORT,
        (void*)0
    );
}

void ElementBuffer::SetData(const vector<GLushort>& rIndices) {
	if (rIndices.empty()) {
		return;
	}
	base::Object::Dbg(string("ElementBuffer::SetData ") + std::to_string(rIndices.size() * sizeof(GLushort)));
	mElementCount = rIndices.size();
	Buffer::SetData(mElementCount * sizeof(GLushort), rIndices.data(), GL_STATIC_DRAW);
}

