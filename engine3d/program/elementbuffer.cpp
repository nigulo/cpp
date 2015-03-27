/*
 * elementbuffer.cpp
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#include "elementbuffer.h"

using namespace engine3d;

ElementBuffer::ElementBuffer(GLenum mode) : Buffer(GL_ELEMENT_ARRAY_BUFFER),
		mMode(mode) {
}

ElementBuffer::~ElementBuffer() {
}

void ElementBuffer::Render() const {
	Buffer::Render();
    glDrawElements(
        mMode,
        mesh->element_count,
        GL_UNSIGNED_SHORT,
        (void*)0
    );
}
