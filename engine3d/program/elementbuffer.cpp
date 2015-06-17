/*
 * elementbuffer.cpp
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#include "elementbuffer.h"
#include "base/object.h"
#include <iostream>

using namespace engine3d;

ElementBuffer::ElementBuffer(GLenum mode) :
		Buffer(GL_ELEMENT_ARRAY_BUFFER),
		mMode(mode),
		mElementCount(0) {
}

ElementBuffer::~ElementBuffer() {
}

void ElementBuffer::Render() const {
	base::Object::Dbg("ElementBuffer::Render");
	Buffer::Render();
	base::Object::Dbg(string("glDrawElements(") + to_string(mMode) + ", " + to_string(mElementCount) + ", GL_UNSIGNED_SHORT, 0)");
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
	string log = "glElementBuffer::SetData";
	for (auto&& index : rIndices) {
		log = log + " " + to_string(index);
	}
	base::Object::Dbg(log);
	mElementCount = rIndices.size();
	//std::cout << "glElementBuffer vertex_coords: " << (void*)rIndices.data() << " " << (void*)&(rIndices[0]) << std::endl;
	Buffer::SetData(mElementCount * sizeof(GLushort), rIndices.data(), GL_STATIC_DRAW);
}

