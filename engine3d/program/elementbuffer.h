/*
 * elementbuffer.h
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#ifndef ELEMENTBUFFER_H_
#define ELEMENTBUFFER_H_

#include "buffer.h"
#include <vector>

using namespace std;

namespace engine3d {

class ElementBuffer: public Buffer {
public:
	ElementBuffer(GLenum mode);
	virtual ~ElementBuffer();

	virtual void Render() const;

	GLenum GetMode() const {
		return mMode;
	}

	void SetData(const vector<GLushort>& rIndices);
private:
	GLenum mMode;
	GLsizei mElementCount;
};

} /* namespace engine3d */

#endif /* ELEMENTBUFFER_H_ */
