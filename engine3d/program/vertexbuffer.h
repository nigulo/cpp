/*
 * buffer.h
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#ifndef VERTEXBUFFER_H_
#define VERTEXBUFFER_H_

#include "buffer.h"

namespace engine3d {

class VertexBuffer : public Buffer {
public:
	VertexBuffer();
	virtual ~VertexBuffer();

	virtual void Render() const;
};

} /* namespace engine3d */

#endif /* VERTEXBUFFER_H_ */
