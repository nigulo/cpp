/*
 * buffer.h
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#ifndef VERTEXBUFFER_H_
#define VERTEXBUFFER_H_

#include "buffer.h"
#include "program.h"
#include "engine3d/meshes/vertex.h"
#include <vector>

namespace engine3d {

class VertexBuffer : public Buffer {
public:
	VertexBuffer(Program* rProgram, int dim = 3, bool textureOrColor = true);
	virtual ~VertexBuffer();

	void SetData(vector<Vertex*> vertices);
	virtual void Render() const;

private:
	Program* mpProgram;
	int mDim;
	Attribute* mpPosition;
	Attribute* mpTexCoord;

};

} /* namespace engine3d */

#endif /* VERTEXBUFFER_H_ */
